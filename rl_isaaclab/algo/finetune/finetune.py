import os
import time
import torch
import numpy as np
from termcolor import cprint

from rl_isaaclab.utils.misc import AverageScalarMeter, tprint
from rl_isaaclab.algo.models.models import ActorCritic
from rl_isaaclab.algo.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter

from torch.utils.data import Dataset, DataLoader, random_split
import h5py

from tqdm import tqdm

class RotateDataset(Dataset):
    def __init__(self, data_dir, device):
        super().__init__()
        files = os.listdir(data_dir)
        self.data = {}
        for f in files:
            with h5py.File(os.path.join(data_dir, f), 'r') as f:
                for key in f.keys():
                    print(f"Loading {key} from {f.filename}, len{f[key][:].shape}")
                    if key in self.data:
                        self.data[key].append(f[key][:])
                    else:
                        self.data[key] = [f[key][:]]
        self.data = {k: np.concatenate(v, axis=0) for k, v in self.data.items()}
        self.length = self.data['action'].shape[0]
        self.device = device
        for key in self.data:
            print(f"{key} len: {self.data[key].shape}")
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obs = torch.tensor(self.data["obs"][idx], dtype=torch.float32, device=self.device)
        histobs = torch.tensor(self.data["hist_obs"][idx], dtype=torch.float32, device=self.device)
        outputs = torch.tensor(self.data["action"][idx], dtype=torch.float32, device=self.device)
        inputs = {
            'obs': obs,
            'hist_obs': histobs,
        }
        return inputs, outputs


class FineTune(object):
    def __init__(self, output_dir, full_config, dataset_dir, create_output_dir=True):
        self.device = full_config.train['device']
        self.network_config = full_config.train["network"]
        self.ppo_config = full_config.train["algorithm"]
        self.finetune_config = full_config.train['finetune']
        self.epochs = self.finetune_config['num_epochs']
        self.batch_size = self.finetune_config['batch_size']

        # Load data
        self.dataset = RotateDataset(dataset_dir, self.device)
        total_len = len(self.dataset)
        val_len = int(total_len * self.finetune_config['val_split'])
        train_len = total_len - val_len
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader =DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        self.obs_shape = (self.dataset[0][0]['obs'].shape[0],)
        self.actions_num = self.dataset[0][1].shape[0]
        
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.proprio_hist_dim = full_config.task.prop_hist_len

        # ---- Model ----
        net_config = {
            'actor_units': self.network_config["mlp"]["units"],
            'priv_mlp_units': self.network_config["priv_mlp"]["units"],
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': True,
            'priv_info_dim': self.priv_info_dim,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.train()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, self.obs_shape[0]//3)).to(self.device)
        self.sa_mean_std.eval()

        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'finetune_nn')
        self.tb_dir = os.path.join(self.output_dir, 'finetune_tb')
        if create_output_dir:
            os.makedirs(self.nn_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)
            writer = SummaryWriter(self.tb_dir)
            self.writer = writer
        self.direct_info = {}
        # ---- Optim ----
        finetune_params = []
        for name, p in self.model.named_parameters():
            finetune_params.append(p)

        self.optim = torch.optim.Adam(finetune_params, lr=float(self.finetune_config['learning_rate']))
        self.criterion = torch.nn.MSELoss()

    def restore_train(self, fn):
        if fn is None: return
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
        self.model.load_state_dict(checkpoint['model'])

    def train(self):
        _t = time.time()
        _last_t = time.time()

        global_step = 0
        for epoch in range(self.epochs):
            pbar = tqdm(self.train_loader, desc = f'Epoch[{epoch+1}/{self.epochs}]')
            for i, (inputs, outputs) in enumerate(pbar):
                global_step += 1
                input_dict = {
                    'obs':self.running_mean_std(inputs['obs']),
                    'proprio_hist':self.sa_mean_std(inputs['hist_obs']),
                }
                actions = outputs
                mu, _, _, _, _ = self.model._actor_critic(input_dict)
                loss = self.criterion(mu, actions)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if global_step % 100 == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step)
                    val_loss = self.eval()
                    self.writer.add_scalar('val_loss', val_loss)
                    pbar.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})
            self.save(os.path.join(self.nn_dir, f'epoch_{epoch}.pth'))

    def eval(self):
        self.model.eval()
        val_loss = 0.0
        for inputs, outputs in self.val_loader:
            input_dict = {
                'obs': self.running_mean_std(inputs['obs']),
                'proprio_hist': self.sa_mean_std(inputs['hist_obs']),
            }
            actions = outputs
            mu, _, _, _, _ = self.model._actor_critic(input_dict)
            loss = self.criterion(mu, actions)
            val_loss += loss
        return val_loss / len(self.val_loader)
    
    def save(self, fn):
        print(f'Saving model to {fn}')
        torch.save({
            'model': self.model.state_dict(),
            'running_mean_std': self.running_mean_std.state_dict(),
            'sa_mean_std': self.sa_mean_std.state_dict(),
        }, fn)
