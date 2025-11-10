import argparse
import importlib
import os

# add argparse arguments
parser = argparse.ArgumentParser(description="Deploy an RL agent.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--cache", type=str, default=None, help="Cache path.")
parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--device", type=str, default='cuda:0', help="Device to use for training.")
parser.add_argument("--enable_on_board", action="store_true", help="Enable on-board Tactile.")
parser.add_argument("--hand_side", type=int, default=None, help="0 for left hand, 1 for right hand.")
parser.add_argument("--pose_id", type=int, default=0)

args_cli, hydra_args = parser.parse_known_args()

import gymnasium as gym
from omegaconf import OmegaConf
import torch
from datetime import datetime

from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.wrapper.sharpa_wave_deploy_env_wrapper import GymStyleEnvWrapper
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper

import rl_isaaclab.tasks.inhand_rotate

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def parse_entry_point(entry_point: str):
    module, target = entry_point.split(":")

    # case 1: Python class
    if target.endswith("Cfg") or target[0].isupper():
        mod = importlib.import_module(module)
        cfg_class = getattr(mod, target)
        return cfg_class()

    # case 2: YAML config
    if target.endswith(".yaml") or target.endswith(".yml"):
        mod = importlib.import_module(module)
        base_dir = os.path.dirname(mod.__file__)
        config_path = os.path.join(base_dir, target)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return OmegaConf.load(config_path)
    raise ValueError(f"Unsupported entry_point format: {entry_point}")

def custom_task_config(task_id):
    def decorator(func):
        def wrapper(*args, **kwargs):
            spec = gym.spec(task_id)
            env_cfg_entry_point = spec.kwargs.get("env_cfg_entry_point", None)
            agent_cfg_entry_point = spec.kwargs.get("agent_cfg_entry_point", None)
            env_cfg = parse_entry_point(env_cfg_entry_point)
            agent_cfg = parse_entry_point(agent_cfg_entry_point)
            func(env_cfg, agent_cfg)
        return wrapper
    return decorator

@custom_task_config(args_cli.task)
def main(env_cfg, agent_cfg: dict):
    env_cfg.enable_on_board = args_cli.enable_on_board
    env_cfg.hand_side = args_cli.hand_side if args_cli.hand_side is not None else env_cfg.hand_side
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    env_cfg.device = agent_cfg["device"]
    env_cfg.pose_id = args_cli.pose_id
    agent_cfg["algo"] = 'ProprioAdapt'
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)

    # load the checkpoint
    resume_path = agent_cfg["load_path"]
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent.restore_test(resume_path)
    agent.set_eval()
    obs_dict = agent.env.reset()
    while True:
        input_dict = {
            'obs': agent.running_mean_std(obs_dict['obs']),
            'proprio_hist': agent.sa_mean_std(obs_dict['proprio_hist'].detach()),
        }
        mu = agent.model.act_inference(input_dict)
        obs_dict, r, done, info = agent.env.step(mu)


if __name__ == "__main__":
    # run the main function
    main()
