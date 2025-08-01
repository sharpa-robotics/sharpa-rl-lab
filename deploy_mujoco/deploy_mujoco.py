import argparse
import os

import yaml
import mujoco
import mujoco.viewer
import numpy as np
import torch

from model import ActorCritic, RunningMeanStd
from utils import _action_hora2mujoco, _obs_mujoco2hora, to_torch, unscale, sharpa_dof_lower, sharpa_dof_upper


class MujocoModel:
    def __init__(self, config):
        self._init_config(config)
        self._init_model()
        self._init_policy()

    def _init_config(self, config):
        config_path = os.path.join(os.path.dirname(__file__), config)
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            return
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = self.config['device']

    def _init_model(self):
        mjcf_path = os.path.join(os.path.dirname(__file__), self.config['model_path'])
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        
        self.dt = self.config['sim']['dt']
        self.model.opt.timestep = self.dt
        self.model.opt.gravity = [0, 0, -9.81]

        mjv_options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(mjv_options)
        mjv_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        mjv_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        mjv_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        self.finger_geom_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'thumb_DP'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'index_DP'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'middle_DP'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ring_DP'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'pinky_DP'),
        ]
        self.cylinder_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'cylinder')
        self.tac_forces = {
            'thumb': np.zeros(6),
            'index': np.zeros(6),
            'middle': np.zeros(6),
            'ring': np.zeros(6),
            'pinky': np.zeros(6),
        }

        self.grasp_cache = np.load(os.path.join(os.path.dirname(__file__), self.config['grasp_cache']))

        self.sharpa_dof_lower = sharpa_dof_lower.to(self.device)
        self.sharpa_dof_upper = sharpa_dof_upper.to(self.device)

    def reset(self):
        chosen_grasp = self.grasp_cache[np.random.randint(0, len(self.grasp_cache))]
        init_joint_pos = _action_hora2mujoco(chosen_grasp[:22])
        init_cylinder_pose = chosen_grasp[22:]
        self.data.qpos[:22] = init_joint_pos
        self.data.qpos[22:25] = init_cylinder_pose[:3]
        self.data.qpos[25] = init_cylinder_pose[6]
        self.data.qpos[26:28] = init_cylinder_pose[3:5]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        obs_buf = torch.zeros((1, 49*3), dtype=torch.float32, device=self.device)
        proprio_hist_buf = torch.zeros((1, 30, 49), dtype=torch.float32, device=self.device)

        obses = to_torch(_obs_mujoco2hora(self.data.qpos[:22]))
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.sharpa_dof_lower, self.sharpa_dof_upper)[None]
        self.get_binary_forces()

        for i in range(3):
            obs_buf[:, i*49+0:i*49+22] = cur_obs_buf.clone()
            obs_buf[:, i*49+22:i*49+44] = prev_target.clone()
            obs_buf[:, i*49+44:i*49+49] = self.binary_forces.clone()

        proprio_hist_buf[:, :, :22] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 22:44] = prev_target.clone()
        proprio_hist_buf[:, :, 44:49] = self.binary_forces.clone()

        return obs_buf, proprio_hist_buf, prev_target, self.data.qpos[:22]

    def _init_policy(self):
        self.device = self.config['device']
        self.policy = ActorCritic({
            'actions_num': 22,
            'input_shape': (147,),
            'actor_units': [512, 256, 128],
            'priv_mlp_units': [256, 128, 8],
            'priv_info': True,
            'proprio_adapt': True,
            'priv_info_dim': 19,
        })
        self.policy.to(self.device)
        self.policy.eval()
        self.running_mean_std = RunningMeanStd((147,)).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((30, 49)).to(self.device)
        self.sa_mean_std.eval()

    def get_binary_forces(self):
        for j, c in enumerate(self.data.contact):
            if c.geom1 in [self.finger_geom_id[0], self.cylinder_geom_id] and c.geom2 in [self.finger_geom_id[0], self.cylinder_geom_id]:
                mujoco.mj_contactForce(self.model, self.data, j, self.tac_forces['thumb'])
            if c.geom1 in [self.finger_geom_id[1], self.cylinder_geom_id] and c.geom2 in [self.finger_geom_id[1], self.cylinder_geom_id]:
                mujoco.mj_contactForce(self.model, self.data, j, self.tac_forces['index'])
            if c.geom1 in [self.finger_geom_id[2], self.cylinder_geom_id] and c.geom2 in [self.finger_geom_id[2], self.cylinder_geom_id]:
                mujoco.mj_contactForce(self.model, self.data, j, self.tac_forces['middle'])
            if c.geom1 in [self.finger_geom_id[3], self.cylinder_geom_id] and c.geom2 in [self.finger_geom_id[3], self.cylinder_geom_id]:
                mujoco.mj_contactForce(self.model, self.data, j, self.tac_forces['ring'])
            if c.geom1 in [self.finger_geom_id[4], self.cylinder_geom_id] and c.geom2 in [self.finger_geom_id[4], self.cylinder_geom_id]:
                mujoco.mj_contactForce(self.model, self.data, j, self.tac_forces['pinky'])

        self.binary_forces = to_torch([
            (np.linalg.norm(self.tac_forces['index'][:3]) > self.config['sim']['tac_force_thresh']) * 0.0, 
            (np.linalg.norm(self.tac_forces['middle'][:3]) > self.config['sim']['tac_force_thresh']) * 1.0, 
            (np.linalg.norm(self.tac_forces['pinky'][:3]) > self.config['sim']['tac_force_thresh']) * 0.0, 
            (np.linalg.norm(self.tac_forces['ring'][:3]) > self.config['sim']['tac_force_thresh']) * 1.0, 
            (np.linalg.norm(self.tac_forces['thumb'][:3]) > self.config['sim']['tac_force_thresh']) * 1.0, 
        ]).reshape(1, -1)

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.policy.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])

    def pd_control(self, tar_q, q, kp, dq, kd):
        torques = (tar_q - q) * kp + dq * kd
        return np.clip(torques, -0.5, 0.5)

    def deploy(self):
        obs_buf, proprio_hist_buf, prev_target, prev_qpos = self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                # reset
                if self.data.qpos[24] < self.config['sim']['reset_height']:
                    obs_buf, proprio_hist_buf, prev_target, prev_qpos = self.reset()

                # deploy policy
                obs = self.running_mean_std(obs_buf.clone())
                action = self.policy.act_inference({
                    'obs': obs,
                    'proprio_hist': self.sa_mean_std(proprio_hist_buf.clone()),
                })
                action = torch.clamp(action, -1.0, 1.0)
                target = prev_target + 1 / 24 * action
                target = torch.clip(target, self.sharpa_dof_lower, self.sharpa_dof_upper)

                commands = target.cpu().numpy()[0]
                commands = _action_hora2mujoco(commands)
                
                for _ in range(self.config['sim']['control_freq_inv']):
                    self.data.ctrl[:22] = self.pd_control(commands,
                                                          self.data.qpos[:22],
                                                          self.config['sim']['kp'],
                                                          (self.data.qpos[:22] - prev_qpos)/self.dt,
                                                          self.config['sim']['kd'])
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    prev_qpos = self.data.qpos[:22]

                obses = to_torch(_obs_mujoco2hora(self.data.qpos[:22]))
                cur_obs_buf = unscale(obses, self.sharpa_dof_lower, self.sharpa_dof_upper)[None]
                prev_obs_buf = obs_buf[:, 49:].clone()
                obs_buf[:, :98] = prev_obs_buf
                obs_buf[:,98:120] = cur_obs_buf.clone()
                obs_buf[:, 120:142] = target.clone()

                self.get_binary_forces()
                obs_buf[:, 142:147] = self.binary_forces

                priv_proprio_buf = proprio_hist_buf[:, 1:30, :].clone()
                cur_proprio_buf = torch.cat([cur_obs_buf, target.clone(), self.binary_forces], dim=-1)[:, None]
                proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_proprio_buf], dim=1)
                
                prev_target = target.clone()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sharpa tactile play")
    parser.add_argument("--config", metavar="config_path", default="configs/config.yml", type=str, help="input config path")
    parser.add_argument("--ckpt", metavar="checkpoint_path", type=str, help="input policy checkpoint path")
    parser.add_argument("--record", action="store_true", help="whether to record frames")
    args = parser.parse_args()

    sharpa_wave = MujocoModel(args.config)
    sharpa_wave.restore(args.ckpt)
    sharpa_wave.deploy()