from __future__ import annotations
import time
import sys
import signal
import os
import json

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rl_isaaclab.utils.keyboard_listener import KeyboardListener, ThreadSafeValue
from rl_isaaclab.utils.misc import dof_sharpa2isaaclab, dof_isaaclab2sharpa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../utils/python'))
from sharpa import (
    SharpaWaveManager,
    ControlMode,
    ControlSource,
)

if TYPE_CHECKING:
    from .sharpa_wave_deploy_env_cfg import SharpaWaveEnvCfg


class SharpaWaveInhandRotateDeployEnv(gym.Env):
    cfg: SharpaWaveEnvCfg

    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.num_envs = 1
        self.num_hand_dofs = self.cfg.action_space
        self.device = self.cfg.device
        self.num_actions = self.cfg.action_space
        self.observation_space = self.cfg.observation_space

        # change tactile config
        if not self.change_tactile_config(on_board=self.cfg.enable_on_board):
            print(f'Failed to change tactile config')
            exit(1)

        # tactile buffers
        CHANNEL = range(0+5*(1-self.cfg.hand_side), 5+5*(1-self.cfg.hand_side))
        self.frames = {ch: [None, None, None, None] for ch in CHANNEL}
        self.frames_raw = {ch: [None, None, None, None] for ch in CHANNEL}
        # deform mapping
        self.tac_uv_map = [np.load('assets/tactile_ha4_map/tactileSensor_map_4F_point.npy')] * 4
        self.tac_uv_map.append(np.load('assets/tactile_ha4_map/tactileSensor_map_TH_point.npy'))

        # init hand
        self._init_hand()

        # init config
        self.target_dt = 1.0 / self.cfg.control_freq

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # buffers for data
        self.obs_buf_lag_history = torch.zeros((self.num_envs, 80, self.cfg.observation_space//3), device=self.device, dtype=torch.float)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.cfg.prop_hist_len, self.cfg.observation_space//3), device=self.device, dtype=torch.float)

        # joint limits
        self.hand_dof_lower_limits = torch.tensor(
            [-0.1745, -0.1745, 0.0000, -0.1745, -0.1745, -0.3491, -0.3491, -0.1745, -0.3491, -0.3491, 0.0000,
             0.0000, -0.3491, 0.0000, -0.5236, 0.0000, 0.0000, 0.0000, 0.0000, -0.3491, 0.0000, 0.0000],
        device=self.device).reshape(1, -1) * self.cfg.dof_limits_scale
        self.hand_dof_upper_limits = torch.tensor(
            [1.5708, 1.5708, 0.2618, 1.5708, 1.9199, 0.3491, 0.3491, 1.5708, 0.3491, 0.3491, 1.7453, 
             1.7453, 0.3491, 1.7453, 1.3963, 1.3963, 1.3963, 1.7453, 1.3963, 0.3491, 1.3963, 1.7453], 
        device=self.device).reshape(1, -1) * self.cfg.dof_limits_scale

        # contact buffers
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self._contact_body_ids_disable = torch.tensor([], dtype=torch.long)
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)

        # episode length
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._last_step_wall = time.perf_counter()

        # keyboard
        if self.cfg.keyboard_listen:
            self.deploy_state_flag = ThreadSafeValue(0)
            self.calib_tactile_flag = ThreadSafeValue(0)
            self.keyboard_proc = KeyboardListener(self.deploy_state_flag, 
                                                  self.calib_tactile_flag,
                                                  hand_ip=self.hand_info.ip)
            self.keyboard_proc.start()
            signal.signal(signal.SIGINT, self.signal_handler)

    def tactile_callback(self, frame):
        ch = frame['channel']
        img, f6, deform, p= None, None, None, None
        if frame['content'].get("RAW") is not None: img = frame['content']['RAW'].squeeze()
        if frame['content'].get('F6') is not None: f6 = frame['content']['F6']
        if frame['content'].get('CONTACT_POINT') is not None: p = frame['content']['CONTACT_POINT']

        f_norm = torch.norm(torch.tensor(f6)[:3])
        contact_pos = torch.zeros((3), dtype=torch.float32, device=self.device)
        if p is not None:
            p = torch.tensor(p)
            p = p.reshape(-1, 3)
            centroid = p[torch.argmax(p[:, 2])]
            contact_pos = self.tac_uv_map[ch-5*(1-self.cfg.hand_side)][int(centroid[1]), int(centroid[0])]
            contact_pos = torch.tensor(contact_pos[:3]) / 1000.0

        self.frames[ch] = img, f_norm, deform, contact_pos

    def signal_handler(self, sig, frame):
        if self.hand:
            self.hand.stop()
        SharpaWaveManager.get_instance().disconnect_all()
        self.keyboard_proc.stop()
        sys.exit(0)

    def reset(self, seed, options):
        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        # return observations
        return self._get_observations(), None

    def _init_hand(self):
        self.hand = self.auto_detect_hand()
        self.hand_info = self.hand.get_device_info()
        if self.hand is None:
            print("Error: No available device found")
            exit(1)
        print("Sharpa Wave Example - Init Hand Running Mode")
        if not self.initialize():
            print("Error: Failed to initialize hand")
            exit(1)
        custom_config_path = "/root/.sharpa-pilot/config/tactile.json"
        error = self.hand.set_tactile_config_file(custom_config_path)
        if error.code == 0:
            print(f"Custom tactile config set for device: {custom_config_path}")
        else:
            print(f"Failed to set tactile config for device: {error.message}")
        self.hand.start()

    def auto_detect_hand(self):
        """Automatically detect device and return device and device serial number"""
        print("Searching for devices...")
        
        try:
            manager = SharpaWaveManager.get_instance()
            time.sleep(1)  # Wait for 1 seconds for device discovery to complete
            while True:
                devices = manager.get_all_device_sn()
                if not devices:
                    print("No available devices found")
                    time.sleep(1)
                    continue
                else:
                    print(f"Device found: {devices[0]}")
                    return manager.connect(devices[0])
        except Exception as e:
            print(f"Failed to connect to device: {str(e)}")
            exit(1)

    def initialize(self):
        error = self.hand.set_control_mode(ControlMode.POSITION)
        if error.code != 0:
            print(f"Failed to set control mode: {error.message}")
            return False
        error = self.hand.set_speed_coeff(0.3)
        if error.code != 0:
            print(f"Failed to set speed coeff: {error.message}")
            return False

        error = self.hand.set_current_coeff(self.cfg.current_coef)
        if error.code != 0:
            print(f"Failed to set current coeff: {error.message}")
            return False
        error = self.hand.set_control_source(ControlSource.SDK)
        if error.code != 0:
            print(f"Failed to set control source: {error.message}")
            return False
        if not self.init_tactile():
            print(f'Failed to initialize tactile')
            return False
        return True

    def init_tactile(self):
        # set tactile callback
        self.hand.set_tactile_callback(self.tactile_callback)
        return True

    def change_tactile_config(self, on_board=False):
        """Modify tactile.json based on inference device (board or host)."""
        tactile_config_path = '/root/.sharpa-pilot/config/tactile.json'

        try:
            with open(tactile_config_path, 'r') as f:
                config = json.load(f)
            
            side = 'left' if self.cfg.hand_side == 0 else 'right'
            fps = 30 if on_board else 180
            infer_from_device = True if on_board else False
            config['cuda'][side]['fps'] = fps
            config['cuda'][side]['infer_from_device'] = infer_from_device
            print(f"[TactileConfig] Set to {'on-board' if on_board else 'host'} mode: fps={fps}, infer_from_device={infer_from_device} for {side} hand")
            with open(tactile_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except FileNotFoundError:
            print(f"[TactileConfig] Warning: {tactile_config_path} not found, skipping config update")
        except Exception as e:
            print(f"[TactileConfig] Error updating config: {e}")
        return False

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone()
        targets = self.prev_targets + self.cfg.action_scale * self.actions
        self.cur_targets = saturate(targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)

    def _apply_action(self) -> None:
        self._refresh_lab()
        command = dof_isaaclab2sharpa(self.cur_targets.squeeze()).cpu().numpy()
        self.hand.set_joint_position(command)
        self.prev_targets = self.cur_targets.clone()

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        observations = {
            "policy": obs,
            "proprio_hist": self.proprio_hist_buf,
        }
        return observations

    def step(self, action):
        if self.cfg.keyboard_listen:
            # tactile
            if self.calib_tactile_flag.get() == 1:
                self.calib_tactile()
                self.calib_tactile_flag.set(0)
            # deploy state
            deploy_state = self.deploy_state_flag.get()
            if deploy_state == 0:
                self.hand.set_joint_position([0] * 22)
            elif deploy_state == 1:
                freeze_actions = self.hand.get_states().angles
                self.hand.set_joint_position(freeze_actions)
            elif deploy_state == 2:
                self._reset_idx([0])
            elif deploy_state == 3:
                action = action.to(self.device)
                self._pre_physics_step(action)
                self._apply_action()
            else:
                freeze_actions = self.hand.get_states().angles
                self.hand.set_joint_position(freeze_actions)
        else:
            action = action.to(self.device)
            self._pre_physics_step(action)
            self._apply_action()
        loop_dt = time.perf_counter() - self._last_step_wall
        if loop_dt < self.target_dt:
            time.sleep(self.target_dt - loop_dt)
        self._last_step_wall = time.perf_counter()
        self.episode_length_buf += 1
        self.obs_buf = self._get_observations()
        return self.obs_buf, None, None, None, None

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if self.cfg.warm_up:
            error = self.hand.set_speed_coeff(0.3)
            error = self.hand.set_current_coeff(self.cfg.current_coef)
            self.hand.set_joint_position([0] * 22)
            time.sleep(3)

        if self.cfg.keyboard_listen:
            if self.deploy_state_flag.get() == 0: return
            self.deploy_state_flag.set(3)

        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        self.episode_length_buf[env_ids] = 0

        error = self.hand.set_speed_coeff(self.cfg.speed_coef)
        error = self.hand.set_current_coeff(self.cfg.current_coef)

        # reset hand
        if self.cfg.warm_up:
            # replay traj until grasp
            traj = np.load('cache/deploy_init_traj.npy')
            tactile_force, _ = self.get_tactile_info()
            j = 0
            self.hand.set_joint_position(traj[j])
            while torch.max(tactile_force) < 1 and j + 3 < len(traj):
                j += 1
                self.hand.set_joint_position(traj[j])
                time.sleep(0.03)
                tactile_force, _ = self.get_tactile_info()
                
            print(f"pick num {j} traj in {len(traj)}")
            self.prev_targets[env_ids] = dof_sharpa2isaaclab(torch.tensor(traj[j+2], dtype=torch.float32, device=self.device))
            self.cur_targets[env_ids] = dof_sharpa2isaaclab(torch.tensor(traj[j+2], dtype=torch.float32, device=self.device))

        self._refresh_lab()

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _refresh_lab(self):
        self.hand_dof_pos = dof_sharpa2isaaclab(torch.tensor(self.hand.get_states().angles)).reshape(1, -1).to(self.device)

    def compute_observations(self):
        # contact
        sensed_contacts, contact_pos = self.get_tactile_info()
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets.unsqueeze(1)
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.unsqueeze(1), contact_pos.unsqueeze(1)], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:22] = unscale(
            self.hand_dof_pos[at_reset_env_ids], 
            self.hand_dof_lower_limits[at_reset_env_ids],
            self.hand_dof_upper_limits[at_reset_env_ids],
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 22:44] = self.hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 44:49] = sensed_contacts[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 49:64] = contact_pos[at_reset_env_ids].unsqueeze(1)
        self.at_reset_buf[at_reset_env_ids] = 0
        obs_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()

        return obs_buf
    
    def calib_tactile(self):
        # calibrate tactile
        if not self.hand.calib_tactile(): 
            print(f'Tactile calibration failed')
            return False
        else:
            print(f'Tactile calibration successful')
            return True

    def get_tactile_info(self):
        force = torch.zeros(5, dtype=torch.float32, device=self.device)
        contact_pos = torch.zeros((5, 3), dtype=torch.float32, device=self.device)

        if not self.cfg.enable_tactile:
            return force.reshape(1, -1), contact_pos.reshape(1, -1)

        # get tactile data
        for ch in range(5):
            _, f_norm, _, contact_pos_ch = self.frames[4-ch+5*(1-self.cfg.hand_side)]
            force[ch] = f_norm
            contact_pos[ch] = contact_pos_ch
        
        # check disable
        force[self.cfg.disable_tactile_ids] = 0.0
        contact_pos[self.cfg.disable_tactile_ids] = 0.0

        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0

        # check thresh
        force *= self.cfg.force_scale
        force[force < self.cfg.contact_threshold] = 0.0
        if self.cfg.binary_contact:
            force = torch.where(force > self.cfg.contact_threshold, 1.0, 0.0)
        contact_pos[force < self.cfg.contact_threshold] = 0.0

        return force.reshape(1, -1), contact_pos.reshape(1, -1)

    def go_home(self):
        self.hand.set_joint_position([0] * 22)
        time.sleep(3)


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
