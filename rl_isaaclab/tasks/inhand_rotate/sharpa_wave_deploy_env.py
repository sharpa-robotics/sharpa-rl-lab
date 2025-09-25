# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import math
import time
import sys
import os

import gymnasium as gym
import numpy as np
import cv2
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

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
        self._init_hand()

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
             0.0000, -0.3491, 0.0000, -0.5236, 0.0000, 0.0000, 0.0000, 0.0000, -0.3491, 0.0000, 0.0000], device=self.device)
        self.hand_dof_upper_limits = torch.tensor(
            [1.5708, 1.5708, 0.2618, 1.5708, 1.9199, 0.3491, 0.3491, 1.5708, 0.3491, 0.3491, 1.7453, 
             1.7453, 0.3491, 1.7453, 1.3963, 1.3963, 1.3963, 1.7453, 1.3963, 0.3491, 1.3963, 1.7453], device=self.device)

        # grasp_cache
        if self.cfg.grasp_cache_path:
            self.saved_grasping_states = torch.from_numpy(np.load(f"{self.cfg.grasp_cache_path}.npy")).float().to(self.device)
        else:
            self.saved_grasping_states = None

        # contact buffers
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self._contact_body_ids_disable = torch.tensor([], dtype=torch.long)
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)

        # deform mapping
        self.tac_uv_map = [np.load('assets/tactile_ha4_map/tactileSensor_map_4F_point.npy')] * 4
        self.tac_uv_map.append(np.load('assets/tactile_ha4_map/tactileSensor_map_TH_point.npy'))

    def _init_hand(self):
        self.hand = self.auto_detect_hand()
        if self.hand is None:
            print("Error: No available device found")
            exit(1)
        print("Sharpa Wave Example - Init Hand Running Mode")
        if not self.initialize():
            print("Error: Failed to initialize hand")
            exit(1)
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
        error = self.hand.set_speed_coeff(self.cfg.speed_coef)
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
        return True

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone()
        targets = self.prev_targets + self.cfg.action_scale * self.actions
        self.cur_targets = saturate(targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)

    def _apply_action(self) -> None:
        self._refresh_lab()
        self.cur_targets = dof_isaaclab2sharpa(self.cur_targets)
        self.hand.set_joint_position(self.cur_targets.squeeze().cpu().numpy())
        self.prev_targets = self.cur_targets.clone()

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        observations = {
            "policy": obs,
            "proprio_hist": self.proprio_hist_buf,
        }
        return observations

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        self.episode_length_buf[env_ids] = 0

        # pose cache
        if self.saved_grasping_states is not None:
            sampled_pose = self.saved_grasping_states[env_ids].clone()
        else:
            raise RuntimeError("No saved grasping states found")

        # reset hand
        dof_pos = sampled_pose[:, :22]

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.hand.set_joint_position(dof_pos.squeeze().cpu().numpy())

        self._refresh_lab()

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _refresh_lab(self):
        self.hand_dof_pos = dof_sharpa2isaaclab(self.hand.get_states().angles)

    def compute_observations(self):
        # contact
        sensed_contacts, contact_pos = self.tactile.get_tactile_info()
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.clone().unsqueeze(1), contact_pos.clone().unsqueeze(1)], dim=-1)
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
    
    def largest_connected_component_centroid(self, binary_image):
        img = (binary_image > 0).astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        if num_labels <= 1: return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        centroid = tuple(centroids[largest_idx])
        return centroid

    def get_tactile_info(self):
        force = [None] * 5
        contact_pos = [None] * 5
        while True:
            if None not in force: break
            for ch in range(5):
                ret = self.hand.fetch_tactile_frame(ch, timeout=0.1)
                if ret is None: continue
                deform_data = ret["content"].get("DEFORM")
                deform = deform_data.reshape(240, 240).astype(np.uint8)
                f6_data = ret["content"].get("F6")
                force[ch] = np.linalg.norm(f6_data[:3])
                _, binary = cv2.threshold(deform, 30, 255, cv2.THRESH_BINARY)
                # get largest connected component centroid
                center = self.largest_connected_component_centroid(binary.astype(np.uint8))
                if center[0] is None or center[1] is None:
                    contact_pos[ch] = np.array([np.nan, np.nan, np.nan])
                else:
                    center_pos_ch = self.tac_uv_map[ch][int(center[0]), int(center[1])]
                    contact_pos[ch] = np.array(center_pos_ch[:3]) / 1000.0
        force.reverse()
        contact_pos.reverse()
        tactile_info = torch.cat((torch.tensor(force), torch.tensor(contact_pos)), dim=-1)
        tactile_info = tactile_info.unsqueeze(0).unsqueeze(0)
        return torch.tensor(force).unsqueeze(0), torch.tensor(contact_pos).unsqueeze(0)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Clamped transform of the tensor. Shape is (N, dims).
    """
    return torch.max(torch.min(x, upper), lower)