# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import zmq
import time
import sys
import json
import os
import threading

import gymnasium as gym
import numpy as np
import cv2
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import rl_isaaclab.utils.sharpa_pb2 as pb
from rl_isaaclab.utils.zmq_wrapper import ZmqWrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../utils/python'))
from sharpa import (
    SharpaWaveManager,
    ControlMode,
    ControlSource,
)

if TYPE_CHECKING:
    from .sharpa_wave_deploy_env_cfg import SharpaWaveEnvCfg


class TactileReceiver:
    def __init__(self, addr="tcp://localhost:48006", hand_side=1):
        self.zmq_sub = ZmqWrapper(zmq.SUB, addr)
        self.f6 = {ch: None for ch in range(5)}
        self.deform = {ch: None for ch in range(5)}
        self.hand_side = hand_side
        self.lock = threading.Lock()

    def start(self):
        self.zmq_sub.start_receiver_thread(self.callback)

    def callback(self, msg):
        tactile_msg = pb.Tactile()
        tactile_msg.ParseFromString(msg)
        f6 = [tactile_msg.force6d.force.x, tactile_msg.force6d.force.y, tactile_msg.force6d.force.z, tactile_msg.force6d.torque.x, tactile_msg.force6d.torque.y, tactile_msg.force6d.torque.z]
        deform = np.frombuffer(tactile_msg.deform.data, dtype=np.uint8).reshape(tactile_msg.deform.height, tactile_msg.deform.width)
        with self.lock:
            ch = 4 - (10019 - 10 * self.hand_side - int(tactile_msg.header.key))
            self.f6[ch] = f6
            self.deform[ch] = deform

    def stop(self):
        self.zmq_sub.close()


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
             0.0000, -0.3491, 0.0000, -0.5236, 0.0000, 0.0000, 0.0000, 0.0000, -0.3491, 0.0000, 0.0000],
        device=self.device).reshape(1, -1) * self.cfg.dof_limits_scale
        self.hand_dof_upper_limits = torch.tensor(
            [1.5708, 1.5708, 0.2618, 1.5708, 1.9199, 0.3491, 0.3491, 1.5708, 0.3491, 0.3491, 1.7453, 
             1.7453, 0.3491, 1.7453, 1.3963, 1.3963, 1.3963, 1.7453, 1.3963, 0.3491, 1.3963, 1.7453], 
        device=self.device).reshape(1, -1) * self.cfg.dof_limits_scale

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

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.recorded_joint_pos = torch.zeros((0, self.num_hand_dofs), dtype=torch.float32, device=self.device)

        if not self.cfg.enable_on_board:
            self.tactile_receiver = TactileReceiver("tcp://localhost:48006", hand_side=self.cfg.hand_side)
            self.tactile_receiver.start()

    def reset(self, seed, options):
        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        # return observations
        return self._get_observations(), None

    def _init_hand(self):
        with open("/home/sharpa/.sharpa-pilot/config/tactile.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        data["none"]["left"]["disable"] = not self.cfg.enable_on_board
        data["none"]["right"]["disable"] = not self.cfg.enable_on_board
        with open("/home/sharpa/.sharpa-pilot/config/tactile.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

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
        error = self.hand.set_speed_coeff(0.1)
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
        command = dof_isaaclab2sharpa(self.cur_targets.squeeze()).cpu().numpy()
        if self.cfg.record:
            self.recorded_joint_pos = torch.cat((self.recorded_joint_pos, self.cur_targets), dim=0)
            if self.recorded_joint_pos.shape[0] >= self.cfg.record_length:
                np.save('cache/recorded_joint_pos_traj_20hz.npy', self.recorded_joint_pos.cpu().numpy())
                exit()
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
        action = action.to(self.device)
        self._pre_physics_step(action)
        self._apply_action()
        time.sleep(1/self.cfg.control_freq)
        self.episode_length_buf += 1
        self.obs_buf = self._get_observations()
        return self.obs_buf, None, None, None, None

    def _reset_idx(self, env_ids: Sequence[int] | None):
        error = self.hand.set_speed_coeff(0.1)
        error = self.hand.set_current_coeff(self.cfg.current_coef)
        self.hand.set_joint_position([0] * 22)
        time.sleep(3)

        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        self.episode_length_buf[env_ids] = 0

        # reset hand
        if self.cfg.use_grasp_cache:
            # pose cache
            if self.saved_grasping_states is not None:
                sampled_pose = self.saved_grasping_states[[self.cfg.pose_id]].clone()
            else:
                raise RuntimeError("No saved grasping states found")
            
            dof_pos = sampled_pose[:, :22]
            self.prev_targets[env_ids] = dof_pos.clone()
            self.cur_targets[env_ids] = dof_pos.clone()
            
            init_joint_pos = dof_isaaclab2sharpa(dof_pos.squeeze()).cpu().numpy()
            init_joint_pos[5:] = 0.0
            self.hand.set_joint_position(init_joint_pos)
            time.sleep(3)
            init_joint_pos = dof_isaaclab2sharpa(dof_pos.squeeze()).cpu().numpy()
            self.hand.set_joint_position(init_joint_pos)
            time.sleep(3)
            breakpoint()
        else:
            # replay traj until grasp
            traj = np.load('cache/ini_traj.npy')
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

        error = self.hand.set_speed_coeff(self.cfg.speed_coef)
        error = self.hand.set_current_coeff(self.cfg.current_coef)

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
    
    def largest_connected_component_centroid(self, binary_image):
        img = (binary_image > 0).astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        if num_labels <= 1: return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        centroid = tuple(centroids[largest_idx])
        return centroid

    def get_tactile_info(self):
        force = torch.zeros(5, dtype=torch.float32, device=self.device)
        contact_pos = torch.zeros((5, 3), dtype=torch.float32, device=self.device)
        fill_ch = [None] * 5
        while True:
            if None not in fill_ch: break
            for ch in range(5):
                if self.cfg.enable_on_board:
                    ret = self.hand.fetch_tactile_frame(ch+5*(1-self.cfg.hand_side), timeout=0.1)
                    if ret is None: continue
                    deform_data = ret["content"].get("DEFORM")
                    deform = deform_data.reshape(240, 240).astype(np.uint8)
                    f6_data = torch.tensor(ret["content"].get("F6"))
                else:
                    with self.tactile_receiver.lock:
                        if self.tactile_receiver.f6[ch] is None: continue
                        f6_data = torch.tensor(self.tactile_receiver.f6[ch])
                        deform = self.tactile_receiver.deform[ch].reshape(240, 240).astype(np.uint8)
                force[ch] = torch.norm(f6_data[:3])
                _, binary = cv2.threshold(deform, 30, 255, cv2.THRESH_BINARY)
                # get largest connected component centroid
                center = self.largest_connected_component_centroid(binary.astype(np.uint8))
                if center[0] is not None and center[1] is not None:
                    center_pos_ch = self.tac_uv_map[ch][int(center[0]), int(center[1])]
                    contact_pos[ch] = torch.tensor(center_pos_ch[:3]) / 1000.0
                fill_ch[ch] = True
        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0
        if not self.cfg.enable_tactile:
            force[:] = 0.0
            contact_pos[:] = 0.0
        force = torch.flip(force, dims=[0])
        force[self.cfg.disable_tactile_ids] = 0.0
        force = force.reshape(1, -1)
        force *= self.cfg.force_scale
        force[force < self.cfg.contact_threshold] = 0.0
        if self.cfg.binary_contact:
            force = torch.where(force > self.cfg.contact_threshold, 1.0, 0.0)
        contact_pos = torch.flip(contact_pos, dims=[0])
        contact_pos[self.cfg.disable_tactile_ids, :] = 0.0
        contact_pos = contact_pos.reshape(1, -1)
        print(f'contact force: {force}')
        return force, contact_pos

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

def dof_isaaclab2sharpa(dof_pos):
    return dof_pos[[4, 9, 14, 19, 21, 0, 5, 10, 15, 1, 6, 11, 16, 3, 8, 13, 18, 2, 7, 12, 17, 20]]

def dof_sharpa2isaaclab(dof_pos):
    return dof_pos[[5, 9, 17, 13, 0, 6, 10, 18, 14, 1, 7, 11, 19, 15, 2, 8, 12, 20, 16, 3, 21, 4]]
