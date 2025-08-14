# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
import time
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

if TYPE_CHECKING:
    from .sharpa_wave_env_grasp_cfg import SharpaWaveEnvCfg
    from .sharpa_wave_inhand_rotate_env import SharpaWaveInhandRotateEnv


class SharpaWaveInhandRotateGraspEnv(SharpaWaveInhandRotateEnv):
    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.saved_grasping_states = torch.zeros((0, 29), dtype=torch.float32, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = torch.zeros_like(actions)
        super()._pre_physics_step(actions)

    def _get_rewards(self) -> torch.Tensor:
        cond1 = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1, p=2).all(-1)
        force_matrix_w = self._contact_sensor.data.force_matrix_w[:, self._contact_body_ids, 0, :]
        cond2 = torch.norm(force_matrix_w, dim=-1, p=2).sum(-1) >= 3
        cond3 = torch.less(quat_to_rot(quat_mul(self.object_rot, quat_conjugate(self.object_init_state[:, 3:7]))), 0.05)
        cond = cond1.float() * cond2.float() * cond3.float()
        self.reset_buf[cond < 1] = 1
        return 0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_lab()
        height_reset_upper = self.object_pos[:, 2] > self.cfg.reset_height_upper
        height_reset_lower = self.object_pos[:, 2] < self.cfg.reset_height_lower
        height_reset = height_reset_upper | height_reset_lower
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return height_reset, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        success = self.episode_length_buf[env_ids] == self.max_episode_length - 1
        all_states = torch.cat([self.hand_dof_pos, self.object_pos], dim=1)
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])
        print('current cache size:', self.saved_grasping_states.shape[0])
        if len(self.saved_grasping_states) >= 5e4:
            name = f'cache/sharpa_grasp_50k_{time.perf_counter()}.npy'
            np.save(name, self.saved_grasping_states[:50000].cpu().numpy())
            exit()

        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        # pd randomize
        if self.cfg.randomize_pd_gains:
            self.p_gain[env_ids] = sample_uniform(self.cfg.randomize_p_gain_lower, self.cfg.randomize_p_gain_upper, (len(env_ids), self.cfg.action_space), device=self.device)
            self.d_gain[env_ids] = sample_uniform(self.cfg.randomize_d_gain_lower, self.cfg.randomize_d_gain_upper, (len(env_ids), self.cfg.action_space), device=self.device)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        # reset hand
        dof_pos = self.hand.data.default_joint_pos[env_ids] + 0.15 * rand_floats[:, 5:5 + self.num_hand_dofs]
        dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids],)
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._refresh_lab()

        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _refresh_lab(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.hand_dof_torque = self.hand.data.computed_torque

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def compute_observations(self):
        # contact
        net_contact_forces_history = self._contact_sensor.data.net_forces_w_history[:, :, self._contact_body_ids, :]
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        binary_contacts = torch.where(smooth_contact_forces > self.cfg.contact_threshold, 1.0, 0.0)
        binary_contacts[:, self._contact_body_ids_disable] = 0.0
        latency_samples = torch.rand_like(self.last_contacts)
        latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
        self.last_contacts = self.last_contacts * latency + binary_contacts * (1 - latency)
        mask = torch.rand_like(self.last_contacts)
        mask = torch.where(mask < self.cfg.contact_sensor_noise, 0.0, 1.0)
        sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)

        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.hand_dof_pos.shape, device=self.device) * 2.0 - 1.0) * self.cfg.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix + self.hand_dof_pos, 
            self.hand_dof_lower_limits, 
            self.hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.clone().unsqueeze(1)], dim=-1)
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
        self.at_reset_buf[at_reset_env_ids] = 0
        obs_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()
        self.priv_info_buf[:, 0:3] = self.object_pos

        return obs_buf
    
    def set_friction(self, asset, value, num_envs):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        value = value.reshape(self.num_envs, 1).repeat(1, materials[..., 0].shape[1])
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def set_com(self, asset, value, num_envs):
        coms = asset.root_physx_view.get_coms().clone()
        coms[:, :3] += value
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_coms(coms, env_ids)
    
    def _setup_reward_config(self):
        self.rot_axis = torch.tensor(self.cfg.rot_axis).repeat(self.num_envs, 1).to(self.device)

    def _joint_idx_gym2lab(self, pos):
        return pos[:, [0, 4, 8, 13, 17, 1, 5, 9, 14, 18, 2, 6, 10, 15, 19, 3, 7, 11, 16, 20, 12, 21]]


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rewards(
    rotate_reward: torch.Tensor, rotate_reward_scale: float,
    object_linvel_penalty: torch.Tensor, object_linvel_penalty_scale: float,
    pos_diff_penalty: torch.Tensor, pos_diff_penalty_scale: float,
    torque_penalty: torch.Tensor, torque_penalty_scale: float,
    work_penalty: torch.Tensor, work_penalty_scale: float,
):
    reward = rotate_reward * rotate_reward_scale
    reward += object_linvel_penalty * object_linvel_penalty_scale
    reward += pos_diff_penalty * pos_diff_penalty_scale
    reward += torque_penalty * torque_penalty_scale
    reward += work_penalty * work_penalty_scale
    return reward

@torch.jit.script
def quat_to_rot(quaternion: torch.Tensor) -> tuple:
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    angle = 2 * torch.acos(quaternion[:, 3])
    return angle

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower