# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
import time

from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .sharpa_wave_grasp_env_cfg import SharpaWaveEnvCfg
from .sharpa_wave_inhand_rotate_env import SharpaWaveInhandRotateEnv


class SharpaWaveInhandRotateGraspEnv(SharpaWaveInhandRotateEnv):
    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.saved_grasping_states = torch.zeros((0, 29), dtype=torch.float32, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        cond1 = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1, p=2).all(-1)
        # force_matrix_w = self._contact_sensor.data.force_matrix_w[:, self._contact_body_ids, 0, :]
        # cond2 = torch.norm(force_matrix_w, dim=-1, p=2).sum(-1) >= 3
        cond3 = torch.less(quat_to_rot(quat_mul(self.object_rot, quat_conjugate(self.object.data.default_root_state.clone()[:, 3:7]))), self.cfg.reset_angle_diff)
        cond = cond1.float() * cond3.float()
        self.reset_buf[cond < 1] = 1
        return 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()
        success = self.episode_length_buf[env_ids] == self.max_episode_length - 1
        all_states = torch.cat([self.hand_dof_pos, self.object_pos, self.object_rot], dim=1)
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])
        print('current cache size:', self.saved_grasping_states.shape[0])
        if len(self.saved_grasping_states) >= 5e4:
            name = f'cache/sharpa_grasp_50k_{time.strftime("%Y%m%d%H%M%S")}.npy'
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

        rand_floats = 2.0 * torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) - 1.0
        
        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        # reset hand
        dof_pos = self.hand.data.default_joint_pos[env_ids] + 0.15 * rand_floats
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
def quat_to_rot(quaternion: torch.Tensor):
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    angle = 2 * torch.acos(quaternion[:, 0])
    return angle
