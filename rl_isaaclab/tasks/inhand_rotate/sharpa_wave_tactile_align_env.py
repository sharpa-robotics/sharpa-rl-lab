# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import time

import numpy as np
import torch
from collections.abc import Sequence

from isaaclab.utils.math import saturate, quat_inv, quat_mul

from .sharpa_wave_tactile_align_env_cfg import SharpaWaveEnvCfg
from .sharpa_wave_env import SharpaWaveInhandRotateEnv


class SharpaWaveInhandRotateTactileAlignEnv(SharpaWaveInhandRotateEnv):
    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_joint_id = 0
        self.action_sequence_id = 0
        self.total_round = 0
        self.elastomer_ids = [self.hand.body_names.index(body_name) for body_name in 
                              ["right_thumb_elastomer", 
                               "right_index_elastomer", 
                               "right_middle_elastomer",
                               "right_ring_elastomer", 
                               "right_pinky_elastomer"]]
        self.force_collect = torch.zeros((self.num_envs, 0, 5, 6), dtype=torch.float32, device=self.device)
        self.pos_diff = torch.zeros((self.num_envs, 0, 22), dtype=torch.float32, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        # contact
        net_contact_forces = torch.cat([self._contact_sensor[id].data.net_forces_w_history[:, 0, 0, :].unsqueeze(1) for id in self._contact_body_ids], dim=1)
        # contact pos
        tactile_frame_pose = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        tactile_frame_pos = tactile_frame_pose[..., :3]
        tactile_frame_quat = tactile_frame_pose[..., 3:7]
        contact_pos = torch.cat([self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for id in self._contact_body_ids], dim=1)

        not_contact_mask = torch.norm(net_contact_forces, dim=-1) < 1.0e-6
        contact_mask = ~not_contact_mask
        contact_pos[not_contact_mask, :] = torch.nan

        world_quat = torch.zeros_like(tactile_frame_quat)
        world_quat[..., 0] = 1.0

        contact_pos[contact_mask, :] = transform_between_frames(contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :], world_quat[contact_mask, :], tactile_frame_quat[contact_mask, :])
        net_contact_forces = transform_between_frames(net_contact_forces, world_quat, tactile_frame_quat)
        
        collect_data = torch.cat([net_contact_forces, contact_pos], dim=-1).unsqueeze(1)
        self.force_collect = torch.cat([self.force_collect, collect_data], dim=1)
        pos_diff = (self.cur_targets - self.hand_dof_pos).unsqueeze(1)
        self.pos_diff = torch.cat([self.pos_diff, pos_diff], dim=1)

        if self.total_round == 1:
            print("collect done")
            np.save("cache/sharpa_tactile_align_tactile_info.npy", self.force_collect.cpu().numpy())
            np.save("cache/sharpa_tactile_align_pos_diff.npy", self.pos_diff.cpu().numpy())
            exit()
        return 0
    
    def _get_dones(self):
        self._refresh_lab()
        reset_empty = torch.zeros(self.num_envs)
        reset_full = torch.ones(self.num_envs)
        self.action_sequence_id += 1
        print(f"collect force data: {self.force_collect.shape}")
        if self.action_sequence_id == len(self.cfg.action_sequence):
            self.action_sequence_id = 0
            self.action_joint_id += 1
        if self.action_joint_id == len(self.cfg.action_sequence_joint):
            self.action_joint_id = 0
            self.total_round += 1
            return reset_empty, reset_full
        else:
            return reset_empty, reset_empty
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        print(self.hand.joint_names)
        super()._pre_physics_step(actions)
        targets = torch.zeros_like(self.cur_targets)
        if self.cfg.action_sequence_joint[self.action_joint_id] == 'right_thumb_CMC_FE':
            targets[:, self.hand.joint_names.index('right_thumb_CMC_AA')] = self.cfg.action_sequence[self.action_sequence_id]
        targets[:, self.hand.joint_names.index(self.cfg.action_sequence_joint[self.action_joint_id])] = self.cfg.action_sequence[self.action_sequence_id]

        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            targets / 180 * torch.pi,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()

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

        rand_floats = 2.0 * torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) - 1.0
        
        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        # reset hand
        dof_pos = self.hand.data.default_joint_pos[env_ids] + 0.02 * rand_floats
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
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v about the rotation described by quaternion(s) q.

    Args:
        q: Quaternion(s) in (w, x, y, z). Shape (..., 4).
        v: Vector(s). Shape (..., 3).

    Returns:
        Rotated vector(s). Shape (..., 3).
    """
    # make v into pure quaternion (0, v)
    zeros = torch.zeros_like(v[..., :1])
    v_as_quat = torch.cat([zeros, v], dim=-1)  # (..., 4)
    # rotate: q * v * q^-1
    v_rot = quat_mul(quat_mul(q, v_as_quat), quat_inv(q))
    return v_rot[..., 1:]  # drop scalar part


@torch.jit.script
def transform_between_frames(p_A: torch.Tensor, q_A: torch.Tensor,
                             q_B: torch.Tensor) -> torch.Tensor:
    """Transform a point from frame A to frame B (rotation only).

    Args:
        p_A: Point(s) in frame A, shape (..., 3).
        q_A: Quaternion of frame A in world, shape (..., 4).
        q_B: Quaternion of frame B in world, shape (..., 4).

    Returns:
        Point(s) in frame B, shape (..., 3).
    """
    # p in world frame
    p_world = quat_rotate(q_A, p_A)
    # p in B frame
    p_B = quat_rotate(quat_inv(q_B), p_world)
    return p_B