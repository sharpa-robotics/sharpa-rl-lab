# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import math

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import isaaclab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_conjugate, quat_mul, axis_angle_from_quat, saturate, quat_inv

if TYPE_CHECKING:
    from .sharpa_wave_env_cfg import SharpaWaveEnvCfg


class SharpaWaveInhandRotateEnv(DirectRLEnv):
    cfg: SharpaWaveEnvCfg

    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        self.reset_height_lower = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.reset_height_upper = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        self._axes_visualizer = None
        if getattr(self.cfg, 'debug_show_axes', True):
            try:
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import FRAME_MARKER_CFG
                # create frame marker configuration for cylinder
                axes_marker_cfg = FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/CylinderAxes"
                )
                # adjust the axes size based on config (default 0.06 m)
                axes_length = getattr(self.cfg, 'vis_cylinder_axes_length', 0.06)
                axes_marker_cfg.markers["frame"].scale = (axes_length, axes_length, axes_length)
                # create the visualization marker
                self._axes_visualizer = VisualizationMarkers(axes_marker_cfg)
            except Exception as e:
                self._axes_visualizer = None

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # buffers for object
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_pos_prev = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot_prev = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_default_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # buffers for data
        self.obs_buf_lag_history = torch.zeros((self.num_envs, 80, self.cfg.observation_space//3), device=self.device, dtype=torch.float)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.cfg.prop_hist_len, self.cfg.observation_space//3), device=self.device, dtype=torch.float)
        self.priv_info_buf = torch.zeros((self.num_envs, self.cfg.priv_info_dim), device=self.device, dtype=torch.float)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0] * self.cfg.dof_limits_scale
        self.hand_dof_upper_limits = joint_pos_limits[..., 1] * self.cfg.dof_limits_scale

        self.p_gain_default = self.hand.data.default_joint_stiffness[:, self.actuated_dof_indices].clone()
        self.d_gain_default = self.hand.data.default_joint_damping[:, self.actuated_dof_indices].clone()

        self.p_gain = self.p_gain_default.clone()
        self.d_gain = self.d_gain_default.clone()

        if self.cfg.torque_control:
            self.hand.data.default_joint_stiffness = torch.zeros_like(self.p_gain_default, device=self.device)
            self.hand.data.default_joint_damping = torch.zeros_like(self.d_gain_default, device=self.device)
        
            for key, act in self.hand.actuators.items():
                act.stiffness = torch.zeros_like(act.stiffness, device=self.device)
                act.damping = torch.zeros_like(act.damping, device=self.device)

        # grasp_cache
        if self.num_envs % self.cfg.scale_range[2] != 0:
            carb.log_error(f"num_envs must be divisible by scale num: {self.cfg.scale_range[2]}")
            exit()
        scale_ids = torch.linspace(0, self.cfg.scale_range[2]-1, self.cfg.scale_range[2], device=self.device, dtype=torch.int32).reshape(-1, 1)
        scale_ids = scale_ids.repeat(1, math.ceil(self.num_envs/self.cfg.scale_range[2]))
        self.scale_ids = scale_ids.reshape(-1, 1)[:self.num_envs]
        if self.cfg.grasp_cache_path:
            self.saved_grasping_states = torch.from_numpy(np.load(f"{self.cfg.grasp_cache_path}_{self.cfg.scale_range[0]}-{self.cfg.scale_range[1]}-{self.cfg.scale_range[2]}.npy")).float().to(self.device)
            self.bucket_grasp = int(self.saved_grasping_states.shape[0] / self.cfg.scale_range[2])
            self.bucket_env = int(self.num_envs / self.cfg.scale_range[2])
        else:
            self.saved_grasping_states = None

        self.rot_axis = torch.tensor(self.cfg.rot_axis, dtype=torch.float32).repeat(self.num_envs, 1).to(self.device)

        # contact buffers
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self._contact_body_ids_disable = torch.tensor(self.cfg.disable_tactile_ids, dtype=torch.long)
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)
        self.elastomer_ids = [self.hand.body_names.index(body_name) for body_name in 
                              ["right_thumb_elastomer", 
                               "right_index_elastomer", 
                               "right_middle_elastomer",
                               "right_ring_elastomer", 
                               "right_pinky_elastomer"]]

        # randomize
        if self.cfg.randomize_friction:
            rand_friction = torch.empty(self.num_envs).uniform_(self.cfg.randomize_friction_scale_lower, self.cfg.randomize_friction_scale_upper)
            rand_friction = rand_friction.reshape(self.num_envs, 1)
            rand_friction_object = rand_friction.clone() * self.cfg.object_base_friction
            self.set_friction(self.object, rand_friction_object, self.num_envs)
            # IMPORTANT, ELASTOMER MATERIAL IDS
            material_elastomer_ids = [19, 20, 22, 24, 25]
            rand_friction_hand = rand_friction.clone().repeat(1, 26) * self.cfg.metal_base_friction
            rand_friction_hand[:, material_elastomer_ids] = rand_friction_hand[:, material_elastomer_ids] / self.cfg.metal_base_friction * self.cfg.elastomer_base_friction
            self.set_friction(self.hand, rand_friction_hand, self.num_envs)
            self.priv_info_buf[:, 3] = rand_friction.squeeze()
        if self.cfg.randomize_com:
            rand_com = torch.empty([self.num_envs, 3]).uniform_(self.cfg.randomize_com_lower, self.cfg.randomize_com_upper)
            self.set_com(self.object, rand_com, self.num_envs)
            self.priv_info_buf[:, 5:8] = self.object.root_physx_view.get_coms().reshape(self.num_envs, -1)[:, :3]
        if self.cfg.randomize_mass:
            rand_mass = torch.empty(self.num_envs).uniform_(self.cfg.randomize_mass_lower, self.cfg.randomize_mass_upper)
            self.set_mass(self.object, rand_mass, self.num_envs)
            self.priv_info_buf[:, 4] = self.object.root_physx_view.get_masses().reshape(self.num_envs)

        # physics_sim_view
        self.physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # contact sensors
        self._contact_sensor = []
        for id in range(len(self.cfg.contact_sensor)):
            self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
            self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone()
        targets = self.prev_targets + self.cfg.action_scale * self.actions
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            targets,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.object_pos_prev[:] = self.object_pos
        self.object_rot_prev[:] = self.object_rot

        if self.cfg.force_scale > 0.0:
            self.rb_forces *= torch.pow(torch.tensor(self.cfg.force_decay, dtype=torch.float32), self.physics_dt / self.cfg.force_decay_interval)
            # apply new forces
            obj_mass = self.object.root_physx_view.get_masses().reshape(self.num_envs).to(self.device)
            prob = self.cfg.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero().to(self.device)
            self.rb_forces[force_indices, :] = torch.randn(self.rb_forces[force_indices, :].shape, device=self.device) * obj_mass[force_indices, None] * self.cfg.force_scale
            self.object.set_external_force_and_torque(forces=self.rb_forces.reshape(self.num_envs, 1, 3), torques=torch.zeros(self.num_envs, 1, 3, device=self.device))

    def _apply_action(self) -> None:
        self._refresh_lab()
        if self.cfg.torque_control:
            self.torques = self.p_gain * (self.cur_targets - self.hand_dof_pos) - self.d_gain * self.hand_dof_vel
            self.hand.set_joint_effort_target(self.torques[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        else:
            self.hand.set_joint_position_target(self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        observations = {
            "policy": obs,
            "priv_info": self.priv_info_buf,
            "proprio_hist": self.proprio_hist_buf,
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        object_angvel = axis_angle_from_quat(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))) / self.step_dt
        rotate_reward = saturate((object_angvel * self.rot_axis).sum(-1), torch.tensor(self.cfg.angvel_clip_min), torch.tensor(self.cfg.angvel_clip_max))
        object_linvel_penalty = torch.norm(self.object_pos - self.object_pos_prev, p=1, dim=-1) / self.step_dt
        pos_diff_penalty = ((self.hand_dof_pos[:, self.actuated_dof_indices] - self.hand.data.default_joint_pos[:, self.actuated_dof_indices]) ** 2).sum(-1)
        torque_penalty = (self.hand_dof_torque[:, self.actuated_dof_indices] ** 2).sum(-1)
        work_penalty = ((self.hand_dof_torque[:, self.actuated_dof_indices] * self.hand_dof_vel[:, self.actuated_dof_indices]).sum(-1)) ** 2
        object_pos_diff = 1.0 / (torch.norm(self.object_pos - self.object_default_pose.clone()[:, :3], dim=-1) + 0.001)

        total_reward = compute_rewards(
            rotate_reward, self.cfg.rotate_reward_scale,
            object_linvel_penalty, self.cfg.object_linvel_penalty_scale,
            pos_diff_penalty, self.cfg.pos_diff_penalty_scale,
            torque_penalty, self.cfg.torque_penalty_scale,
            work_penalty, self.cfg.work_penalty_scale,
            object_pos_diff, self.cfg.object_pos_reward_scale,
        )

        self.extras["rotate_reward"] = rotate_reward.mean()
        self.extras["object_linvel_penalty"] = object_linvel_penalty.mean()
        self.extras["pos_diff_penalty"] = pos_diff_penalty.mean()
        self.extras["torque_penalty"] = torque_penalty.mean()
        self.extras["work_penalty"] = work_penalty.mean()
        self.extras['object_pos_diff'] = object_pos_diff.mean()
        self.extras['roll'] = object_angvel[:, 0].mean()
        self.extras['pitch'] = object_angvel[:, 1].mean()
        self.extras['yaw'] = object_angvel[:, 2].mean()
        self.extras['gravity_x'] = self.physics_sim_view.get_gravity()[0]
        self.extras['gravity_y'] = self.physics_sim_view.get_gravity()[1]
        self.extras['gravity_z'] = self.physics_sim_view.get_gravity()[2]
        self.extras['total_reward'] = total_reward.mean()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_lab()
        height_reset_upper = self.object_pos[:, 2] > self.reset_height_upper
        height_reset_lower = self.object_pos[:, 2] < self.reset_height_lower
        height_reset = height_reset_upper | height_reset_lower
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras['height_reset_upper'] = height_reset_upper.float().mean()
        self.extras['height_reset_lower'] = height_reset_lower.float().mean()
        self.extras['time_out'] = time_out.float().mean()
        if self.extras['height_reset_upper'] < 5e-4 and self.extras['height_reset_lower'] < 5e-4 and self.cfg.gravity_curriculum and self.common_step_counter > 1000:
            gravity_amp = self.physics_sim_view.get_gravity()
            gravity_amp = torch.sqrt(torch.tensor(gravity_amp[0]**2+gravity_amp[1]**2+gravity_amp[2]**2))
            if gravity_amp < 10: # max gravity set to 10
                new_gravity = carb.Float3(0.0, 0.0, -gravity_amp - 0.05)
                self.physics_sim_view.set_gravity(new_gravity)
                print(f"update gravity: {new_gravity}")
        return height_reset, time_out

    def _rand_pd_scales(self, lower, upper, num_envs, n_dofs):
        rand_scale_s = torch.distributions.Uniform(lower, 1).sample((num_envs, n_dofs)).to(self.device)
        rand_scale_l = torch.distributions.Uniform(1, upper).sample((num_envs, n_dofs)).to(self.device)
        mask_choice = torch.rand((num_envs, n_dofs), device=self.device) > 0.5
        rand_scale = torch.where(mask_choice, rand_scale_s, rand_scale_l)
        return rand_scale

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # pd randomize
        if self.cfg.randomize_pd_gains:
            assert self.cfg.randomize_p_gain_scale_lower <= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_p_gain_scale_upper >= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_d_gain_scale_lower <= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_d_gain_scale_upper >= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            rand_scale = self._rand_pd_scales(self.cfg.randomize_p_gain_scale_lower, self.cfg.randomize_p_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.p_gain[env_ids] = self.p_gain_default[env_ids] * rand_scale
            rand_scale = self._rand_pd_scales(self.cfg.randomize_d_gain_scale_lower, self.cfg.randomize_d_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.d_gain[env_ids] = self.d_gain_default[env_ids] * rand_scale

        # pose cache
        if self.saved_grasping_states is not None:
            sampled_pose_idx = torch.randint(0, self.bucket_grasp, size=(self.bucket_env,))
            saved_grasping_states_picked = torch.zeros((self.num_envs, 29), device=self.device)
            for i in range(self.cfg.scale_range[2]):
                saved_grasping_states_picked[i*self.bucket_env:(i+1)*self.bucket_env] = self.saved_grasping_states[i*self.bucket_grasp:(i+1)*self.bucket_grasp][sampled_pose_idx]
            sampled_pose = saved_grasping_states_picked[env_ids].clone()
        else:
            raise RuntimeError("No saved grasping states found")
        
        if self.cfg.reset_random_quat:
            rotate_center = self.hand.data.default_root_state.clone()[env_ids, :3]
            q_rand = get_random_rotation(env_ids, self.device)
            self.rot_axis[env_ids] = torch.tensor(self.cfg.rot_axis, device=self.device, dtype=torch.float32)
            self.rot_axis[env_ids] = rotate_axis_by_quat(self.rot_axis[env_ids], q_rand)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        # global object positions
        if self.cfg.reset_random_quat:
            _, object_default_pos = apply_random_rotation_with_center(object_default_state[:, 3:7], object_default_state[:, 0:3], rotate_center, q_rand)
            self.object_default_pose[env_ids, :3] = object_default_pos.clone()
            object_default_state[:, 3:7], object_default_state[:, 0:3] = apply_random_rotation_with_center(sampled_pose[:, 25:29], sampled_pose[:, 22:25], rotate_center, q_rand)
            object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        else:
            self.object_default_pose[env_ids, :3] = object_default_state[:, :3].clone()
            object_default_state[:, 0:3] = sampled_pose[:, 22:25] + self.scene.env_origins[env_ids]
            object_default_state[:, 3:7] = sampled_pose[:, 25:29]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.object_default_pose[env_ids, 3:7] = object_default_state[:, 3:7]
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = object_default_state[:, 2] - (self.cfg.reset_height_upper - self.cfg.reset_height_lower) / 2
        self.reset_height_upper[env_ids] = object_default_state[:, 2] + (self.cfg.reset_height_upper - self.cfg.reset_height_lower) / 2

        # reset hand
        hand_default_state = self.hand.data.default_root_state.clone()[env_ids]
        if self.cfg.reset_random_quat:
            hand_default_state[:, 3:7], hand_default_state[:, 0:3] = apply_random_rotation_with_center(hand_default_state[:, 3:7], hand_default_state[:, :3], rotate_center, q_rand)
        hand_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        self.hand.write_root_state_to_sim(hand_default_state, env_ids)
        dof_pos = sampled_pose[:, :22]
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
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(self.num_envs, self.num_fingertips, 3)
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.hand_dof_torque = self.hand.data.applied_torque

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # visualize coordinate axes for cylinder using VisualizationMarkers
        if getattr(self.cfg, 'debug_show_axes', True) and self._axes_visualizer is not None and self.num_envs > 0:
            try:
                # world poses are already with env origins; add back origins for vis API if needed
                cyl_pos_w = self.object.data.root_pos_w
                cyl_quat_w = self.object.data.root_quat_w
                self._axes_visualizer.visualize(translations=cyl_pos_w, orientations=cyl_quat_w)
            except Exception:
                pass

    def compute_observations(self):
        # contact
        tactile_frame_pose = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        tactile_frame_pos = tactile_frame_pose[..., :3]
        tactile_frame_quat = tactile_frame_pose[..., 3:7]
        world_quat = torch.zeros_like(tactile_frame_quat)
        world_quat[..., 0] = 1.0

        net_contact_forces_history = torch.cat([self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2) for id in self._contact_body_ids], dim=2)
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        smooth_contact_forces[:, self._contact_body_ids_disable] = 0.0
        if self.cfg.binary_contact:
            binary_contacts = torch.where(smooth_contact_forces > self.cfg.contact_threshold, 1.0, 0.0)
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + binary_contacts * (1 - latency)
            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.cfg.contact_sensor_noise, 0.0, 1.0)
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
        else:
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + smooth_contact_forces * (1 - latency)
            sensed_contacts = self.last_contacts.clone()

        # contact pos
        not_contact_mask = sensed_contacts < 1.0e-6
        not_contact_mask[:, self._contact_body_ids_disable] = True
        contact_mask = ~not_contact_mask

        contact_pos = torch.cat([self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for id in self._contact_body_ids], dim=1)
        contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
        contact_pos[contact_mask, :] = transform_between_frames(contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :], world_quat[contact_mask, :], tactile_frame_quat[contact_mask, :])
        contact_pos[not_contact_mask, :] = 0.0
        contact_pos = contact_pos.reshape(self.num_envs, -1)
        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0

        if not self.cfg.enable_tactile:
            contact_pos[:] = 0.0
            sensed_contacts[:] = 0.0

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
        self.priv_info_buf[:, 0:3] = self.object_pos - self.object_default_pose[:, :3]

        return obs_buf
    
    def set_friction(self, asset, value, num_envs):
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def set_com(self, asset, value, num_envs):
        coms = asset.root_physx_view.get_coms().clone()
        coms[:, :3] += value
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_coms(coms, env_ids)

    def set_mass(self, asset, value, num_envs):
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_masses(value, env_ids)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def compute_rewards(
    rotate_reward: torch.Tensor, rotate_reward_scale: float,
    object_linvel_penalty: torch.Tensor, object_linvel_penalty_scale: float,
    pos_diff_penalty: torch.Tensor, pos_diff_penalty_scale: float,
    torque_penalty: torch.Tensor, torque_penalty_scale: float,
    work_penalty: torch.Tensor, work_penalty_scale: float,
    object_pos_diff: torch.Tensor, object_pos_reward_scale: float,
):
    reward = rotate_reward * rotate_reward_scale
    reward += object_linvel_penalty * object_linvel_penalty_scale
    reward += pos_diff_penalty * pos_diff_penalty_scale
    reward += torque_penalty * torque_penalty_scale
    reward += work_penalty * work_penalty_scale
    reward += object_pos_diff * object_pos_reward_scale
    return reward

@torch.jit.script
def angle_between_axis_and_z(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """
    quat: (...,4) 格式 [w,x,y,z]
    返回:
        angles: (...,) 与 z 轴夹角，单位弧度 [0, pi]
        对于零旋转，返回 0
    """
    v = axis_angle_from_quat(quat, eps)
    v_norm = torch.linalg.norm(v, dim=-1)
    zero_mask = v_norm <= eps
    safe_norm = torch.clamp(v_norm, min=eps).unsqueeze(-1)
    axis_unit = v / safe_norm
    cos_theta = torch.clamp(axis_unit[..., 2], -1.0, 1.0)
    angle = torch.acos(cos_theta)
    angle = torch.where(zero_mask, torch.zeros_like(angle), angle)
    return angle

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

@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.zeros((B, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

@torch.jit.script
def get_random_rotation(env_ids: torch.Tensor, device: str) -> torch.Tensor:
    N = env_ids.shape[0]

    u1 = torch.rand(N, device=device)
    u2 = torch.rand(N, device=device) * 2.0 * torch.pi
    u3 = torch.rand(N, device=device) * 2.0 * torch.pi
    q1 = torch.sqrt(1.0 - u1) * torch.sin(u2)
    q2 = torch.sqrt(1.0 - u1) * torch.cos(u2)
    q3 = torch.sqrt(u1) * torch.sin(u3)
    q4 = torch.sqrt(u1) * torch.cos(u3)
    q_rand = torch.stack([q4, q1, q2, q3], dim=-1)

    return q_rand

@torch.jit.script
def apply_random_rotation_with_center(
    qs_init: torch.Tensor, pos_init: torch.Tensor, center: torch.Tensor, q_rand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qs_new = quat_mul(q_rand, qs_init)

    R = quat_to_rotmat(q_rand)
    offset = pos_init - center
    new_offset = torch.bmm(R, offset.unsqueeze(-1)).squeeze(-1)
    pos_new = new_offset + center

    return qs_new, pos_new

@torch.jit.script
def rotate_axis_by_quat(axis: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    axis_q = torch.cat([torch.zeros(axis.shape[:-1] + (1,), device=axis.device), axis], dim=-1)
    quat_conj = quat_conjugate(quat)
    rotated_q = quat_mul(quat_mul(quat, axis_q), quat_conj)
    return rotated_q[..., 1:]
