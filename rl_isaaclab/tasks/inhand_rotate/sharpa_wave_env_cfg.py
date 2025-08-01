# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class SharpaWaveEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 12
    episode_length_s = 20.0
    action_space = 22
    observation_space = 147  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    clip_obs = 5.0
    clip_actions = 1.0
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608, # 2**23
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"usd/sharpa_wave.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.257551, 0.283045, 0.683330, -0.621782),
            joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.28},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    actuated_joint_names = [
        "index_joint_0",
        "middle_joint_0",
        "ring_joint_0",
        "thumb_joint_0",
        "index_joint_1",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_1",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_1",
        "ring_joint_2",
        "ring_joint_3",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    ]
    fingertip_body_names = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=0.25, replicate_physics=True)
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
