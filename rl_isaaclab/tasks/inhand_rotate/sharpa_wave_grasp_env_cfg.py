# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class SharpaWaveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12
    action_space = 22
    observation_space = 147
    prop_hist_len = 30
    priv_info_dim = 8
    state_space = 0
    asymmetric_obs = False
    # control
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    pgain = 60
    dgain = 4
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
            gpu_max_rigid_patch_count=5*2**18
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/renrenyuan/sharpa_tac_rl/assets/sharpa_ha4/ha4.usd",
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
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002, 
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.9063078, -0.4226183, 0, 0),
            joint_pos={
                "thumb_CMC_FE": 1.918,
                "thumb_CMC_AA": -0.3489,
                "thumb_MCP_FE": 0.1484,
                "thumb_MCP_AA": -0.1745,
                "thumb_IP": 0.4835,
                "index_MCP_FE": 0.7191, 
                "index_MCP_AA": -0.1676,
                "index_PIP": 0.3840,
                "index_DIP": 0.7505,
                "middle_MCP_FE": 0.1623,
                "middle_MCP_AA": -0.0349,
                "middle_PIP": 0.8203,
                "middle_DIP": 0.3630,
                "ring_MCP_FE": 0.1937,
                "ring_MCP_AA": 0.1745,
                "ring_PIP": 0.7994,
                "ring_DIP": 0.3316,
                "pinky_CMC": 0.0,
                "pinky_MCP_FE": 0.6109,
                "pinky_MCP_AA": 0.2217,
                "pinky_PIP": 0.5585,
                "pinky_DIP": 0.6458,
            },
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.5,
                stiffness=0.0,
                damping=0.0,
                friction=0.1,
                armature=0.1,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_DP",
        history_length=3,
        force_threshold=0.01,
        filter_prim_paths_expr=["/World/envs/env_.*/object"],
    )

    actuated_joint_names = [
        "thumb_CMC_FE",
        "thumb_CMC_AA",
        "thumb_MCP_FE",
        "thumb_MCP_AA",
        "thumb_IP",
        "index_MCP_FE",
        "index_MCP_AA",
        "index_PIP",
        "index_DIP",
        "middle_MCP_FE",
        "middle_MCP_AA",
        "middle_PIP",
        "middle_DIP",
        "ring_MCP_FE",
        "ring_MCP_AA",
        "ring_PIP",
        "ring_DIP",
        "pinky_CMC",
        "pinky_MCP_FE",
        "pinky_MCP_AA",
        "pinky_PIP",
        "pinky_DIP",
    ]
    fingertip_body_names = [
        "thumb_DP",
        "index_DP",
        "middle_DP",
        "ring_DP",
        "pinky_DP",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/renrenyuan/sharpa_tac_rl/assets/cylinder/cylinder.usd",
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
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002, 
                rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            scale=(0.75, 0.75, 0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.003, 0.054, 0.635), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=0.75, replicate_physics=True)
    # reset
    reset_height_lower = 0.63
    reset_height_upper = 0.64
    reset_angle_diff = 0.05
    rot_axis = (0, 0, 1)
    # grasp cache
    grasp_cache_path = None
    # noise
    joint_noise_scale = 0.02
    # contact
    contact_smooth = 0.5
    contact_threshold = 0.2
    contact_latency = 0.005
    contact_sensor_noise = 0.01
    # randomize
    randomize_pd_gains = False
    randomize_p_gain_lower = 40
    randomize_p_gain_upper = 100
    randomize_d_gain_lower = 3
    randomize_d_gain_upper = 5
    randomize_friction = False
    randomize_friction_lower = 0.3
    randomize_friction_upper = 3.0
    randomize_com = False
    randomize_com_lower = -0.01
    randomize_com_upper = 0.01
    randomize_mass = True
    randomize_mass_lower = 0.05
    randomize_mass_upper = 0.051
    # random forces applied to the object
    force_scale = 0.0
    random_force_prob_scalar = 0.0
    force_decay = 0.9
    force_decay_interval = 0.08
