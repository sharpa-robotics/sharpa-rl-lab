# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rl_isaaclab.utils.modified_events import randomize_rigid_body_scale


@configclass
class EventCfg:
    def rand_params(self, scale_range: list[float, float, int]):
        '''
        Randomize the scale of the object.
        '''
        self.randomize_scale = EventTermCfg(
            func=randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "scale_range": scale_range,
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class SharpaWaveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0 # Episode length in seconds
    action_space = 22
    observation_space = 192
    prop_hist_len = 30      # Proprioception hist frames used in policy
    priv_info_dim = 8
    state_space = 0
    asymmetric_obs = False
    # control
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -0.05),
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
    hand_init_pose = ((0.0, 0.0, 0.5), (0.819152, 0.0, -0.5735764, 0.0))
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"../../../assets/SharpaWave/right_sharpa_wave.usda"),
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
            pos=hand_init_pose[0],
            rot=hand_init_pose[1],
            joint_pos={
                "right_thumb_CMC_FE": math.pi/180 * 95.12771,
                "right_thumb_CMC_AA": math.pi/180 * -3.11244,
                "right_thumb_MCP_FE": math.pi/180 * 14.81626,
                "right_thumb_MCP_AA": math.pi/180 * -1.03493,
                "right_thumb_IP": math.pi/180 * 12.23986,
                "right_index_MCP_FE": math.pi/180 * 65.21091, 
                "right_index_MCP_AA": math.pi/180 * 6.1133,
                "right_index_PIP": math.pi/180 * 15.58495,
                "right_index_DIP": math.pi/180 * 5.90325,
                "right_middle_MCP_FE": math.pi/180 * 31.74149,
                "right_middle_MCP_AA": math.pi/180 * -0.95812,
                "right_middle_PIP": math.pi/180 * 41.88173,
                "right_middle_DIP": math.pi/180 * 12.844,
                "right_ring_MCP_FE": math.pi/180 * 31.72383,
                "right_ring_MCP_AA": math.pi/180 * 9.84458,
                "right_ring_PIP": math.pi/180 * 35.22366,
                "right_ring_DIP": math.pi/180 * 18.02839,
                "right_pinky_CMC": math.pi/180 * 10.9712,
                "right_pinky_MCP_FE": math.pi/180 * 68.30895,
                "right_pinky_MCP_AA": math.pi/180 * 7.99151,
                "right_pinky_PIP": math.pi/180 * 5.89626,
                "right_pinky_DIP": math.pi/180 * 5.89875,
            },
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    contact_sensor = [
        # elastomer
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_thumb_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_index_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ring_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_pinky_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        # DP
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_thumb_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_index_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ring_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_pinky_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        )
    ]

    actuated_joint_names = [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ]
    fingertip_body_names = [
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"../../../assets/cylinder/cylinder.usd"),
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
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.09559, -0.00517, 0.61906), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=0.75, replicate_physics=False)
    # event
    events: EventCfg = EventCfg()
    # reset
    reset_height_lower = 0.59906
    reset_height_upper = 0.63906
    reset_angle_diff = 45 / 180 * math.pi # Not used.
    reset_random_quat = False             # If True, the quaternion of the hand and object is randomized.
    # reward
    rot_axis = (0, 0, 1)                  # Rotation axis used in reward function.
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5                 
    rotate_reward_scale = 2.5            
    object_linvel_penalty_scale = -0.3
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    object_pos_reward_scale = 0.003
    # grasp cache
    grasp_cache_path = 'cache/sharpa_grasp_linspace' # Grasp cache used in training.
    # noise
    joint_noise_scale = 0.02
    # contact
    enable_tactile = True       # If True, the tactile sensor is enabled.
    binary_contact = False      # If True, the output tactile force will be binarized according to the contact_threshold.
    enable_contact_pos = False  # Not tested yet. If True, the tactile sensor will output the contact position.
    disable_tactile_ids = []    # Set 0 to according tactile ids.
                                # 0, 1, 2, 3, 4 are thumb, index, middle, ring, pinky finger, respectively.
    contact_smooth = 0.5        # Smoothing factor for tactile force.
    contact_threshold = 0.05    # Binary contact force threshold, only used when binary_contact is True.
    contact_latency = 0.005     # Contact latency.
    contact_sensor_noise = 0.01 # Contact sensor noise, only used when binary_contact is True.
    # align real
    dof_limits_scale = 0.9      # Multiply a scale to the URDF joint limits.
    # randomize
    scale_range = [0.5, 0.5, 1] # Scale size of the object, [lower, upper, num].
    events.rand_params(scale_range)
    randomize_pd_gains = True   # Randomize PD gains.
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    randomize_friction = True   # Randomize friction.
    randomize_friction_scale_lower = 0.5
    randomize_friction_scale_upper = 2.0
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    randomize_com = True        # Randomize center of mass.
    randomize_com_lower = -0.01
    randomize_com_upper = 0.01
    randomize_mass = True       # Randomize object mass.
    randomize_mass_lower = 0.01
    randomize_mass_upper = 0.25
    # random forces applied to the object
    force_scale = 2
    random_force_prob_scalar = 0.25
    force_decay = 0.9
    force_decay_interval = 0.08
    # curriculum
    gravity_curriculum = True # If True, gravity is gradually increased during training, upper limits is 10m/s^2.
    # debug
    debug_show_axes = False   # If True, visualize the coordinate axes of the object.
