class SharpaWaveEnvCfg:
    # record joint pos
    auto_record = False
    auto_record_length = 610
    # record state-action
    record_state_action = True
    # env
    hand_side = 1
    action_space = 22
    observation_space = 192
    prop_hist_len = 30
    asymmetric_obs = False
    # control
    control_freq = 20
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    current_coef = 0.3
    speed_coef = 0.5
    dof_limits_scale = 0.9
    # grasp cache
    use_grasp_cache = False
    grasp_cache_path = 'cache/sharpa_grasp_linspace_0.4-0.6-8'
    # contact
    enable_on_board = True
    enable_tactile = True
    force_scale = 1/1.5
    binary_contact = False
    enable_contact_pos = False
    disable_tactile_ids = []
    contact_threshold = 0.2
