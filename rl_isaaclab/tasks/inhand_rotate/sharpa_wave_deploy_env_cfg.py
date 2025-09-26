class SharpaWaveEnvCfg:
    # env
    action_space = 22
    observation_space = 192
    prop_hist_len = 30
    asymmetric_obs = False
    # control
    control_freq = 20
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    current_coef = 0.4
    speed_coef = 0.5
    # grasp cache
    grasp_cache_path = 'cache/sharpa_grasp_linspace_0.7-0.8-16'
    # contact
    enable_tactile = True
    binary_contact = True
    enable_contact_pos = False
    contact_threshold = 0.2
