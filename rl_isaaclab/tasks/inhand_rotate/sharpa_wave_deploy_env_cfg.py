class SharpaWaveEnvCfg:
    # keyboard listener
    keyboard_listen = True  # If True, policy will not automatically start, 
                            # press 'e' to start, press 'w' to freeze, press 'q' to go home.
                            # Otherwise, policy will start automatically
    # env
    hand_side = 1           # Determine the hand side, 0 for left hand, 1 for right hand.
    action_space = 22
    observation_space = 192
    prop_hist_len = 30
    asymmetric_obs = False
    # control
    warm_up = True          # If True, the hand will warm up before starting the policy.
    control_freq = 20       # Used for modifying the control frequency.
                            # Be aware, it only a time sleep during policy,
                            # which is not exactly the same as the real control frequency.
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    current_coef = 0.3     # Current coefficient applied in hand.
    speed_coef = 0.5       # Speed coefficient applied in hand.
    dof_limits_scale = 0.9 # Multiply a scale to the URDF joint limits,
                           # the hand apply the intersection of this limits and the firmware limits
    # contact
    enable_on_board = None     # Changed in deploy.py
    enable_tactile = True      # If True, the tactile sensor is enabled.
    force_scale = 1/1.5        # Multiply a scale to the output tactile force.
    binary_contact = False     # If True, the output tactile force will be binarized according to the contact_threshold.
    enable_contact_pos = False # Not tested yet. If True, the tactile sensor will output the contact position.
    disable_tactile_ids = []   # Set 0 to according tactile ids.
                               # 0, 1, 2, 3, 4 for thumb, index, middle, ring, pinky finger.
    contact_threshold = 0.2    # If binary_contact is True, this will be used for thresholding.
                               # If binary_contact is False, forces under this threshold will be set to zero, while the rest remains the same.
