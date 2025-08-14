import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.inhand_rotate.sharpa_wave_inhand_rotate_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:gym_style_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.inhand_rotate.sharpa_wave_inhand_rotate_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SharpaWavePPORunnerCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:gym_style_ppo_cfg.yaml",
    },
)