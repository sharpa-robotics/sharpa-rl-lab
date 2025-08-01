import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

inhand_task_entry = "isaaclab_rl.task.inhand_rotate"

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-v0",
    entry_point=f"{inhand_task_entry}.sharpa_wave_inhand_rotate_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_hand_env_cfg:AllegroHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
