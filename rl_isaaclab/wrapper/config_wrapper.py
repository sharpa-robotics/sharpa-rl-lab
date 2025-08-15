class ConfigWrapper:
    def __init__(self, agent_cfg, env_cfg, test=False):
        self.train = agent_cfg
        self.task = env_cfg
        self.test = test
