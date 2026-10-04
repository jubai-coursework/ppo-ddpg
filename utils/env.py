import gymnasium as gym

_state_dim = None
_action_dim = None
_action_low = None
_action_high = None


def make_env(render_mode=None, seed=None):
    env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_env_specs():
    global _state_dim, _action_dim, _action_low, _action_high
    if _state_dim is None:
        env = make_env()
        obs_space = env.observation_space
        act_space = env.action_space
        _state_dim = obs_space.shape[0]
        _action_dim = act_space.shape[0]
        _action_low = float(act_space.low[0])
        _action_high = float(act_space.high[0])
        env.close()
    return _state_dim, _action_dim, _action_low, _action_high