from .ant_box import AntBoxEnv
from .ant_bridge import AntBridgeEnv 
from .ant_goal import AntGoalEnv
from .ant_mixed import AntMixLongEnv
from .ant_wrappers import DoubleToFloatWrapper


def make_ant_env(env_name, num_envs=1, eval=False):
    """
    create the env environment
    """
    import gym
    from portable.policy.vec_env.vec_monitor import VecMonitor

    assert 'ant' in env_name
    if env_name == "ant_box":
        env = AntBoxEnv(eval=eval)
    elif env_name == "ant_bridge":
        env = AntBridgeEnv(eval=eval)
    elif env_name == "ant_goal":
        env = AntGoalEnv(eval=eval)
    elif env_name == "ant_mixed":
        env = AntMixLongEnv(eval=eval)
    else:
        raise NotImplementedError(f"env_name {env_name} not found")

    venv = gym.vector.AsyncVectorEnv([lambda: env for _ in range(num_envs)])
    venv = DoubleToFloatWrapper(venv)
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return venv
