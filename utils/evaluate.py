import numpy as np
from typing import Tuple, List
import json


def evaluate_policy(agent, env_fn, episodes: int = 10, render: bool = False) -> Tuple[float, List[float]]:
    rewards = []
    render_mode = "human" if render else None
    env = env_fn(render_mode=render_mode)

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            if isinstance(action, tuple):
                action = action[0]
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)

    env.close()
    avg_reward = float(np.mean(rewards))
    return avg_reward, rewards


def save_test_result(save_path: str, avg_reward: float, rewards: List[float], episodes: int = 10):
    result = {
        "avg_reward": avg_reward,
        "episodes": episodes,
        "rewards": rewards,
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards))
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)