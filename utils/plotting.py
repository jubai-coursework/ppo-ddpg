import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_training_curve(csv_path: str, save_path: str, window: int = 20, title: str = "Training Curve"):
    df = pd.read_csv(csv_path)
    rewards = df["reward"].values
    episodes = df["episode"].values

    moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
    plt.plot(episodes, moving_avg, label=f"Moving Avg (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_comparison_curve(ppo_csv: str, ddpg_csv: str, save_path: str, window: int = 20):
    ppo_df = pd.read_csv(ppo_csv)
    ddpg_df = pd.read_csv(ddpg_csv)

    ppo_rewards = ppo_df["reward"].values
    ddpg_rewards = ddpg_df["reward"].values
    ppo_episodes = ppo_df["episode"].values
    ddpg_episodes = ddpg_df["episode"].values

    ppo_avg = pd.Series(ppo_rewards).rolling(window=window, min_periods=1).mean()
    ddpg_avg = pd.Series(ddpg_rewards).rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(ppo_episodes, ppo_rewards, alpha=0.2, color="blue", label="PPO Reward")
    plt.plot(ppo_episodes, ppo_avg, color="blue", linewidth=2, label=f"PPO Moving Avg (window={window})")
    plt.plot(ddpg_episodes, ddpg_rewards, alpha=0.2, color="red", label="DDPG Reward")
    plt.plot(ddpg_episodes, ddpg_avg, color="red", linewidth=2, label=f"DDPG Moving Avg (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO vs DDPG Training Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()