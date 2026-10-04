import argparse
import os
import numpy as np

from utils.config import PPOConfig, TrainConfig
from utils.seed import set_seed
from utils.env import make_env, get_env_specs
from utils.logger import EpisodeLogger
from utils.evaluate import evaluate_policy, save_test_result
from utils.plotting import plot_training_curve
from agents.ppo_agent import PPOAgent


def train(args):
    # Set seed
    set_seed(args.seed)

    # Get environment specs
    state_dim, action_dim, action_low, action_high = get_env_specs()
    print(f"Environment: LunarLanderContinuous-v3")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Action range: [{action_low}, {action_high}]")

    # Create environment
    env = make_env(seed=args.seed)

    # Create agent
    ppo_config = PPOConfig()
    agent = PPOAgent(state_dim, action_dim, ppo_config, device=args.device)
    print(f"Device: {agent.device}")

    # Setup output directory
    save_dir = os.path.join(args.save_dir, "ppo")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Logger
    logger = EpisodeLogger(os.path.join(save_dir, "train_log.csv"))

    # Training variables
    total_steps = 0
    best_reward = float("-inf")
    episode = 0

    print(f"\nStarting PPO training for {args.episodes} episodes...")
    print(f"Config: rollout_steps={ppo_config.rollout_steps}, batch_size={ppo_config.batch_size}, lr={ppo_config.lr}")

    while episode < args.episodes:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, float(done), log_prob, value, next_state)

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Update when buffer is full
            if len(agent.buffer) >= ppo_config.rollout_steps:
                # Get last value for incomplete episode
                if not done:
                    _, _, last_value = agent.select_action(state, training=True)
                else:
                    last_value = 0.0
                agent.update(last_value)

        # Log episode
        logger.log(episode, episode_reward, episode_length, total_steps)
        episode += 1

        # Print progress
        if episode % 10 == 0:
            rewards = logger.get_rewards()
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode}/{args.episodes} | Reward: {episode_reward:.2f} | Avg(10): {avg_reward:.2f} | Steps: {total_steps}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(checkpoint_dir, "best.pt"))

    # Save final model
    agent.save(os.path.join(checkpoint_dir, "last.pt"))

    # Save training log
    logger.save()
    print(f"\nTraining completed. Total steps: {total_steps}")

    # Plot training curve
    plot_training_curve(
        os.path.join(save_dir, "train_log.csv"),
        os.path.join(save_dir, "train_curve.png"),
        window=20,
        title="PPO Training Curve"
    )
    print(f"Training curve saved to {os.path.join(save_dir, 'train_curve.png')}")

    # Final evaluation
    print("\nRunning 10-episode evaluation...")
    avg_reward, rewards = evaluate_policy(
        agent,
        make_env,
        episodes=10,
        render=args.render_test
    )
    print(f"Evaluation: Avg Reward = {avg_reward:.2f}")
    print(f"Individual rewards: {[f'{r:.2f}' for r in rewards]}")

    # Save test result
    save_test_result(
        os.path.join(save_dir, "test_result.json"),
        avg_reward,
        rewards,
        episodes=10
    )

    env.close()
    print(f"\nAll outputs saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on LunarLanderContinuous-v3")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--render_test", action="store_true", help="Render during test")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()
    train(args)