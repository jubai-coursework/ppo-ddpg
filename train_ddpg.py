import argparse
import os
import numpy as np

from utils.config import DDPGConfig, TrainConfig
from utils.seed import set_seed
from utils.env import make_env, get_env_specs
from utils.logger import EpisodeLogger
from utils.evaluate import evaluate_policy, save_test_result
from utils.plotting import plot_training_curve
from agents.ddpg_agent import DDPGAgent


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
    ddpg_config = DDPGConfig()
    agent = DDPGAgent(
        state_dim, action_dim, ddpg_config,
        action_low=action_low, action_high=action_high,
        device=args.device
    )
    print(f"Device: {agent.device}")

    # Setup output directory
    save_dir = os.path.join(args.save_dir, "ddpg")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Logger
    logger = EpisodeLogger(os.path.join(save_dir, "train_log.csv"))

    # Training variables
    total_steps = 0
    best_reward = float("-inf")
    episode = 0

    print(f"\nStarting DDPG training for {args.episodes} episodes...")
    print(f"Config: buffer_size={ddpg_config.buffer_size}, batch_size={ddpg_config.batch_size}")
    print(f"        actor_lr={ddpg_config.actor_lr}, critic_lr={ddpg_config.critic_lr}")
    print(f"        warmup_steps={ddpg_config.warmup_steps}, noise_std={ddpg_config.noise_std}")

    while episode < args.episodes:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Warmup: use random actions
            if total_steps < ddpg_config.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, training=True, add_noise=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.push_transition(state, action, reward, next_state, float(done))

            # Update after warmup
            if total_steps >= ddpg_config.warmup_steps:
                agent.update()

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

        # Decay exploration noise
        if total_steps >= ddpg_config.warmup_steps:
            agent.decay_noise()

        # Log episode
        logger.log(episode, episode_reward, episode_length, total_steps)
        episode += 1

        # Print progress
        if episode % 10 == 0:
            rewards = logger.get_rewards()
            avg_reward = np.mean(rewards[-10:])
            buffer_size = len(agent.buffer)
            print(f"Episode {episode}/{args.episodes} | Reward: {episode_reward:.2f} | Avg(10): {avg_reward:.2f} | Steps: {total_steps} | Buffer: {buffer_size}")

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
        title="DDPG Training Curve"
    )
    print(f"Training curve saved to {os.path.join(save_dir, 'train_curve.png')}")

    # Final evaluation (no noise)
    print("\nRunning 10-episode evaluation (no noise)...")
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
    parser = argparse.ArgumentParser(description="Train DDPG on LunarLanderContinuous-v3")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--render_test", action="store_true", help="Render during test")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()
    train(args)