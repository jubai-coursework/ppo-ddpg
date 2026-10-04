import os
from utils.plotting import plot_comparison_curve


def main():
    ppo_csv = "outputs/ppo/train_log.csv"
    ddpg_csv = "outputs/ddpg/train_log.csv"
    save_path = "outputs/comparison/comparison_curve.png"

    if not os.path.exists(ppo_csv):
        print(f"Error: PPO training log not found at {ppo_csv}")
        print("Please run train_ppo.py first.")
        return

    if not os.path.exists(ddpg_csv):
        print(f"Error: DDPG training log not found at {ddpg_csv}")
        print("Please run train_ddpg.py first.")
        return

    plot_comparison_curve(ppo_csv, ddpg_csv, save_path, window=20)
    print(f"Comparison curve saved to {save_path}")


if __name__ == "__main__":
    main()