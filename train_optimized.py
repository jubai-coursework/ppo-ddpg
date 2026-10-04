"""
超参数优化训练脚本
运行优化后的超参数进行训练，并与之前结果对比
"""
import os
import subprocess
import json
from datetime import datetime


def run_training(algo, episodes=1000, seed=42):
    """运行训练脚本"""
    print(f"\n{'='*60}")
    print(f"开始训练 {algo.upper()} (优化后超参数)")
    print(f"Episodes: {episodes}, Seed: {seed}")
    print(f"{'='*60}\n")

    cmd = [
        "python", f"train_{algo}.py",
        "--episodes", str(episodes),
        "--seed", str(seed)
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def load_result(algo):
    """加载测试结果"""
    path = f"outputs/{algo}/test_result.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def print_comparison():
    """打印对比结果"""
    print("\n" + "="*70)
    print("                    训练结果对比 (优化后超参数)")
    print("="*70)

    results = {}
    for algo in ["ppo", "ddpg"]:
        result = load_result(algo)
        if result:
            results[algo] = result

    if not results:
        print("未找到训练结果")
        return

    # 表头
    print(f"{'指标':<20} {'PPO':>15} {'DDPG':>15} {'目标':>15}")
    print("-"*70)

    # 数据
    metrics = [
        ("平均奖励", "avg_reward", "≥200"),
        ("标准差", "std_reward", "越低越好"),
        ("最高奖励", "max_reward", "-"),
        ("最低奖励", "min_reward", "-"),
    ]

    for name, key, target in metrics:
        ppo_val = results.get("ppo", {}).get(key, "N/A")
        ddpg_val = results.get("ddpg", {}).get(key, "N/A")

        if isinstance(ppo_val, float):
            ppo_val = f"{ppo_val:.2f}"
        if isinstance(ddpg_val, float):
            ddpg_val = f"{ddpg_val:.2f}"

        print(f"{name:<20} {ppo_val:>15} {ddpg_val:>15} {target:>15}")

    print("-"*70)

    # 成功率
    for algo, result in results.items():
        rewards = result.get("rewards", [])
        success = sum(1 for r in rewards if r >= 200)
        print(f"{algo.upper()} 成功着陆: {success}/10 回合")

    print("="*70)

    # 判断是否达标
    print("\n训练目标评估:")
    for algo, result in results.items():
        avg = result.get("avg_reward", 0)
        if avg >= 200:
            print(f"  ✓ {algo.upper()}: 达标 (平均奖励 {avg:.2f} ≥ 200)")
        else:
            print(f"  ✗ {algo.upper()}: 未达标 (平均奖励 {avg:.2f} < 200)")


def main():
    print("\n" + "="*70)
    print("         PPO/DDPG 超参数优化训练")
    print("="*70)
    print("\n优化内容:")
    print("  PPO:")
    print("    - 降低学习率 (3e-4 → 1e-4) 提高稳定性")
    print("    - 增加 rollout (2048 → 4096) 收集更多数据")
    print("    - 降低熵系数 (0.01 → 0.005) 减少随机探索")
    print("    - 增大 batch size (64 → 128)")
    print("  DDPG:")
    print("    - 降低学习率提高稳定性")
    print("    - 增大回放池和 batch size")
    print("    - 降低噪声和软更新率")
    print("="*70)

    choice = input("\n是否开始训练? [Y/n]: ").strip().lower()
    if choice in ['n', 'no']:
        print("取消训练")
        return

    # 训练回合数
    episodes_input = input("训练回合数 [默认 1000]: ").strip()
    episodes = int(episodes_input) if episodes_input else 1000

    # 运行训练
    for algo in ["ppo", "ddpg"]:
        success = run_training(algo, episodes)
        if not success:
            print(f"\n{algo.upper()} 训练失败!")
            return

    # 打印对比结果
    print_comparison()

    # 生成对比图
    print("\n生成对比曲线...")
    os.system("python compare_plot.py")

    print("\n训练完成! 可运行以下命令查看可视化:")
    print("  python visualize.py")


if __name__ == "__main__":
    main()
