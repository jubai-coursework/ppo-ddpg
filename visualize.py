import os
import numpy as np
import torch

from utils.env import make_env, get_env_specs
from utils.config import PPOConfig, DDPGConfig
from agents.ppo_agent import PPOAgent
from agents.ddpg_agent import DDPGAgent


def load_ppo_agent(checkpoint_path, state_dim, action_dim, device):
    config = PPOConfig()
    agent = PPOAgent(state_dim, action_dim, config, device=device)
    agent.load(checkpoint_path)
    return agent


def load_ddpg_agent(checkpoint_path, state_dim, action_dim, action_low, action_high, device):
    config = DDPGConfig()
    agent = DDPGAgent(state_dim, action_dim, config, action_low=action_low, action_high=action_high, device=device)
    agent.load(checkpoint_path)
    return agent


def visualize(agent, env_fn, episodes=10):
    """可视化智能体表现"""
    env = env_fn(render_mode="human")

    all_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        print(f"\n=== Episode {ep + 1}/{episodes} ===")

        while not done:
            action = agent.select_action(state, training=False)
            if isinstance(action, tuple):
                action = action[0]

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

        all_rewards.append(ep_reward)
        print(f"Episode {ep + 1}: Reward = {ep_reward:.2f}, Steps = {step}")

        if ep_reward >= 200:
            print(">>> 成功着陆! <<<")
        elif ep_reward >= 0:
            print(">>> 部分成功 <<<")
        else:
            print(">>> 坠毁或飞出边界 <<<")

    env.close()

    print(f"\n{'='*40}")
    print(f"总计 {episodes} 回合:")
    print(f"  平均奖励: {np.mean(all_rewards):.2f}")
    print(f"  标准差: {np.std(all_rewards):.2f}")
    print(f"  最高: {np.max(all_rewards):.2f}")
    print(f"  最低: {np.min(all_rewards):.2f}")
    print(f"  成功次数: {sum(1 for r in all_rewards if r >= 200)}/{episodes}")
    print(f"{'='*40}")

    return all_rewards


def find_available_models():
    """查找可用的模型文件"""
    models = {}

    # 查找 PPO 模型
    ppo_best = "outputs/ppo/checkpoints/best.pt"
    ppo_last = "outputs/ppo/checkpoints/last.pt"
    if os.path.exists(ppo_best):
        models["ppo_best"] = ("PPO (最佳模型)", ppo_best)
    if os.path.exists(ppo_last):
        models["ppo_last"] = ("PPO (最终模型)", ppo_last)

    # 查找 DDPG 模型
    ddpg_best = "outputs/ddpg/checkpoints/best.pt"
    ddpg_last = "outputs/ddpg/checkpoints/last.pt"
    if os.path.exists(ddpg_best):
        models["ddpg_best"] = ("DDPG (最佳模型)", ddpg_best)
    if os.path.exists(ddpg_last):
        models["ddpg_last"] = ("DDPG (最终模型)", ddpg_last)

    return models


def main():
    print("\n" + "="*50)
    print("   LunarLanderContinuous 智能体可视化测试")
    print("="*50)

    # 获取环境信息
    state_dim, action_dim, action_low, action_high = get_env_specs()

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n环境: LunarLanderContinuous-v3")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"设备: {device}")

    # 查找可用模型
    models = find_available_models()

    if not models:
        print("\n" + "!"*50)
        print("未找到训练好的模型!")
        print("请先运行训练脚本:")
        print("  python train_ppo.py --episodes 500")
        print("  python train_ddpg.py --episodes 500")
        print("!"*50)
        return

    # 显示选项菜单
    print("\n可用的模型:")
    print("-"*50)
    model_list = list(models.items())
    for i, (key, (name, path)) in enumerate(model_list, 1):
        print(f"  [{i}] {name}")
        print(f"      路径: {path}")

    print(f"  [0] 退出")
    print("-"*50)

    # 用户选择
    while True:
        try:
            choice = input("\n请选择要测试的模型 [0-{}]: ".format(len(model_list)))
            choice = int(choice)

            if choice == 0:
                print("退出程序")
                return

            if 1 <= choice <= len(model_list):
                break
            else:
                print(f"无效选择，请输入 0-{len(model_list)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出程序")
            return

    # 获取选择的模型
    key, (name, checkpoint_path) = model_list[choice - 1]
    algo = "ppo" if "ppo" in key else "ddpg"

    print(f"\n已选择: {name}")
    print(f"加载模型: {checkpoint_path}")

    # 加载智能体
    if algo == "ppo":
        agent = load_ppo_agent(checkpoint_path, state_dim, action_dim, device)
    else:
        agent = load_ddpg_agent(checkpoint_path, state_dim, action_dim, action_low, action_high, device)

    # 选择测试回合数
    while True:
        try:
            episodes_input = input("\n请输入测试回合数 [默认 10]: ").strip()
            if episodes_input == "":
                episodes = 10
            else:
                episodes = int(episodes_input)
            if episodes > 0:
                break
            print("回合数必须大于 0")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出程序")
            return

    print(f"\n开始可视化测试 ({episodes} 回合)...")
    print("提示: 关闭窗口可提前结束测试\n")

    # 运行可视化
    visualize(agent, make_env, episodes=episodes)

    # 询问是否继续测试
    while True:
        try:
            cont = input("\n是否继续测试其他模型? [y/N]: ").strip().lower()
            if cont in ['y', 'yes']:
                main()  # 递归调用重新开始
                return
            else:
                print("感谢使用!")
                return
        except KeyboardInterrupt:
            print("\n退出程序")
            return


if __name__ == "__main__":
    main()
