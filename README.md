# PPO 与 DDPG 算法实现与对比

分别实现 PPO（on-policy）与 DDPG（off-policy）两种连续控制算法，并在相同环境下对比训练表现。

实验要求、理论推导与结果分析见 [实验报告.md](实验报告.md)。

## 结构

| 路径 | 说明 |
|---|---|
| `agents/ppo_agent.py` | PPO 智能体 |
| `agents/ddpg_agent.py` | DDPG 智能体 |
| `models/actor_critic.py` | Actor-Critic 网络 |
| `models/ddpg_networks.py` | DDPG 的 Actor / Critic 网络 |
| `memory/rollout_buffer.py` | PPO 的轨迹缓冲区 |
| `memory/replay_buffer.py` | DDPG 的经验回放池 |
| `utils/` | 配置、环境封装、评估、日志、绘图与随机种子 |
| `train_ppo.py` / `train_ddpg.py` | 两种算法的训练入口 |
| `train_optimized.py` | 优化后的训练脚本 |
| `compare_plot.py` | 生成两算法对比曲线 |
| `visualize.py` | 训练结果可视化 |
| `outputs/` | 训练日志、曲线、测试结果与模型 checkpoint |

## 运行

```bash
python train_ppo.py        # 训练 PPO
python train_ddpg.py       # 训练 DDPG
python compare_plot.py     # 生成对比曲线
```

训练脚本支持命令行参数，具体可用 `python train_ppo.py --help` 查看。

## 结果

`outputs/` 中已附带训练完成的模型（`checkpoints/best.pt`、`last.pt`）、训练曲线与测试指标（`test_result.json`），以及 `outputs/comparison/comparison_curve.png` 对比曲线，可直接查看结果。
