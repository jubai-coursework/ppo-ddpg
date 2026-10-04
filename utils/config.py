from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    # Round 2: lr=3e-4导致不稳定(avg 206 vs 原229)，用更保守的clip和更大batch补偿
    # clip_eps: 0.15→0.1 (更保守, PDF: 0.1~0.2)
    # batch_size: 128→256 (更大batch, PDF: 64~256)
    # rollout_steps: 4096→2048 (更频繁更新, PDF: 1024~4096)
    lr: float = 3e-4              # 恢复到较高的学习率 (3e-4) 以提升初期探索能力
    gamma: float = 0.99
    gae_lambda: float = 0.95      # PDF: 0.95
    clip_eps: float = 0.2         # 恢复至 OpenAI PPO 默认建议值 (0.2) 以防止陷入局部最优
    update_epochs: int = 8        # PDF范围: 3~10
    rollout_steps: int = 2048     # PDF范围: 1024~4096, 更频繁
    batch_size: int = 256         # PDF范围: 64~256, 更大batch
    entropy_coef: float = 0.005   # PDF范围: 0.001~0.01
    value_coef: float = 0.5       # (不在PDF中，保持不变)
    max_grad_norm: float = 0.5    # (不在PDF中，保持不变)


@dataclass
class DDPGConfig:
    # Round 3: 降gamma + 升actor_lr + 多训练
    # gamma: 0.99→0.98 (更低折扣, PDF: 0.98~0.99)
    # actor_lr: 3e-4→1e-4 (调低以增强后期稳定性)
    # noise_std: 0.2→0.15 (适中噪声, PDF: 0.1~0.3)
    # episodes: 500→1500 (更多训练, PDF: 500~1500)
    actor_lr: float = 1e-4        # PDF范围: 1e-4~3e-4, 调小以增强后期稳定性
    critic_lr: float = 1e-3       # PDF范围: 1e-3~3e-3
    gamma: float = 0.98           # PDF范围: 0.98~0.99, 更低折扣
    tau: float = 0.01             # PDF范围: 0.005~0.01, 稳定目标
    buffer_size: int = 200000     # PDF范围: 50,000~200,000
    batch_size: int = 256         # PDF范围: 64~256
    noise_std: float = 0.1        # 初始噪音标准差
    noise_decay: float = 0.995    # 每episode噪音衰减率 (新增)
    min_noise: float = 0.01       # 最小噪音标准差 (新增)
    warmup_steps: int = 5000      # PDF范围: 1000~5000
    max_grad_norm: float = 1.0    # (不在PDF中，保持不变)


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 1000
    device: str = "auto"
    save_dir: str = "outputs"
    render_test: bool = False