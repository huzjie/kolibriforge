# 训练

三条训练路径：

| 路径 | 入口 | 目标 |
|---|---|---|
| `pretrain_moe` | `train/pretrain.py` | 下一 token + 负载均衡 |
| `sft_honesty` | `train/sft.py` | 直接监督校准 skill |
| `MerlinRLTrainer` | `train/merlin_rl.py` | RL 塑造诚实弃答行为 |

## skill 单调趋 1

mock 后端的 `train_step` 让 skill **单调上升**（对错都升，正确升得更快），避免
「对+1 错-1」在噪声下把 skill 压到 0 的问题。
