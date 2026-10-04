# 配方 03：稀疏 MoE 调参

## 目标
避免路由塌缩，最大化稀疏收益。

## 步骤
1. 调 `moe.num_experts` 与 `moe.top_k`；
2. 观察 `moe/layer.py` 的 `activation_entropy()`；
3. 增大 `balance.py` 的负载均衡损失权重（`pretrain.py` 里 `0.1 * aux`）。

## 验收
各专家激活计数大致均匀，路由熵接近 `log(num_experts)`。
