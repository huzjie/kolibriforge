# 配方 12：负载均衡调优

1. 跑 `examples/02_moe_forward.py` 看激活分布与路由熵；
2. 若少数专家垄断（熵 << log(num_experts)），调大 `pretrain.py` 里
   `total_loss = ce + 0.1 * aux` 的 `0.1`；
3. 重跑观察 `activation_counts` 是否更均匀。
