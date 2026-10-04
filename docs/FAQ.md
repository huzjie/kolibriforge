# FAQ

**Q: 这是真的 Kolibri-1 权重吗？**
A: 不是。本仓库是 Kolibri-1 四大技术思想的**零依赖参考实现**，不含真实权重。
真实权重见 Aleph Alpha 官方 Hugging Face（Apache-2.0）。

**Q: 为什么内核零依赖？**
A: 为了在任何 Python 3.9+（含沙箱/托管环境）都能直接跑，训练与基准都可复现。

**Q: Merlin-Arthur 和 RLHF 有什么区别？**
A: RLHF 对齐「人类偏好」，Merlin-Arthur 对齐「诚实弃答」——用不对称奖励让模型
学会拿不准就说「我不知道」，专门治幻觉。

**Q: 推理预算四档能自定义吗？**
A: 能。改 `configs/*.yaml` 的 `budget` 段，或 `reason/budget.py` 的 `LEVEL_COSTS`。
