# 架构设计

kolibriforge 由四个正交的技术主线 + 一个编排层组成。

## 数据流

1. **分词**：`tokenizer/unibpe.py` 先做复合词预切分，再字节级 BPE。
2. **预算路由**：`reason/budget.py` 估计难度，映射到四个思考深度。
3. **生成**：`backends/*` 产出候选答案与置信度。
4. **诚实门控**：`merlin/protocol.py` 用 Arthur 校验 + 门控决定 accept/abstain/reject。

## 关键设计决策

- **零依赖内核**：`core/` 手写张量与 autodiff，保证托管/沙箱 Python 也能跑。
- **确定性 mock**：`utils/stable.py` 用 md5 前 8 字节做种子，训练单调、基准可复现。
- **可训练 skill**：mock 后端的 `skill` 标量真正参与打分，训练才有意义。
- **配置回退**：`utils/yamlish.py` 提供极简 YAML 子集解析，无 PyYAML 也能读 `.yaml`。
