# 配方 01：诚实问答（防幻觉）

## 目标
让模型对拿不准的问题说「我不知道」，而不是编造。

## 步骤
1. 用 `prompts/honest_qa.txt` 作为提示词；
2. 设置 `merlin.abstain_threshold = 0.5`；
3. 用 `kolibriforge train` 跑 Merlin-Arthur RL 训练；
4. 用 `kolibriforge evaluate` 检查幻觉率是否下降。

## 验收
`honesty.hallucination_rate` 应随训练下降，`abstention.abstain_on_unknown` 应上升。
