# Merlin-Arthur 诚实弃答协议（防幻觉）

## 名称由来

交互式证明系统里，Merlin 是拥有无限算力的「证明者」，Arthur 是有限的「验证者」。
借用这个比喻：模型既当 Merlin（生成答案）又当 Arthur（验证可信度），可信度不足
就弃答。

## 三态门控

| 置信度区间 | 动作 | 输出 |
|---|---|---|
| `>= abstain_threshold` | accept | 直接回答 |
| `reject_threshold ~ abstain_threshold` | abstain | "I don't know" |
| `< reject_threshold` | reject | None（安全相关，拒绝） |

## 为什么有效

幻觉的本质是**校准失败**。一个校准良好的模型，其置信度应该约等于正确率。把
置信度显式化、并设置弃答阈值，等于给模型装了一个「我确定吗？」的开关。

## 训练信号（merlin_rl.py）

```
reward = correctness_reward        # 接受且正确
       - hallucination_penalty     # 接受但错误（最重）
       + abstain_reward            # 弃答且确实未知
```

不对称惩罚让策略学会「拿不准就说不知道」。

## 可复用提示词模板

见 `prompts/honest_qa.txt` 与 `prompts/merlin_arthur_system.txt`。
