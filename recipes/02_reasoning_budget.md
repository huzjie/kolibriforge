# 配方 02：推理预算旋钮

## 目标
按任务难度显式控制思考深度，平衡算力与质量。

## 步骤
1. `kolibriforge route "<query>"` 看难度估计与档位；
2. 简单任务用 `--reasoning_level none`，复杂用 `high`；
3. HTTP 调用时传 `reasoning_level` 字段。

## 验收
低档位 token 预算明显更小，同时简单任务准确率不下降。
