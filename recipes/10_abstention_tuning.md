# 配方 10：弃答阈值调优

## 目标
在「过度弃答」（错过能答的）与「幻觉」（答错还自信）之间取平衡。

## 步骤
1. 跑 `bench/honesty.py` 与 `bench/abstention.py` 拿到当前指标；
2. 提高 `abstain_threshold` 减少弃答、但可能增加幻觉；
3. 降低 `abstain_threshold` 更保守、但召回下降。

## 验收
结合业务成本：答错的代价远大于弃答时，选更保守的阈值。
