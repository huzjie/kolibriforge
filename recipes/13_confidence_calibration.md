# 配方 13：置信度校准

```python
from kolibriforge.merlin.honesty import expected_calibration_error
ece = expected_calibration_error([0.9, 0.6, 0.4], [1, 0, 0])
print(ece)
```
ECE 越接近 0 越校准良好；过大说明要调 abstain_threshold 或温度缩放。
