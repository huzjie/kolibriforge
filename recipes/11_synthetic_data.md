# 配方 11：合成数据生成

```python
from kolibriforge.data.synth import honesty_triplets, math_pairs, SynthCorpus
from kolibriforge.data.pipeline import DataPipeline

triplets = honesty_triplets(n=200)
pipe = DataPipeline(triplets)
print(len(pipe.train), len(pipe.eval))
```
