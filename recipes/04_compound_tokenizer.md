# 配方 04：复合词分词器

## 目标
降低德语等复合词语言的 token 数。

## 步骤
1. 收集语料，统计高频成分词表；
2. 填入 `CompoundLexicon.entries`；
3. `kolibriforge tokenize "<复合词>"` 查看压缩比。

## 验收
`tokenization.compression_ratio` 明显大于 1。
