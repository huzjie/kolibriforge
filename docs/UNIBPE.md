# UniBPE：复合词感知的字节对编码

## 问题

德语/荷兰语等日耳曼语系会无限拼接名词，形成超长复合词。普通 BPE 对此有两种
坏结果：

1. 整词不在词表 → OOV，回退到字符级，token 数爆炸；
2. 切成稀有子词 → 浪费词表预算。

## 解法：两道工序

```
word ──▶ CompoundLexicon.segment（贪心最长匹配成分）──▶ dampf-schiff-fahrt-gesellschaft
                                                          │
                                                          ▼
                                                 字节级 BPE 编码
```

## 工程要点

- 成分词典按频次排序，贪心取最长匹配；
- 边界用连字符显式标注，下游可复用；
- 词典可以离线构建（统计语料中的高频成分）。

## 复现成本

一个成分词表 + 一个 `segment()` 贪心函数即可，见 `tokenizer/compound.py`。
