# kolibriforge

**主权 MoE 推理大模型，带「诚实弃答」防幻觉能力** —— 复刻 Aleph Alpha Kolibri-1
（2026-10-03 开源，78B 参数 / 3.46B 活跃 / 1M 上下文）四大可复用工程思想，做成一
个**零依赖、可直接运行**的训练 + 推理框架。

> 一句话：与其让模型「硬编一个可能错的答案」，不如让它学会**什么时候该说「我不知道」**。

[![ci](https://img.shields.io/badge/CI-passing-brightgreen)]()
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)]()
[![python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![deps](https://img.shields.io/badge/dependencies-zero-lightgrey)]()

---

## 这个框架解决什么问题

大模型最危险的失败不是「答错」，而是**自信地答错**（幻觉）。Kolibri-1 用
**Merlin-Arthur 协议**把这个点做成了可训练行为：Merlin（生成者）提答案，
Arthur（验证者）判可信度，可信度不够就弃答——输出「我不知道」，而不是胡编。

本仓库把它的四条技术主线拆成可直接照抄的工程件：

| 技术点 | 一句话原理 | 对应模块 |
|---|---|---|
| **稀疏 MoE** | 78B 总参数、每 token 只激活 ~3.46B，用 Top-K 路由挑专家 | `kolibriforge/moe/` |
| **UniBPE 分词** | 字节级 BPE 前加一道「复合词预切分」，德语长复合词不再裂成稀有子词 | `kolibriforge/tokenizer/` |
| **Merlin-Arthur** | 生成-验证双角色，可信度低就弃答，防幻觉 | `kolibriforge/merlin/` |
| **推理预算** | 推理时显式四档思考深度（none/low/medium/high），算力-质量可控 | `kolibriforge/reason/` |

---

## 快速开始

```bash
# 零依赖，任何 Python 3.9+ 直接跑
python -m kolibriforge doctor      # 组件自检
python -m kolibriforge tokenize "Donaudampfschifffahrtsgesellschaft"
python -m kolibriforge route "Why does the sky look blue?"
python -m kolibriforge abstain "What is 2+2?"
python -m kolibriforge train       # Merlin-Arthur RL 训练诚实策略
python -m kolibriforge evaluate    # 跑全套基准
python -m kolibriforge serve       # 起 OpenAI 兼容 HTTP 服务
```

```bash
# OpenAI 兼容调用
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

---

## 核心可复用思路

### 1. Merlin-Arthur：把「诚实」写成一个奖励函数

这是整个框架最有复用价值的部分。幻觉的根源是**校准失败**——模型不知道
自己不知道。解法不是更大的模型，而是一个**不对称奖励**：

```
reward = +1.0                    若接受且正确
         -1.5                    若接受但错误   ← 幻觉，最重惩罚
         +0.2                    若弃答且确实不知道  ← 诚实的「不知道」
          0.0                    其余
```

关键在「接受但错误」的惩罚**大于**「弃答」的收益差——于是策略学会：拿不准
就闭嘴。见 `kolibriforge/train/merlin_rl.py` 与 `kolibriforge/merlin/protocol.py`。

### 2. UniBPE：复合词预切分再 BPE

德语「Donaudampfschifffahrtsgesellschaft」（多瑙河汽船航运公司）这类长复合词，
普通 BPE 要么整词 OOV，要么切成一堆稀有子词。UniBPE 的解法是**两道工序**：

1. 用频次词典对单词做贪心最长匹配，切出 `dampf-schiff-fahrt-gesellschaft`；
2. 再交给字节级 BPE。

复现时只需一个按频次排好的成分词表 + 贪心切分，见
`kolibriforge/tokenizer/compound.py`。

### 3. 稀疏 MoE：路由 + 负载均衡两个都要

MoE 省算力的前提是「每个 token 只走 K 个专家」。两个容易踩的坑：

- **路由塌缩**：所有 token 挤到少数专家，算力没省还热点拥塞 → 加**负载均衡辅助损失**；
- **门控权重**：Top-K 选中的专家权重要重新归一化，否则梯度偏。

见 `kolibriforge/moe/balance.py` 与 `kolibriforge/moe/router.py`。

### 4. 推理预算：把「想多久」变成显式旋钮

四个深度档位（none/low/medium/high）映射到不同的额外 token 预算。难度估计
用一个廉价启发式（长度 + 疑问词密度 + 算术/可验证标记），把「该不该多想」
从隐式变显式。见 `kolibriforge/reason/budget.py`。

---

## 架构

```
query
  │
  ├─ UniBPE tokenizer ──▶ token ids
  ├─ DifficultyRouter ──▶ reasoning budget (none/low/medium/high)
  │
  ▼
Backend (mock/cpu/openai/vllm/transformers)
  │  generate(answer, confidence)
  ▼
Merlin-Arthur protocol
  ├─ Arthur.confidence()  ──▶ calibrated confidence
  └─ AbstentionGate       ──▶ accept / abstain / reject
                              │
                              ▼
                     answer | "I don't know" | None
```

---

## 后端

| 后端 | 说明 | 可训练 |
|---|---|---|
| `mock` | 确定性可训练，诚实 skill 单调趋 1 | ✅ |
| `cpu` | 用内置张量核跑 KolibriMoE | — |
| `openai` | 任意 OpenAI 兼容网关（含 Azure/vLLM/Ollama） | — |
| `vllm` | vLLM 本地高吞吐 | — |
| `transformers` | Hugging Face 适配层 | — |

---

## 基准

`python -m kolibriforge evaluate` 跑五组基准：诚实度（ECE + 幻觉率）、弃答率
（已知要答 / 未知要弃）、AIME 式数学、长上下文检索、分词压缩率。

---

## 部署

- **Docker**：`docker build -f docker/Dockerfile -t kolibriforge .`
- **Kubernetes**：`kubectl apply -f k8s/`
- **Helm**：`helm install kolibriforge helm/kolibriforge`
- **CI**：`.github/workflows/ci.yml`

---

## 目录结构

```
kolibriforge/
├── kolibriforge/          # 主包（零依赖内核）
│   ├── core/              # 纯 Python 张量 + 注意力 + 神经网络原语
│   ├── tokenizer/         # UniBPE + 复合词词典
│   ├── moe/               # 稀疏 MoE（路由/专家/负载均衡/层）
│   ├── merlin/            # Merlin-Arthur 协议（Arthur/校准/门控）
│   ├── reason/            # 推理预算 + 难度路由
│   ├── model/             # KolibriMoE 模型 + LM 头
│   ├── backends/          # 5 后端
│   ├── train/             # pretrain / sft / merlin-rl
│   ├── bench/             # 5 组基准
│   ├── serving/           # OpenAI 兼容 stdlib HTTP 服务
│   ├── integrations/      # LangChain / MCP
│   └── cli/               # 命令行
├── examples/              # 7 个可运行示例
├── tests/                 # 单元测试
├── configs/               # default / kolibri-78b / kolibri-flash
├── prompts/               # 提示词库
├── recipes/               # 可复用配方
├── docs/                  # 架构与设计文档
├── docker/ k8s/ helm/     # 部署
└── .github/workflows/     # CI
```

## License

Apache-2.0。
