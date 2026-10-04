# 后端对比

| 后端 | 依赖 | 吞吐 | 可训练 | 适用 |
|---|---|---|---|---|
| mock | 无 | 极高 | ✅ | 测试/演示/CI |
| cpu | 无 | 中 | — | 本地推理 |
| openai | 无(urllib) | 网络受限 | — | 任意兼容网关 |
| vllm | vllm | 高 | — | 本地批量 |
| transformers | transformers | 低 | — | HF 模型 |

## 切换

```yaml
backend:
  name: openai
  api_base: https://your-gateway/v1
  api_key: ${OPENAI_API_KEY}
```
