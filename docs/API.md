# API

## CLI

```
kolibriforge doctor | tokenize | route | abstain | train | evaluate | serve
```

## Python

```python
from kolibriforge.config import load_config
from kolibriforge.engine import KolibriEngine

engine = KolibriEngine(load_config())
verdict = engine.answer("What is 2+2?")
print(verdict["action"], verdict["answer"])
```

## HTTP

```
GET  /health
GET  /v1/models
POST /v1/chat/completions   # {"messages":[...], "reasoning_level": "medium"}
POST /v1/completions        # {"prompt": "..."}
```

响应里额外带 `kolibri` 字段，含 `action` / `confidence` / `raw_confidence`。
