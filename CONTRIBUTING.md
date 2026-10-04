# 贡献指南

1. 保持内核零依赖（`kolibriforge/core` 只用 stdlib）。
2. 新增后端用 `@register_backend`，并在 `backends/__init__.py` 显式 import。
3. 新增基准用 `@register_benchmark`，并在 `bench/__init__.py` 显式 import。
4. 提交前 `python -m compileall kolibriforge && python -m unittest discover -s tests -t .`。
