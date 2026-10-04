# 配方 05：OpenAI 兼容服务

## 目标
用 OpenAI 客户端调用 kolibriforge。

## 步骤
1. `kolibriforge serve --port 8000`；
2. `curl /v1/chat/completions`；
3. 响应里读 `kolibri.action` 判断 accept/abstain。

## 验收
`/health` 返回 200，`/v1/chat/completions` 返回带 `kolibri` 字段的 JSON。
