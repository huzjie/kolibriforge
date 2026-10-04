# 配方 06：MCP 集成

## 目标
把诚实问答暴露为 MCP 工具。

## 步骤
1. `from kolibriforge.integrations.mcp import build_mcp_tool`；
2. 把返回的 `tool` schema 写入你的 MCP server 配置；
3. 用 `call(question=..., reasoning_level=...)` 调用。

## 验收
MCP 客户端能列出并调用 `kolibri_honest_answer` 工具。
