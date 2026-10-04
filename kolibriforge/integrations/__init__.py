"""Third-party integrations (LangChain, MCP)."""
from .langchain import KolibriLLM
from .mcp import build_mcp_tool

__all__ = ["KolibriLLM", "build_mcp_tool"]
