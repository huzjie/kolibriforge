"""MCP (Model Context Protocol) tool definition.

Exposes the honest-answer + reasoning-budget capability as an MCP tool JSON
schema, ready to paste into an MCP server config. The actual invocation is
through the same engine used everywhere else.
"""


def build_mcp_tool(cfg=None):
    from ..engine import KolibriEngine
    engine = KolibriEngine(cfg)

    def honest_answer(question: str, reasoning_level: str = "medium") -> dict:
        return engine.answer(question, reasoning_level=reasoning_level)

    tool_schema = {
        "name": "kolibri_honest_answer",
        "description": "Answer with honest abstention (says 'I don't know' instead of hallucinating).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer."},
                "reasoning_level": {"type": "string", "enum": ["none", "low", "medium", "high"],
                                    "description": "Reasoning depth / budget."},
            },
            "required": ["question"],
        },
    }
    return {"tool": tool_schema, "call": honest_answer}
