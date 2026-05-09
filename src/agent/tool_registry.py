"""
Tool schema module — exposes agent and customer tools in a standardised format.

Re-exports definitions from ``toolset.py`` and adds helpers for prompt injection.
"""

from typing import Any, Dict, List

from .toolset import (
    ALL_TOOLS,
    CUSTOMER_TOOLS,
)


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_agent_tools() -> List[Dict[str, Any]]:
    """All agent tools (read + write) in OpenAI function-calling format."""
    return ALL_TOOLS


def get_customer_tools() -> List[Dict[str, Any]]:
    """All customer tools in OpenAI function-calling format."""
    return CUSTOMER_TOOLS


def get_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    """Extract function names from a list of tool definitions."""
    return [t["function"]["name"] for t in tools]


# ---------------------------------------------------------------------------
# Prompt-injection formatter (for providers without native function calling)
# ---------------------------------------------------------------------------

def _format_tool_entries(tools: List[Dict[str, Any]]) -> List[str]:
    """Return lines describing each tool's name, description, and parameters."""
    lines = []
    for tool in tools:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        lines.append(f"**{name}**")
        lines.append(f"  Description: {desc}")
        if params:
            lines.append("  Parameters:")
            for param_name, param_info in params.items():
                req_marker = " (required)" if param_name in required else ""
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")
                lines.append(
                    f"    - {param_name}{req_marker}: {param_type} — {param_desc}"
                )
        lines.append("")
    return lines


def format_tools_for_prompt_detailed() -> str:
    """Format all agent tool definitions as a text block for prompt injection."""
    lines = [
        "## Available Tools",
        "",
        "You have access to the following tools. When you need to use a tool,",
        "include it in the `tool_calls_made` array in your JSON response.",
        "",
    ]
    lines.extend(_format_tool_entries(ALL_TOOLS))
    lines.extend([
        "### Tool Call Format",
        "",
        "Include tool calls in your response JSON like this:",
        "```json",
        "{",
        '  "tool_calls_made": [',
        "    {",
        '      "tool_name": "get_order_details",',
        '      "tool_call_id": "call_001",',
        '      "arguments": {"order_id": "ORDER-123456"}',
        "    }",
        "  ],",
        "  ... rest of your response ...",
        "}",
        "```",
        "",
        "IMPORTANT: Tool results ARE provided back to you in the conversation",
        "history as tool_result entries. Use them to inform your decisions.",
    ])
    return "\n".join(lines)


def format_customer_tools_for_prompt_detailed() -> str:
    """Format customer tools for prompt injection."""
    lines = [
        "## Available Customer Actions",
        "",
        "You have access to the following tools to perform actions during the conversation:",
        "",
    ]
    lines.extend(_format_tool_entries(CUSTOMER_TOOLS))
    return "\n".join(lines)
