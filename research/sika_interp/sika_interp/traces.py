"""Read verifiers eval traces back into the chat requests a model saw.

An episode trace stores its conversation as linked nodes, with tools and assistant tool calls
in the flat Responses-style shape. The model was sent Chat Completions requests, with both
nested under "function", and the two shapes render to different prompt tokens. `turns`
rebuilds each request exactly, so the internal-measurement jobs can replay an episode turn
by turn.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Turn:
    task_id: str
    language: str
    index: int
    """Which of the episode's model turns this is, from 0."""
    messages: list[dict]
    """Everything the model was shown, in order."""
    tools: list[dict]
    """Tool definitions as sent, in Chat Completions shape."""
    response: dict
    """What the model sampled: reasoning_content, content and tool_calls."""
    prompt_tokens: int | None
    """Prompt length the server reported, to check a re-rendering against."""


def chat_tool(tool: dict) -> dict:
    """A trace's flat tool definition in the nested shape the model was sent."""
    if "function" in tool:
        return tool
    fields = {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
    return {"type": "function", "function": fields}


def chat_message(message: dict) -> dict:
    """A trace message as it was sent: assistant tool calls nested under "function"."""
    if not message.get("tool_calls"):
        return message
    calls = []
    for call in message["tool_calls"]:
        if "function" not in call:
            call = {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": {"name": call["name"], "arguments": call.get("arguments", "{}")},
            }
        calls.append(call)
    return dict(message, tool_calls=calls)


def load(path: str | Path) -> Iterator[dict]:
    """The episode traces in a vf-eval `traces.jsonl` (one per rollout)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield from json.loads(line)["traces"]


def turns(trace: dict) -> list[Turn]:
    """Every model turn of an episode, with the request that produced it."""
    data = trace["task"]["data"]
    nodes = trace["nodes"]
    messages = [chat_message(n["message"] if isinstance(n["message"], dict) else json.loads(n["message"])) for n in nodes]
    tools = [chat_tool(t) for t in nodes[0].get("tools") or trace.get("tools") or []]
    # A call the server rejected (e.g. a malformed tool call) produced no node.
    usage = {c["node"]: c.get("usage", {}).get("prompt_tokens") for c in trace.get("calls", []) if "node" in c}
    out = []
    for i, node in enumerate(nodes):
        if node.get("sampled") in (True, "true"):
            history = _chain(nodes, messages, i)
            out.append(
                Turn(
                    task_id=data["id"],
                    language=data["language"],
                    index=len(out),
                    messages=history[:-1],
                    tools=tools,
                    response=history[-1],
                    prompt_tokens=usage.get(i),
                )
            )
    return out


def _chain(nodes: list[dict], messages: list[dict], i: int) -> list[dict]:
    """Messages from the root to node `i`, following parent links."""
    path = []
    while i is not None:
        path.append(messages[i])
        parent = nodes[i].get("parent")
        i = int(parent) if parent not in (None, "", "null") else None
    return path[::-1]
