"""Serving adapters: make each model see the conversation the way its makers recommend.

Ministral 3 Reasoning only thinks when Mistral's reasoning instructions follow the system
prompt, with the `[THINK]` example passed as a structured thinking block (its model card:
"append it to your custom system prompt"). Its earlier reasoning should also stay in context.
`for_ministral` rewrites an OpenAI-style request that way. The proxy below applies it in
front of a vLLM server, and the internal-measurement jobs apply the same function when they
render prompts, so both see identical tokens.

Qwen3.5 needs no adapter: its chat template opens a thinking block itself.

    python -m sika_interp.serving --upstream http://localhost:8001 --port 8002 \\
        --reasoning-prompt SYSTEM_PROMPT.txt
"""

import argparse
import copy

THINK_OPEN, THINK_CLOSE = "[THINK]", "[/THINK]"


def reasoning_system_parts(own: str, reasoning_prompt: str) -> list[dict]:
    """Our system prompt, then Mistral's reasoning prompt with its `[THINK]` example as a
    closed thinking block."""
    start, end = reasoning_prompt.find(THINK_OPEN), reasoning_prompt.find(THINK_CLOSE)
    if start < 0 or end < start:
        raise ValueError("reasoning prompt has no [THINK]...[/THINK] example")
    return [
        {"type": "text", "text": own + "\n\n" + reasoning_prompt[:start]},
        {"type": "thinking", "thinking": reasoning_prompt[start + len(THINK_OPEN) : end], "closed": True},
        {"type": "text", "text": reasoning_prompt[end + len(THINK_CLOSE) :]},
    ]


def for_ministral(messages: list[dict], reasoning_prompt: str) -> list[dict]:
    """A copy of `messages` with Mistral's reasoning prompt added to the system message and
    each earlier assistant turn's `reasoning_content` turned back into a thinking block."""
    out = copy.deepcopy(messages)
    for message in out:
        if message["role"] == "system" and isinstance(message.get("content"), str):
            message["content"] = reasoning_system_parts(message["content"], reasoning_prompt)
        elif message["role"] == "assistant" and message.get("reasoning_content"):
            parts = [{"type": "thinking", "thinking": message.pop("reasoning_content"), "closed": True}]
            if message.get("content"):
                parts.append({"type": "text", "text": message["content"]})
            message["content"] = parts
        else:
            message.pop("reasoning_content", None)
    return out


def proxy(upstream: str, reasoning_prompt: str):
    """An app that forwards OpenAI requests to `upstream`, rewriting chat requests with
    `for_ministral` and streaming responses back unchanged."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse

    app = FastAPI()
    client = httpx.AsyncClient(base_url=upstream, timeout=None)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        body = await request.json() if request.method == "POST" else None
        if body is not None and path.endswith("chat/completions"):
            body["messages"] = for_ministral(body["messages"], reasoning_prompt)
        upstream_request = client.build_request(request.method, "/" + path, json=body)
        response = await client.send(upstream_request, stream=True)
        if "text/event-stream" in response.headers.get("content-type", ""):
            return StreamingResponse(response.aiter_raw(), media_type="text/event-stream")
        content = await response.aread()
        return Response(content, status_code=response.status_code, media_type=response.headers.get("content-type"))

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--reasoning-prompt", required=True, help="Mistral's SYSTEM_PROMPT.txt")
    args = ap.parse_args()
    import uvicorn

    with open(args.reasoning_prompt, encoding="utf-8") as f:
        uvicorn.run(proxy(args.upstream, f.read()), host="127.0.0.1", port=args.port, log_level="warning")
