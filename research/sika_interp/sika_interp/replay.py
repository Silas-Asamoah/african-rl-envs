"""Replay an episode from any of its turns, for the anchors' behavioural check.

`rebuild` re-runs a recorded episode's tool calls on a fresh wallet (each result must match
the trace) to recover the exact wallet and conversation before each model turn. `rollout`
then lets the model play on from there against the same simulator and scripted user as
sika-agent (a reply without a tool call ends a segment; a reply with a "?" gets the task's
scripted answer, up to `USER_TURNS` times) and grades the final wallet.

Resampling from turn k many times, against resampling from k+1 (turn k kept), shows how much
turn k decides the outcome: the resampling importance of Thought Anchors, at turn level.

    python -m sika_interp.replay --traces traces.jsonl --task-id eval/promo_fee/en-tw/1000000 \\
        --base-url http://localhost:8000/v1 --n 10 --out anchors.json
"""

import argparse
import asyncio
import inspect
import json
from dataclasses import dataclass, field

from sika_agent import wallet
from sika_agent.checks import Expect, evaluate
from sika_agent.wallet import SikaState, scripted_reply

from sika_interp.traces import load, turns

USER_TURNS = 3
"""Plain-text questions the scripted user answers (SikaEnvConfig.user_turns)."""
MAX_TURNS = 30
"""Model turns before an episode is cut off (SikaTaskConfig.max_turns)."""
TOOLS = {
    name: fn
    for name, fn in inspect.getmembers(wallet, inspect.isfunction)
    if fn.__module__ == wallet.__name__ and next(iter(inspect.signature(fn).parameters), "") == "s"
}


@dataclass
class Point:
    """The episode just before one model turn."""

    messages: list[dict]
    state: SikaState
    user_turns_used: int
    turns_used: int
    said: list[str] = field(default_factory=list)


def call_tool(state: SikaState, name: str, arguments: str) -> str:
    """Run one tool call against the wallet, as the toolset would."""
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: unknown tool {name}"
    try:
        return str(fn(state, **json.loads(arguments or "{}")))
    except Exception as e:  # a bad call is the model's mistake; it sees the error
        return f"Error: {e}"


def rebuild(trace: dict) -> list[Point]:
    """The episode just before each of its model turns, rebuilt by re-running its tool calls.
    Raises if a re-run tool result differs from the recorded one."""
    data = trace["task"]["data"]
    state = SikaState.model_validate(data["world"])
    points, said, user_used = [], [], 0
    for t in turns(trace):
        history = t.messages
        # Bring the state up to this turn: execute the calls the previous turn made, and
        # record a plain-text question the scripted user answered.
        if points:
            prev = points[-1]
            done = len(prev.messages)
            for message in history[done:]:
                if message["role"] == "assistant":
                    for call in message.get("tool_calls") or []:
                        fn = call["function"]
                        result = call_tool(state, fn["name"], fn["arguments"])
                        recorded = next(
                            (m["content"] for m in history if m.get("tool_call_id") == call["id"]), None
                        )
                        if recorded is not None and recorded != result:
                            raise ValueError(f"{data['id']}: re-run {fn['name']} gave a different result")
                    if message.get("content"):
                        said.append(message["content"])
                elif message["role"] == "user":
                    question = next(m["content"] for m in reversed(history[: history.index(message)]) if m["role"] == "assistant")
                    state.events.append("ask_user")
                    state.questions.append(question)
                    user_used += 1
        points.append(Point(history, state.model_copy(deep=True), user_used, t.index, list(said)))
    return points


async def rollout(client, model: str, sampling: dict, tools: list[dict], point: Point, expect: Expect) -> float:
    """Play the episode on from `point`; the outcome reward (1 success, 0 miss, −1 harmful)."""
    state = point.state.model_copy(deep=True)
    messages = list(point.messages)
    said = list(point.said)
    user_used, turns_used = point.user_turns_used, point.turns_used
    while turns_used < MAX_TURNS:
        response = await client.chat.completions.create(model=model, messages=messages, tools=tools, **sampling)
        turns_used += 1
        m = response.choices[0].message
        message = {"role": "assistant", "content": m.content}
        reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
        if reasoning:
            message["reasoning_content"] = reasoning
        calls = m.tool_calls or []
        if calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ]
        messages.append(message)
        if m.content:
            said.append(m.content)
        if calls:
            for c in calls:
                result = call_tool(state, c.function.name, c.function.arguments)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result, "name": c.function.name})
            continue
        if user_used >= USER_TURNS or "?" not in (m.content or ""):
            break
        state.events.append("ask_user")
        state.questions.append(m.content)
        messages.append({"role": "user", "content": scripted_reply(state)})
        user_used += 1
    verdict = evaluate(state, expect, "\n".join(said))
    return 1.0 if verdict.success else -1.0 if verdict.harmful else 0.0


async def resample(client, model: str, sampling: dict, trace: dict, n: int, concurrency: int = 32) -> list[dict]:
    """`n` rollouts from just before every model turn of `trace`."""
    tools = turns(trace)[0].tools
    expect = Expect.model_validate(trace["task"]["data"]["expect"])
    points = rebuild(trace)
    gate = asyncio.Semaphore(concurrency)

    async def one(point: Point) -> float:
        async with gate:
            return await rollout(client, model, sampling, tools, point, expect)

    out = []
    for k, point in enumerate(points):
        rewards = await asyncio.gather(*(one(point) for _ in range(n)), return_exceptions=True)
        out.append(
            {
                "turn": k,
                "rewards": [r for r in rewards if isinstance(r, float)],
                "errors": sum(not isinstance(r, float) for r in rewards),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True, nargs="+")
    ap.add_argument("--task-id", required=True, nargs="+")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sampling", default="{}", help="JSON sampling options, as in configs/<model>.toml")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    wanted = set(args.task_id)
    found = {}
    for path in args.traces:
        for trace in load(path):
            if trace["task"]["data"]["id"] in wanted and trace["agent"]["config"]["model"] == args.model:
                found[trace["task"]["data"]["id"]] = trace

    async def run() -> dict:
        jobs = {tid: resample(client, args.model, json.loads(args.sampling), tr, args.n) for tid, tr in found.items()}
        return dict(zip(jobs, await asyncio.gather(*jobs.values())))

    result = asyncio.run(run())
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n": args.n, "episodes": result}, f, ensure_ascii=False, indent=1)
    print(f"{len(result)} episodes resampled into {args.out}")


if __name__ == "__main__":
    main()
