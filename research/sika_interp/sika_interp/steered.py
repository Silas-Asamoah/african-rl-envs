"""Steered transcripts: replay decisions with a direction pushed into the model for the rest of
the episode, and keep what it writes and how the episode ends.

    python -m sika_interp.steered --traces a.jsonl b.jsonl --model Qwen/Qwen3.5-9B \\
        --base-url http://localhost:8000/v1 --rpc http://localhost:8000 \\
        --config configs/qwen.toml --plan plan-qwen.json --out steered-qwen.jsonl

The server runs `steer_serve.SteerExtension` (see remote/pod.sh serve-steer). The plan gives
decisions ({"task_id", "turn"}), directions ({"name", "layer", "neuron"} or {"name", "layer",
"seed"} for a random one), doses, signs and samples per decision. Settings run in plan order,
the unsteered one first; within a setting every decision and sample runs at once, since the
push applies to every request the server holds. Each finished replay is appended to `--out`
as one JSON line (setting, decision, sample, outcome, transcript), and a rerun skips what is
already there.
"""

import argparse
import asyncio
import json
import tomllib
from pathlib import Path

from sika_agent.checks import Expect

from sika_interp.replay import rebuild, rollout
from sika_interp.traces import load, turns


def settings(plan: dict) -> list[dict]:
    """The unsteered setting, then each direction × dose × sign, in plan order."""
    out = [{"name": "unsteered"}]
    for d in plan["directions"]:
        for dose in plan["doses"]:
            for sign in plan["signs"]:
                out.append(d | {"dose": dose, "sign": sign, "name": f"{d['name']} {'+' if sign > 0 else '−'}{dose}"})
    return out


async def rpc(client, url: str, method: str, arg: dict | None = None) -> dict:
    body = {"method": method} | ({"args": [json.dumps(arg)]} if arg is not None else {})
    r = await client.post(f"{url}/collective_rpc", json=body, timeout=60)
    r.raise_for_status()
    return r.json()["results"][0]


async def run(args) -> None:
    import httpx
    from openai import AsyncOpenAI

    plan = json.loads(Path(args.plan).read_text())
    sampling = tomllib.loads(Path(args.config).read_text())["sampling"]
    sampling["max_tokens"] = min(sampling.get("max_tokens", args.max_tokens), args.max_tokens)
    wanted = {(d["task_id"], d["turn"]) for d in plan["decisions"]}
    ids = {t for t, _ in wanted}
    points = {}
    for path in args.traces:
        for trace in load(path):
            tid = trace["task"]["data"]["id"]
            if tid in ids and trace["agent"]["config"]["model"] == args.model:
                built = rebuild(trace)
                tools = turns(trace)[0].tools
                expect = Expect.model_validate(trace["task"]["data"]["expect"])
                for t, k in wanted:
                    if t == tid:
                        points[(t, k)] = (built[k], tools, expect, trace["rewards"]["outcome"]["score"])
    missing = wanted - set(points)
    if missing:
        raise SystemExit(f"decisions not found in traces: {sorted(missing)[:5]}")
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["task_id"], r["turn"], r["sample"]))
    llm = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)
    gate = asyncio.Semaphore(args.concurrency)
    n = plan.get("samples", 4)
    async with httpx.AsyncClient() as http:
        for s in settings(plan):
            todo = [(t, k, i) for (t, k) in sorted(points) for i in range(n) if (s["name"], t, k, i) not in done]
            if not todo:
                continue
            if s["name"] == "unsteered":
                await rpc(http, args.rpc, "unsteer")
            else:
                spec = {k: s[k] for k in ("layer", "dose", "sign")} | ({"neuron": s["neuron"]} if "neuron" in s else {"seed": s["seed"]})
                await rpc(http, args.rpc, "steer", spec)
            state = await rpc(http, args.rpc, "steer_state")
            print(f"{s['name']}: {len(todo)} replays; server {state}", flush=True)

            async def one(t, k, i):
                point, tools, expect, original = points[(t, k)]
                transcript: list = []
                row = {"setting": s["name"], "direction": {k2: s[k2] for k2 in ("layer", "neuron", "seed", "dose", "sign") if k2 in s},
                       "task_id": t, "turn": k, "sample": i, "original": original}
                async with gate:
                    try:
                        row["outcome"] = await rollout(llm, args.model, sampling, tools, point, expect, transcript)
                    except Exception as e:  # noqa: BLE001 - one failed replay shouldn't stop the setting
                        row["error"] = f"{type(e).__name__}: {e}"[:300]
                row["transcript"] = transcript
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return row

            rows = await asyncio.gather(*(one(*x) for x in todo))
            ok = [r["outcome"] for r in rows if "outcome" in r]
            print(f"  done: right {sum(o == 1 for o in ok)}, harmful {sum(o == -1 for o in ok)}, other {sum(o == 0 for o in ok)},"
                  f" errors {len(rows) - len(ok)}", flush=True)
        await rpc(http, args.rpc, "unsteer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True, nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--rpc", required=True, help="the vLLM server's own address, for /collective_rpc")
    ap.add_argument("--config", required=True, help="configs/<model>.toml, for its sampling")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096, help="cap per model turn")
    ap.add_argument("--concurrency", type=int, default=160)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
