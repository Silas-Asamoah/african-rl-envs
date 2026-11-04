"""Patched transcripts: replay decisions with the template tokens after the request set to
another prompt's state for the rest of the episode, and keep what the model writes and how
the episode ends.

    python -m sika_interp.patched --traces a.jsonl b.jsonl --model Qwen/Qwen3.5-9B \\
        --base-url http://localhost:8000/v1 --rpc http://localhost:8000 \\
        --sampling sampling.json --plan plan.json --out patched.jsonl

The server runs `steer_serve.SteerExtension` with the spec files of `modal_app.patch_specs`.
The plan (`patching.replay_plan`) gives decisions ({"task_id", "turn"}), settings ({"name"}
for the unpatched one, else {"name", "folder", "source", "layers"}) and samples per decision.
Settings run in plan order; within a setting every decision and sample runs at once, since
the patch is matched per request. Each finished replay is appended to `--out` as one JSON
line (setting, decision, sample, outcome, transcript), and a rerun skips what is there. The
server's hit count is printed per setting: it should grow by at least one template token per
model turn of every replay, and stays at zero if the signatures don't match the prompts.
"""

import argparse
import asyncio
import json
from pathlib import Path

from sika_agent.checks import Expect

from sika_interp.replay import rebuild, rollout
from sika_interp.traces import load, turns


async def rpc(client, url: str, method: str, arg: dict | None = None) -> dict:
    body = {"method": method} | ({"args": [json.dumps(arg)]} if arg is not None else {})
    r = await client.post(f"{url}/collective_rpc", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["results"][0]


def points_for(traces: list[str], model: str, wanted: set[tuple[str, int]]) -> dict:
    """(task_id, turn) → (point, tools, expect, original reward) for the decisions wanted."""
    ids = {t for t, _ in wanted}
    points = {}
    for path in traces:
        for trace in load(path):
            tid = trace["task"]["data"]["id"]
            if tid in ids and trace["agent"]["config"]["model"] == model:
                built = rebuild(trace, upto=max(k for t, k in wanted if t == tid))
                tools = turns(trace)[0].tools
                expect = Expect.model_validate(trace["task"]["data"]["expect"])
                for t, k in wanted:
                    if t == tid:
                        points[(t, k)] = (built[k], tools, expect, trace["rewards"]["outcome"]["score"])
    return points


async def run(args) -> None:
    import httpx
    from openai import AsyncOpenAI

    plan = json.loads(Path(args.plan).read_text())
    sampling = json.loads(Path(args.sampling).read_text())
    sampling["max_tokens"] = min(sampling.get("max_tokens", args.max_tokens), args.max_tokens)
    standard = {"temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "seed", "stop", "n"}
    extra = {k: v for k, v in sampling.items() if k not in standard}
    sampling = {k: v for k, v in sampling.items() if k in standard} | ({"extra_body": extra} if extra else {})
    decisions = plan["decisions"][: args.limit] if args.limit else plan["decisions"]
    wanted = {(d["task_id"], d["turn"]) for d in decisions}
    points = points_for(args.traces, args.model, wanted)
    missing = wanted - set(points)
    if missing:
        raise SystemExit(f"decisions not found in traces: {sorted(missing)[:5]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["task_id"], r["turn"], r["sample"]))
    llm = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)
    gate = asyncio.Semaphore(args.concurrency)
    n = args.samples or plan.get("samples", 4)
    async with httpx.AsyncClient() as http:
        for s in plan["settings"]:
            todo = [(t, k, i) for (t, k) in sorted(points) for i in range(n) if (s["name"], t, k, i) not in done]
            if not todo:
                continue
            if "source" in s:
                info = await rpc(http, args.rpc, "patch", {k: s[k] for k in ("folder", "source", "layers")})
            else:
                info = await rpc(http, args.rpc, "unpatch")
            print(f"{s['name']}: {len(todo)} replays; server {info}", flush=True)

            async def one(t, k, i):
                point, tools, expect, original = points[(t, k)]
                transcript: list = []
                row = {"setting": s["name"], "source": s.get("source"), "layers": s.get("layers"),
                       "task_id": t, "turn": k, "sample": i, "original": original}
                async with gate:
                    try:
                        row["outcome"] = await rollout(llm, args.model, sampling, tools, point, expect, transcript, point.turns_used + args.turns)
                        row["turn_cap"] = args.turns
                    except Exception as e:  # noqa: BLE001 - one failed replay shouldn't stop the setting
                        row["error"] = f"{type(e).__name__}: {e}"[:300]
                row["transcript"] = transcript
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return row

            rows = await asyncio.gather(*(one(*x) for x in todo))
            ok = [r["outcome"] for r in rows if "outcome" in r]
            state = await rpc(http, args.rpc, "patch_state")
            print(f"  done: right {sum(o == 1 for o in ok)}, harmful {sum(o == -1 for o in ok)}, other {sum(o == 0 for o in ok)},"
                  f" errors {len(rows) - len(ok)}; server {state}", flush=True)
            if "source" in s:
                probe = await rpc(http, args.rpc, "patch_probe")
                print(f"  probe: {json.dumps(probe)[:1500]}", flush=True)
        await rpc(http, args.rpc, "unpatch")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True, nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--rpc", required=True, help="the vLLM server's own address, for /collective_rpc")
    ap.add_argument("--sampling", required=True, help="a JSON file with the model's sampling settings")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096, help="cap per model turn")
    ap.add_argument("--concurrency", type=int, default=240)
    ap.add_argument("--turns", type=int, default=12, help="model turns a replay may take from its decision")
    ap.add_argument("--samples", type=int, default=0, help="replays per decision (default: the plan's)")
    ap.add_argument("--limit", type=int, default=0, help="only the first decisions of the plan (smoke tests)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
