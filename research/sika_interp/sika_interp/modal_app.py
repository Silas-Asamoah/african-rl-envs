"""Internal-measurement jobs on Modal: load a model on a GPU and look at what it computes on
the prompts it saw in sika-agent episodes.

    uvx modal run -m sika_interp.modal_app --traces outputs/<run>/traces.jsonl[,<more>]

For now this re-renders every turn of each episode, checks the prompt length against what
the vLLM server reported (so internal measurements see exactly the served tokens), and
records the residual stream and MLP neurons at the last prompt token of each turn.
Versions match the vLLM install on the Prime pod (torch 2.13, transformers 5.19).
"""

import json
from pathlib import Path

import modal

from sika_interp.traces import load, turns

MINISTRAL = "mistralai/Ministral-3-8B-Reasoning-2512"
cache = modal.Volume.from_name("sika-interp-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.13.0",
        "transformers==5.19.0",
        "accelerate",
        "mistral-common>=1.8.6",
        "huggingface_hub",
        "numpy",
        "scikit-learn",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("sika_interp")
)
app = modal.App("sika-interp", image=image)


def for_template(messages: list[dict]) -> list[dict]:
    """Tool-call arguments as objects, as vLLM hands them to chat templates."""
    out = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                call = json.loads(json.dumps(call))
                fn = call.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append(call)
            message["tool_calls"] = calls
        out.append(message)
    return out


@app.cls(gpu="L40S", volumes={"/cache": cache}, timeout=3600, scaledown_window=60)
class Model:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.layers = next(
            m
            for name, m in self.model.named_modules()
            if name.endswith("language_model.layers") and isinstance(m, torch.nn.ModuleList)
        )
        self.reasoning_prompt = None
        if self.model_id == MINISTRAL:
            self.reasoning_prompt = Path(hf_hub_download(MINISTRAL, "SYSTEM_PROMPT.txt")).read_text()
        cache.commit()

    def render(self, messages: list[dict], tools: list[dict]):
        from sika_interp.serving import for_ministral

        if self.reasoning_prompt:
            messages = for_ministral(messages, self.reasoning_prompt)
        else:
            messages = for_template(messages)
        enc = self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        return enc["input_ids"].to("cuda")

    @modal.method()
    def last_token(self, requests: list[dict]) -> list[dict]:
        """For each request: rendered length, the server's length, and the residual stream
        and MLP neurons at the last prompt token (norms here; arrays saved to the volume)."""
        import numpy as np
        import torch

        results = []
        for req in requests:
            ids = self.render(req["messages"], req["tools"])
            neurons: dict[int, torch.Tensor] = {}

            def keep(i):
                def hook(module, args):
                    neurons[i] = args[0][0, -1].float().cpu()

                return hook

            handles = [layer.mlp.down_proj.register_forward_pre_hook(keep(i)) for i, layer in enumerate(self.layers)]
            with torch.no_grad():
                out = self.model(input_ids=ids, output_hidden_states=True)
            for h in handles:
                h.remove()
            residual = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
            mlp = torch.stack([neurons[i] for i in range(len(self.layers))])
            name = f"{self.model_id.split('/')[-1]}/{req['task_id'].replace('/', '_')}_{req['index']}.npz"
            path = Path("/cache/acts") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, residual=residual.numpy(), mlp=mlp.numpy(), ids=ids[0].cpu().numpy())
            top = out.logits[0, -1].float().softmax(-1).topk(3)
            results.append(
                {
                    "task_id": req["task_id"],
                    "index": req["index"],
                    "tokens": int(ids.shape[1]),
                    "server_tokens": req["prompt_tokens"],
                    "residual": list(residual.shape),
                    "mlp": list(mlp.shape),
                    "next": [(self.tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top.values, top.indices)],
                    "saved": name,
                }
            )
        cache.commit()
        return results


    def render_turn(self, messages: list[dict], tools: list[dict], response: dict):
        """Token ids for the prompt alone and for the prompt followed by the model's own
        response (reasoning and action), rendered by the chat template."""
        from sika_interp.serving import for_ministral

        full = messages + [dict(response, role="assistant")]
        if self.reasoning_prompt:
            # Mistral's tokenizer won't end on an assistant turn, so placeholders follow it.
            # Attention is causal: they can't change activations up to the action.
            calls = response.get("tool_calls") or []
            full += [{"role": "tool", "tool_call_id": c["id"], "content": ""} for c in calls] or [
                {"role": "user", "content": "."}
            ]
            full = for_ministral(full, self.reasoning_prompt)
        else:
            full = for_template(full)
        enc = self.tok.apply_chat_template(full, tools=tools, return_dict=True, return_tensors="pt")
        return self.render(messages, tools)[0].tolist(), enc["input_ids"][0].tolist()

    @modal.method()
    def read_points(self, run: str, episodes: list[dict]) -> list[dict]:
        """Residual stream at every layer and MLP neurons at three points of each episode's
        first turn: the end of the request, the last reasoning token, and the first token
        of the action (tool call or reply). Saved as float16 to the volume under
        acts/<run>/<model>/; returns the positions and checks for each episode."""
        import numpy as np
        import torch

        close = "[/THINK]" if self.reasoning_prompt else "</think>"
        rows = []
        for ep in episodes:
            prompt, full = self.render_turn(ep["messages"], ep["tools"], ep["response"])
            row = {k: ep[k] for k in ("task_id", "family", "category", "language", "seed", "reward")}
            row["prefix_ok"] = full[: len(prompt)] == prompt
            pieces = [self.tok.decode([i]) for i in full[len(prompt) :]]
            end = next((j for j, p in enumerate(pieces) if p.strip() == close), None)
            after = (end + 1) if end is not None else 0
            action = next((j for j in range(after, len(pieces)) if pieces[j].strip()), None)
            points = {
                "request_end": len(prompt) - 1,
                "reasoning_end": len(prompt) + end - 1 if end else None,
                "action_start": len(prompt) + action if action is not None else None,
            }
            row.update(points, tokens=len(full), reasoning_tokens=end or 0)
            row["action_token"] = pieces[action] if action is not None else None
            positions = [p if p is not None else points["request_end"] for p in points.values()]
            ids = torch.tensor([full], device="cuda")
            captured: dict[int, torch.Tensor] = {}

            def keep(i):
                def hook(module, args):
                    captured[i] = args[0][0, positions].float().cpu()

                return hook

            handles = [layer.mlp.down_proj.register_forward_pre_hook(keep(i)) for i, layer in enumerate(self.layers)]
            with torch.no_grad():
                out = self.model(input_ids=ids, output_hidden_states=True)
            for h in handles:
                h.remove()
            residual = torch.stack([h[0, positions].float().cpu() for h in out.hidden_states], dim=1)
            mlp = torch.stack([captured[i] for i in range(len(self.layers))], dim=1)
            path = Path("/cache/acts") / run / self.model_id.split("/")[-1] / (ep["task_id"].replace("/", "_") + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                residual=residual.numpy().astype(np.float16),
                mlp=mlp.numpy().astype(np.float16),
                points=np.array([p if p is not None else -1 for p in points.values()]),
            )
            rows.append(row)
        cache.commit()
        return rows


THREATS = ("scams", "injection")
"""Categories where the right move is to refuse: the target of the threat probe."""


POINTS = ("request_end", "reasoning_end", "action_start")
"""The read points `read_points` stores, in order (probes.POINTS mirrors this)."""
QUESTIONS = ("family", "threat", "language", "placement")
"""What `analyse` asks of each read point: task type, scam-or-injection, request language,
and where Twi–English sits between its twins."""


@app.function(volumes={"/cache": cache}, cpu=8, memory=32768, timeout=7200)
def analyse(run: str, model: str, point: str, question: str, categories: dict[str, str], matched_below: int) -> dict:
    """One question about one read point of one model (see probes.py), next to the stored
    arrays. `categories` maps family to category; seeds below `matched_below` are the matched
    twins. Per-layer results come back as lists."""
    import numpy as np

    from sika_interp import probes

    X, family, language, seed = probes.load(Path("/cache/acts") / run / model.split("/")[-1], point)
    out: dict = {"n": {lang: int((language == lang).sum()) for lang in ("en", "tw", "en-tw")}}
    if question in ("family", "threat"):
        y = family if question == "family" else np.array([categories[f] in THREATS for f in family])
        out["within"] = {
            lang: probes.layer_accuracy(X[language == lang], y[language == lang], seed[language == lang]).tolist()
            for lang in ("en", "tw", "en-tw")
        }
        cross = probes.cross_language_accuracy(X, y, language, seed, test=("en", "tw", "en-tw"))
        out["from_en"] = {k: v.tolist() for k, v in cross.items()}
        out["chance"] = probes.chance(y)
    elif question == "language":
        out["within"] = probes.layer_accuracy(X, language, seed).tolist()
        out["chance"] = probes.chance(language)
    else:
        twins = seed < matched_below
        placed = probes.twin_placement(X[twins], family[twins], seed[twins], language[twins])
        out.update({k: v.tolist() for k, v in placed.items()})
    return out


@app.function(volumes={"/cache": cache}, cpu=8, memory=32768, timeout=7200)
def episode_scores(run: str, model: str, point: str, categories: dict[str, str]) -> dict:
    """Per episode at one read point, every layer: the scam-or-injection probability from a
    probe trained on English episodes of other seeds, and (Twi–English with both twins) how
    much nearer the English twin than the Twi twin it sits."""
    import numpy as np

    from sika_interp import probes

    X, family, language, seed = probes.load(Path("/cache/acts") / run / model.split("/")[-1], point)
    threat = np.array([categories[f] in THREATS for f in family])
    scores = np.stack([probes.held_out_scores(X[:, layer], threat, language, seed) for layer in range(X.shape[1])], 1)
    margins: dict[int, list[float]] = {}
    for layer in range(X.shape[1]):
        for i, m in probes.placement_margins(X[:, layer], family, seed, language).items():
            margins.setdefault(i, []).append(m)
    return {
        "episodes": [
            {
                "family": str(family[i]),
                "language": str(language[i]),
                "seed": int(seed[i]),
                "threat_score": [round(float(v), 4) for v in scores[i]],
                "margin": [round(v, 4) for v in margins[i]] if i in margins else None,
            }
            for i in range(len(family))
        ]
    }


@app.local_entrypoint()
def link(run: str, model: str, points: str):
    """Tie the inside to behaviour for one model: per read point, the English-trained scam
    probe's score on scam and injection episodes by outcome and language, and Twi–English
    placement by outcome. `points`: the read-points JSON from `read` (outcomes, categories)."""
    rows = json.loads(Path(points).read_text())
    categories = {r["family"]: r["category"] for r in rows}
    reward = {(r["family"], r["language"], r["seed"]): r["reward"] for r in rows}
    calls = {point: episode_scores.spawn(run, model, point, categories) for point in POINTS}
    res = {point: call.get() for point, call in calls.items()}
    out = Path(f"link-{run}-{model.split('/')[-1]}.json")
    out.write_text(json.dumps(res))
    outcome = {1.0: "success", 0.0: "miss", -1.0: "harmful"}

    def mean(xs: list[float]) -> str:
        return f"{sum(xs) / len(xs):.2f} (n={len(xs)})" if xs else "-"

    for point, r in res.items():
        eps = [dict(e, reward=reward[(e["family"], e["language"], e["seed"])]) for e in r["episodes"]]
        layers = len(eps[0]["threat_score"])
        print(f"== {point}")
        for layer in range(4, layers, 4):
            line = []
            for lang in ("en", "tw", "en-tw"):
                threats = [e for e in eps if e["language"] == lang and categories[e["family"]] in THREATS]
                by = " ".join(
                    f"{outcome[k][:4]} {mean([e['threat_score'][layer] for e in threats if e['reward'] == k])}"
                    for k in (1.0, -1.0)
                )
                line.append(f"{lang}: {by}")
            print(f"  L{layer:<2} scam score on scam/injection tasks | " + " | ".join(line))
        for layer in range(4, layers, 4):
            mixed = [e for e in eps if e["margin"]]
            ok = [e["margin"][layer] for e in mixed if e["reward"] == 1.0]
            bad = [e["margin"][layer] for e in mixed if e["reward"] != 1.0]
            print(f"  L{layer:<2} en-tw nearer English twin (margin): success {mean(ok)} | fail {mean(bad)}")
    print(f"saved {out}")


@app.local_entrypoint()
def probe(run: str, model: str, points: str, matched_below: int = 1_000_010):
    """Run every `analyse` question on every read point of one model, in parallel, and save
    the summary JSON locally. `points`: the read-points JSON the `read` entrypoint wrote
    (for each family's category). Seeds below `matched_below` are the matched twins."""
    categories = {r["family"]: r["category"] for r in json.loads(Path(points).read_text())}
    calls = {
        (point, q): analyse.spawn(run, model, point, q, categories, matched_below) for point in POINTS for q in QUESTIONS
    }
    res: dict = {}
    for (point, q), call in calls.items():
        res.setdefault(point, {})[q] = call.get()
    out = Path(f"probes-{run}-{model.split('/')[-1]}.json")
    out.write_text(json.dumps(res, indent=1))

    def best(xs: list[float]) -> str:
        i = max(range(len(xs)), key=lambda k: xs[k] if xs[k] == xs[k] else -1)
        return f"{xs[i]:.2f}@L{i}"

    for point, r in res.items():
        print(f"{point}: n={r['family']['n']}")
        for q in ("family", "threat"):
            within = " ".join(f"{lang} {best(r[q]['within'][lang])}" for lang in ("en", "tw", "en-tw"))
            cross = " ".join(f"{lang} {best(r[q]['from_en'][lang])}" for lang in ("tw", "en-tw"))
            print(f"  {q:7} within: {within} | trained on en, tested on: {cross} | chance {r[q]['chance']:.2f}")
        closer = [x for x in r["placement"]["closer_to_en"] if x == x]
        print(f"  en-tw nearer its English twin than its Twi twin: {min(closer):.2f}–{max(closer):.2f} across layers")
    print(f"saved {out}")


def first_turns(paths: list[str]) -> dict[str, list[dict]]:
    """Each episode's first turn, grouped by model, with what read_points needs."""
    by_model: dict[str, list[dict]] = {}
    for path in paths:
        for trace in load(path):
            ts = turns(trace)
            if not ts or "outcome" not in trace.get("rewards", {}):
                continue
            data = trace["task"]["data"]
            first = ts[0]
            by_model.setdefault(trace["agent"]["config"]["model"], []).append(
                {
                    "task_id": first.task_id,
                    "family": data["family"],
                    "category": data["category"],
                    "language": data["language"],
                    "seed": data["seed"],
                    "reward": trace["rewards"]["outcome"]["score"],
                    "messages": first.messages,
                    "tools": first.tools,
                    "response": first.response,
                }
            )
    return by_model


@app.local_entrypoint()
def read(traces: str, run: str, batch: int = 100):
    """Record the three read points for every episode in `traces` (comma-separated
    traces.jsonl paths), in batches of `batch` episodes per GPU container."""
    jobs = []
    for model, eps in first_turns(traces.split(",")).items():
        for i in range(0, len(eps), batch):
            jobs.append((model, Model(model_id=model).read_points.spawn(run, eps[i : i + batch])))
    rows: list[dict] = []
    for model, job in jobs:
        rows += [dict(r, model=model) for r in job.get()]
    out = Path(f"read-points-{run}.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        print(
            f"{model}: {len(mine)} episodes | prompt is a prefix: {sum(r['prefix_ok'] for r in mine)}"
            f" | reasoning found: {sum(r['reasoning_end'] is not None for r in mine)}"
            f" | action found: {sum(r['action_start'] is not None for r in mine)}"
        )
    print(f"details in {out}")


@app.local_entrypoint()
def main(traces: str):
    """`traces`: comma-separated vf-eval traces.jsonl paths."""
    by_model: dict[str, list[dict]] = {}
    for path in traces.split(","):
        for trace in load(path):
            model = trace["agent"]["config"]["model"]
            for t in turns(trace):
                by_model.setdefault(model, []).append(
                    {
                        "task_id": t.task_id,
                        "index": t.index,
                        "messages": t.messages,
                        "tools": t.tools,
                        "prompt_tokens": t.prompt_tokens,
                    }
                )
    calls = {model: Model(model_id=model).last_token.spawn(reqs) for model, reqs in by_model.items()}
    for model, call in calls.items():
        rows = call.get()
        same = sum(r["tokens"] == r["server_tokens"] for r in rows)
        print(f"{model}: {same}/{len(rows)} turns render to the server's prompt length")
        for r in rows:
            mark = "ok" if r["tokens"] == r["server_tokens"] else "DIFF"
            print(f"  {mark:4} {r['task_id']} turn {r['index']}: {r['tokens']} vs {r['server_tokens']}  next {r['next'][:2]}")
