"""Internal-measurement jobs on Modal: load a model on a GPU and look at what it computes on
the prompts it saw in sika-agent episodes. Entrypoints (`modal run sika_interp/modal_app.py::<name>`):

- main: re-render every turn and check it against the server's prompt length;
- read: the residual stream and MLP neurons at three read points of each episode;
- probe, link: per-layer probes, and the probes tied to each episode's outcome;
- tokens: token-level states of each request, for switch points;
- cues: which words carry the scam cue in Twi–English requests (masking);
- build_map: every neuron's scores and place, and cards for the strongest;
- steer_run: the steer lens on the map's candidates at scam decision points.

Versions match the vLLM install on the Prime pod (torch 2.13, transformers 5.19).
"""

import json
from contextlib import contextmanager
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
        # Qwen3.5's linear-attention layers: the optimised kernels instead of transformers'
        # slow reference code (same maths; vLLM serves Qwen with kernels of this kind).
        "flash-linear-attention==0.5.2",
    )
    .env({"HF_HOME": "/cache/hf"})
    # The environment itself, for the word tagger (wording pools, names, system prompt).
    .add_local_dir(
        # Repo path when building locally; inside the container this file sits at /root and
        # the image is already built, so the walk just stops at "/".
        Path(__file__).resolve().parent.parent.parent.parent / "environments" / "sika_agent",
        "/opt/sika_agent",
        copy=True,
        ignore=[".venv", "dist", "outputs", "**/__pycache__", "scripts/.cache"],
    )
    .run_commands("pip install /opt/sika_agent")
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

    def user_span(self, ids: list[int], text: str) -> tuple[int, list[str]] | None:
        """Where the user's message sits in the prompt: its first token, and what each prefix
        of its tokens decodes to (for `switching.align`). None if it can't be found."""
        start = len(ids) - 1
        while start > 0 and text not in self.tok.decode(ids[start:]):
            start -= 1
        if text not in self.tok.decode(ids[start:]):
            return None
        prefixes, end = [], start
        while end < len(ids):
            end += 1
            prefixes.append(self.tok.decode(ids[start:end]))
            if text in prefixes[-1]:
                return start, prefixes
        return None

    @modal.method()
    def token_states(self, run: str, batch: int, episodes: list[dict]) -> list[dict]:
        """For each episode's request, the residual stream at every TOKEN_LAYER_STEP-th layer
        for each token of the user's message, with its word's tag and whether it opens a
        switch; saved to the volume under tokens/<run>/<model>/. MLP neuron sums at switch
        and other word tokens, and at English and Twi words, of Twi–English requests are
        saved per batch for analyse_tokens."""
        import numpy as np
        import torch

        from sika_interp.switching import token_tags

        keep = list(range(TOKEN_LAYER_STEP, len(self.layers) + 1, TOKEN_LAYER_STEP))
        width = self.layers[0].mlp.down_proj.in_features
        sums = {
            f"{g}_{k}": np.zeros((len(self.layers), width)) for g in ("switch", "other", "en", "tw") for k in ("sum", "sq")
        } | {f"{g}_n": np.zeros(1) for g in ("switch", "other", "en", "tw")}
        base = Path("/cache/tokens") / run / self.model_id.split("/")[-1]
        base.mkdir(parents=True, exist_ok=True)
        rows = []
        for ep in episodes:
            ids = self.render(ep["messages"], ep["tools"])[0].tolist()
            text = ep["messages"][-1]["content"]
            span = self.user_span(ids, text)
            row = {"task_id": ep["task_id"], "language": ep["language"], "found": span is not None}
            if span is None:
                rows.append(row)
                continue
            start, prefixes = span
            tags, opens = token_tags(text, prefixes)
            positions = list(range(start, start + len(prefixes)))
            captured: dict[int, torch.Tensor] = {}

            def grab(i):
                def hook(module, args):
                    captured[i] = args[0][0, positions].float().cpu()

                return hook

            handles = [layer.mlp.down_proj.register_forward_pre_hook(grab(i)) for i, layer in enumerate(self.layers)]
            with torch.no_grad():
                out = self.model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True)
            for h in handles:
                h.remove()
            residual = torch.stack([out.hidden_states[k][0, positions].float().cpu() for k in keep], dim=1)
            np.savez(
                base / (ep["task_id"].replace("/", "_") + ".npz"),
                residual=residual.numpy().astype(np.float16),
                tags=np.array(tags),
                opens=np.array(opens),
                layers=np.array(keep),
            )
            if ep["language"] == "en-tw":
                mlp = torch.stack([captured[i] for i in range(len(self.layers))], dim=1).numpy()  # [tokens, layers, width]
                tags_a, opens_a = np.array(tags), np.array(opens)
                is_word = np.isin(tags_a, ["en", "tw", "mixed"])
                for group, mask in (
                    ("switch", opens_a & is_word),
                    ("other", ~opens_a & is_word),
                    ("en", tags_a == "en"),
                    ("tw", tags_a == "tw"),
                ):
                    sel = mlp[mask].astype(np.float64)
                    sums[f"{group}_sum"] += sel.sum(0)
                    sums[f"{group}_sq"] += (sel**2).sum(0)
                    sums[f"{group}_n"] += len(sel)
            row.update(tokens=len(prefixes), switches=int(sum(opens)))
            rows.append(row)
        np.savez(base / f"mlp-{batch:03d}.npz", **sums)
        cache.commit()
        return rows


    @modal.method()
    def masked_scores(self, run: str, categories: dict[str, str], episodes: list[dict], seed: int = 0) -> list[dict]:
        """The English-trained scam probe's score at the end of the request, every layer, for
        each request as written and with its English words, its Twi words, or as many random
        words as it has English words masked. The probe is trained here on this run's English
        request_end arrays (scam and injection tasks against the rest)."""
        import random

        import numpy as np
        import torch

        from sika_interp import probes
        from sika_interp.switching import MASK, masked
        from sika_interp.words import tag

        X, family, language, _ = probes.load(Path("/cache/acts") / run / self.model_id.split("/")[-1], "request_end")
        threat = np.array([categories[f] in THREATS for f in family])
        en = language == "en"
        fitted = [probes.probe().fit(X[en, j].astype(np.float64), threat[en]) for j in range(X.shape[1])]
        rng = random.Random(seed)
        rows = []
        for ep in episodes:
            text = ep["messages"][-1]["content"]
            words = tag(text)
            n_en = sum(w.tag == "en" for w in words)
            pick = {w.start for w in rng.sample(words, min(n_en, len(words)))}
            pieces, at = [], 0
            for w in words:
                pieces += [text[at : w.start], MASK if w.start in pick else w.text]
                at = w.end
            variants = {
                "as_written": text,
                "no_english": masked(text, {"en"}),
                "no_twi": masked(text, {"tw"}),
                "random": "".join(pieces) + text[at:],
            }
            scores = {}
            for name, variant in variants.items():
                messages = ep["messages"][:-1] + [dict(ep["messages"][-1], content=variant)]
                with torch.no_grad():
                    hidden = self.model(input_ids=self.render(messages, ep["tools"]), output_hidden_states=True).hidden_states
                states = np.stack([h[0, -1].float().cpu().numpy() for h in hidden]).astype(np.float64)
                scores[name] = [
                    round(float(fitted[j].predict_proba(states[j : j + 1])[0, list(fitted[j].classes_).index(True)]), 4)
                    for j in range(len(fitted))
                ]
            rows.append(
                {
                    "task_id": ep["task_id"],
                    "family": ep["family"],
                    "language": ep["language"],
                    "threat": categories[ep["family"]] in THREATS,
                    "reward": ep["reward"],
                    "english_words": n_en,
                    "twi_words": sum(w.tag == "tw" for w in words),
                    "scores": scores,
                }
            )
        return rows


    def decision_prefix(self, d: dict) -> tuple[list[int], dict[str, int]]:
        """Token ids up to where the model names a tool at decision `d` (its own reasoning
        kept, a tool call opened), and the first token of each tool name in that place."""
        placeholder = "zzplaceholderzz"

        def ids_with(name: str) -> list[int]:
            call = {"id": "call_0", "type": "function", "function": {"name": name, "arguments": "{}"}}
            response = {"reasoning_content": d["response"].get("reasoning_content") or "", "tool_calls": [call]}
            return self.render_turn(d["messages"], d["tools"], response)[1]

        base = ids_with(placeholder)
        at = self.tok.decode(base).rfind(placeholder)  # where the tool name begins, in characters

        def first_reaching(chars: int) -> int:
            lo, hi = 0, len(base)  # first k with decode(base[:k+1]) longer than `chars`
            while lo < hi:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if len(self.tok.decode(base[: mid + 1])) > chars else (mid + 1, hi)
            return lo

        cut = first_reaching(at)
        prefix = base[:cut]
        firsts = {}
        for name in PAY + PROTECT:
            other = ids_with(name)
            if other[:cut] == prefix:
                firsts[name] = other[cut]
        return prefix, firsts

    @modal.method()
    def steer(self, decisions: list[dict], neurons: list[list[int]], controls: int, dose: float = 0.8, seed: int = 0) -> dict:
        """Steer lens at decision points. For each decision, the model's pay-versus-protect
        margin (log-sum-exp of the logits of paying tools' first tokens minus protecting
        tools') when naming a tool, unsteered and with one push added at that position:
        each listed neuron's output direction, and `controls` random directions per layer
        used, at ±`dose` × the residual norm leaving the layer there. The prefix runs once;
        each push reuses a copy of its cache."""
        import copy

        import numpy as np
        import torch

        rng = np.random.default_rng(seed)
        used = sorted({layer for layer, _ in neurons})
        width = self.model.config.get_text_config().hidden_size
        directions = {
            ("neuron", layer, i): self.layers[layer].mlp.down_proj.weight[:, i].float()
            for layer, i in neurons
        }
        for layer in used:
            for c in range(controls):
                v = torch.tensor(rng.standard_normal(width), dtype=torch.float32)
                directions[("control", layer, c)] = v
        directions = {k: (v / v.norm()).to("cuda") for k, v in directions.items()}
        results = {"decisions": [], "effects": {}}
        for d in decisions:
            prefix, firsts = self.decision_prefix(d)
            pay = sorted({firsts[n] for n in PAY if n in firsts})
            protect = sorted({firsts[n] for n in PROTECT if n in firsts} - set(pay))
            if not pay or not protect:
                results.setdefault("skipped", []).append(
                    {"task_id": d["task_id"], "firsts": firsts, "tail": self.tok.decode(prefix[-12:])}
                )
                continue
            norms: dict[int, float] = {}

            def measure(i):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    norms[i] = float(hidden[0, -1].float().norm())

                return hook

            with torch.no_grad():
                cache = self.model(input_ids=torch.tensor([prefix[:-1]], device="cuda"), use_cache=True).past_key_values
                last = torch.tensor([[prefix[-1]]], device="cuda")

                def margin(push=None):
                    handle = None
                    if push is not None:
                        layer, vec = push

                        def add(module, args, output):
                            if isinstance(output, tuple):
                                return (output[0] + vec.to(output[0].dtype),) + tuple(output[1:])
                            return output + vec.to(output.dtype)

                        handle = self.layers[layer].register_forward_hook(add)
                    logits = self.model(input_ids=last, past_key_values=copy.deepcopy(cache), use_cache=True).logits[0, -1].float()
                    if handle:
                        handle.remove()
                    return float(torch.logsumexp(logits[pay], 0) - torch.logsumexp(logits[protect], 0))

                # The unsteered pass also records each used layer's residual norm at this position.
                handles = [self.layers[layer].register_forward_hook(measure(layer)) for layer in used]
                base = margin()
                for h in handles:
                    h.remove()
                for key, vec in directions.items():
                    _, layer, _ = key
                    for sign in (-1, 1):
                        delta = margin((layer, sign * dose * norms[layer] * vec)) - base
                        results["effects"].setdefault(f"{key[0]}:{layer}:{key[2]}:{sign:+d}", []).append(round(delta, 4))
            results["decisions"].append({k: d[k] for k in ("task_id", "language", "index")} | {"margin": round(base, 4)})
        return results

    def action_prefix(self, d: dict) -> dict:
        """The model's turn at decision `d` rendered with its own reasoning, then either a call
        to each tool (arguments "{}") or a reply: `renders` maps each tool to its token ids,
        `reply` is the reply's. `Pushes` works out where they part ways."""
        reasoning = d["response"].get("reasoning_content") or ""

        def calling(name: str) -> list[int]:
            call = {"id": "call_0", "type": "function", "function": {"name": name, "arguments": "{}"}}
            return self.render_turn(d["messages"], d["tools"], {"reasoning_content": reasoning, "tool_calls": [call]})[1]

        renders = {tool["function"]["name"]: calling(tool["function"]["name"]) for tool in d["tools"]}
        reply = self.render_turn(d["messages"], d["tools"], {"reasoning_content": reasoning, "content": "zzplaceholderzz"})[1]
        return {"renders": renders, "reply": reply}

    @modal.method()
    def steer_moves(
        self, run: str, decisions: list[dict], neurons: list[list[int]], controls: int, doses: list[float],
        seed: int = 0, diagnose: bool = False,
    ) -> list[dict]:
        """The steer lens over every move, at any decision. For each decision: the log-probability
        of replying and of each tool (read along the tools' name tokens: `Pushes`), unsteered and
        with one push added at the positions where the model chooses: each listed neuron's
        output direction, and `controls` random directions per layer used, at ±each dose × the
        residual norm leaving the layer there. Full arrays go to the volume under
        steer/<run>/<model>/; returns per-decision margins (right moves − wrong moves), with the
        directions (kind, layer, index) on the first row.

        Pushes are batched. The first decision tries each batching way against plain forward
        passes and keeps the first whose zero-push numbers match (moves with probability at
        least e^-5 within 0.25 nats, the bf16 floor); every later decision repeats that check
        for the way kept and records it. Shifts are measured from a zero push done the same way.
        `diagnose` runs only the checks, every way, on every decision."""
        import time

        import numpy as np
        import torch

        rng = np.random.default_rng(seed)
        used = sorted({layer for layer, _ in neurons})
        width = self.model.config.get_text_config().hidden_size
        meta = [("neuron", layer, i) for layer, i in neurons]
        vecs = [self.layers[layer].mlp.down_proj.weight[:, i].float() for layer, i in neurons]
        for layer in used:  # same draws, in the same order, as `steer`
            for c in range(controls):
                meta.append(("control", layer, c))
                vecs.append(torch.tensor(rng.standard_normal(width), dtype=torch.float32))
        directions = torch.stack([v.to("cuda") / v.norm() for v in vecs])
        rows = [(j, s, k) for j in range(len(meta)) for s in (-1, 1) for k in range(len(doses))]
        folder = Path("/cache/steer") / run / self.model_id.split("/")[-1]
        folder.mkdir(parents=True, exist_ok=True)
        layer_types = getattr(self.model.config.get_text_config(), "layer_types", None) or []
        order = ["tiled", "tiled-stepwise"]
        if all(t == "full_attention" for t in layer_types):
            order = ["siblings", "siblings+cache_position"] + order

        def ok(check: dict) -> bool:
            # Every way, one-token-at-a-time included, lands up to ~0.16 nats from the plain
            # passes on minority moves (bf16 rounding; more on long prompts); shifts are taken
            # from a zero push done the same way, so a way is kept if it stays within that floor.
            return "likely" in check and check["likely"] < 0.25

        def gap(check: dict) -> float:
            return check.get("likely", float("inf"))

        way, out = None, []
        for d in decisions:
            started = time.time()
            row = {k: d[k] for k in ("task_id", "family", "category", "language", "seed", "kind", "index", "reward", "took", "rule")}
            try:
                at = Pushes(self, d, used, [m[1] for m in meta], directions, doses)
                with at.hooked(), torch.no_grad():
                    if diagnose:
                        row["checks"] = {w: at.check(w) for w in order}
                        out.append(row | at.summary(None) | {"seconds": round(time.time() - started, 1)})
                        continue
                    checks = {}
                    if way is None:
                        for candidate in order:
                            checks[candidate] = at.check(candidate)
                            if ok(checks[candidate]):
                                way = candidate
                                break
                        if way is None:  # the fastest way that runs; judged against stepwise below
                            way = next(w for w in order if "likely" in checks[w])
                    else:
                        checks[way] = at.check(way)
                    flagged = False
                    if not ok(checks[way]) and way != "tiled-stepwise":
                        # Beyond the usual floor: is this prompt just noisier? Compare with the
                        # one-token-at-a-time route, the one the model generates with.
                        checks.setdefault("tiled-stepwise", at.check("tiled-stepwise"))
                        flagged = gap(checks[way]) > 2 * gap(checks["tiled-stepwise"]) + 0.1
                    effects, zero = at.run(way, rows)
                    effects, zero = effects.view(len(meta), 2, len(doses), -1).numpy(), zero.numpy()
                row |= at.summary(effects, zero)
                name = f"{d['task_id'].replace('/', '_')}_{d['kind']}.npz"
                np.savez_compressed(folder / name, base=at.base, zero=zero, effects=effects.astype(np.float16), labels=np.array(at.labels))
                row |= {"saved": name, "way": way, "checks": checks, "flagged": flagged, "batch": at.batch(way)}
            except Exception as e:  # noqa: BLE001 - keep the rest of the batch; the row says what failed
                row["error"] = f"{type(e).__name__}: {e}"[:500]
                torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - started, 1)
            out.append(row)
        cache.commit()
        return [dict(r, directions=meta) if i == 0 else r for i, r in enumerate(out)]

    @modal.method()
    def neuron_contexts(self, episodes: list[dict], neurons: list[list[int]], top: int = 8, window: int = 14) -> dict:
        """Where each listed MLP neuron fires most in the user's requests: per neuron and request
        language, the `top` requests by the neuron's peak activation over the request's tokens,
        each with the activations of the `window` tokens either side of the peak. Token texts
        come from decoded prefixes, so Twi letters split across tokens stay whole."""
        import heapq

        import torch

        by_layer: dict[int, list[int]] = {}
        for layer, i in neurons:
            by_layer.setdefault(layer, []).append(i)
        heaps: dict[tuple[int, int, str], list] = {}
        texts: dict[int, list[str]] = {}
        for e, ep in enumerate(episodes):
            ids = self.render(ep["messages"], ep["tools"])[0].tolist()
            span = self.user_span(ids, ep["messages"][-1]["content"])
            if span is None:
                continue
            start, prefixes = span
            pieces = [p[len(prefixes[k - 1]) :] if k else p for k, p in enumerate(prefixes)]
            n = len(pieces)
            acts: dict[int, torch.Tensor] = {}

            def grab(layer):
                idx = torch.tensor(by_layer[layer], device="cuda")

                def hook(module, args):
                    acts[layer] = args[0][0, start : start + n][:, idx].float().cpu()  # [tokens, neurons]

                return hook

            handles = [self.layers[layer].mlp.down_proj.register_forward_pre_hook(grab(layer)) for layer in by_layer]
            with torch.no_grad():
                self.model(input_ids=torch.tensor([ids[: start + n]], device="cuda"), logits_to_keep=1)
            for h in handles:
                h.remove()
            used = False
            for layer, idxs in by_layer.items():
                a = acts[layer]
                peak, where = a.max(0)
                for k, i in enumerate(idxs):
                    key = (layer, i, ep["language"])
                    heap = heaps.setdefault(key, [])
                    item = (float(peak[k]), e, int(where[k]))
                    if len(heap) < top or item[0] > heap[0][0]:
                        lo, hi = max(0, item[2] - window), min(n, item[2] + window + 1)
                        entry = (item[0], e, item[2], lo, [round(float(x), 3) for x in a[lo:hi, k]])
                        (heapq.heappush if len(heap) < top else heapq.heapreplace)(heap, entry)
                        used = True
            if used:
                texts[e] = pieces
        out: dict = {"neurons": {}, "episodes": {}}
        keep = set()
        for (layer, i, lang), heap in heaps.items():
            rows = sorted(heap, reverse=True)
            out["neurons"].setdefault(f"{layer}:{i}", {})[lang] = [
                {"peak": round(v, 3), "episode": e, "at": pos, "from": lo, "acts": w} for v, e, pos, lo, w in rows
            ]
            keep.update(e for _, e, *_ in rows)
        out["episodes"] = {e: {"task_id": episodes[e]["task_id"], "pieces": texts[e]} for e in keep}
        return out

    @modal.method()
    def profile_moves(self, d: dict, neurons: list[list[int]], way: str, repeats: int = 3) -> dict:
        """Seconds spent in each part of one batch of pushes at decision `d`, for tuning."""
        import copy
        import time

        import torch

        used = sorted({layer for layer, _ in neurons})
        directions = torch.stack([self.layers[l].mlp.down_proj.weight[:, i].float().cuda() for l, i in neurons])
        directions = directions / directions.norm(dim=1, keepdim=True)
        at = Pushes(self, d, used, [l for l, _ in neurons], directions, [0.8])
        n = at.batch(way)
        rows = [(j % len(neurons), 1, 0) for j in range(n)]
        times: dict[str, list[float]] = {}

        def timed(name, fn):
            torch.cuda.synchronize()
            t0 = time.time()
            out = fn()
            torch.cuda.synchronize()
            times.setdefault(name, []).append(round(time.time() - t0, 4))
            return out

        with at.hooked(), torch.no_grad():
            for _ in range(repeats):
                full = timed("fill", lambda: at.fill(rows, n))
                cache = timed("tile", lambda: _tile(at.head, n))
                at.cur = {i: f[:, : at.S] for i, f in full.items()}
                ids = torch.tensor([at.tail] * n, device="cuda")
                logits = timed("tail_forward", lambda: self.model(input_ids=ids, past_key_values=cache, use_cache=True).logits[:, -1])
                c = timed("node_copy", lambda: copy.deepcopy(cache))
                at.cur = {i: f[:, at.S : at.S + 1] for i, f in full.items()}
                tok = torch.full((n, 1), at.nodes[0][-1], device="cuda")
                logits = timed("node_forward", lambda: self.model(input_ids=tok, past_key_values=c, use_cache=True).logits[:, -1])
                timed("move_logp", lambda: at.move_logp(logits, logits, [logits] * at.K))
                del cache, c, logits
                timed("whole_batch", lambda: at.pushed(way, rows, n))
                times.setdefault("peak_gb", []).append(round(torch.cuda.max_memory_allocated() / 1e9, 1))
        per = sum(t.numel() * t.element_size() for _, _, t in _tensors(at.head) if t.dim() > 0 and t.shape[0] == 1)
        kinds = sorted({type(t).__name__ for t in [at.head]} | {type(x).__name__ for x in getattr(at.head, "layers", [])})
        return {"batch": n, "S": at.S, "K": at.K, "prompt": at.prompt_tokens, "cache_mb_per_copy": round(per / 1e6, 1),
                "cache_types": kinds, "seconds": times}

    def turn_logprobs(self, messages: list[dict], tools: list[dict], response: dict):
        """Log-probabilities over the vocabulary at each token of `response` (reasoning and
        action), teacher-forced after `messages` as the model saw them."""
        import torch

        prompt, full = self.render_turn(messages, tools, response)
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([full], device="cuda")).logits[0].float()
        end = len(full)
        if self.reasoning_prompt:  # placeholder messages follow the turn for Mistral's tokenizer
            end = next(
                (k for k in range(len(prompt), len(full)) if self.tok.decode([full[k]]).strip() in ("[TOOL_RESULTS]", "[INST]")),
                len(full),
            )
        return torch.log_softmax(logits[len(prompt) - 1 : end - 1], -1)

    @modal.method()
    def step_links(self, episode: dict) -> dict:
        """Thought-anchors links between an episode's steps, from the inside: for each step i
        (a user message, a tool result, or a model turn with its tool results) and each later
        model turn j, the mean KL divergence over turn j's tokens between the turn as the model
        saw it and with step i deleted from its history."""
        import torch

        turns_ = episode["turns"]  # [{messages, response}] in order
        last = turns_[-1]["messages"] + [dict(turns_[-1]["response"], role="assistant")]
        steps = episode_steps(last)
        base = [self.turn_logprobs(t["messages"], episode["tools"], t["response"]) for t in turns_]
        links, skipped = [], 0
        for i, group in enumerate(steps):
            for j, t in enumerate(turns_):
                if group[0] >= len(t["messages"]):
                    continue  # step i comes after turn j
                if group[0] == 1:
                    continue  # deleting the request leaves nothing to compare
                kept = [m for n, m in enumerate(t["messages"]) if n not in group]
                try:
                    altered = self.turn_logprobs(kept, episode["tools"], t["response"])
                except ValueError:  # e.g. two assistant turns in a row, which Mistral's tokenizer refuses
                    skipped += 1
                    continue
                n = min(len(altered), len(base[j]))
                kl = (base[j][:n].exp() * (base[j][:n] - altered[:n])).sum(-1).mean()
                links.append([i, j, round(float(kl), 4)])
        described = []
        for group in steps:
            m = last[group[0]]
            calls = [c["function"]["name"] for c in m.get("tool_calls") or []]
            described.append(
                {
                    "role": m["role"],
                    "calls": calls,
                    "text": (m.get("content") or "")[:300],
                    "reasoning": (m.get("reasoning_content") or "")[:600],
                    "results": [last[g]["content"][:200] for g in group[1:]],
                }
            )
        turn_step = [next(i for i, g in enumerate(steps) if g[0] == len(t["messages"])) for t in turns_]
        return {"task_id": episode["task_id"], "steps": described, "turn_step": turn_step, "links": links, "skipped": skipped}

    def to_token(self, ids: list[int]):
        """char → index of the token holding that character of decode(ids)."""
        lengths: dict[int, int] = {}

        def decoded(k: int) -> int:
            if k not in lengths:
                lengths[k] = len(self.tok.decode(ids[:k]))
            return lengths[k]

        def find(c: int) -> int:
            lo, hi = 0, len(ids)  # first k with decode(ids[:k+1]) longer than c
            while lo < hi:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if decoded(mid + 1) > c else (mid + 1, hi)
            return min(lo, len(ids) - 1)

        return find

    def full_attention_modules(self) -> list:
        """(layer index, attention module) for layers with full softmax attention."""
        return [(i, layer.self_attn) for i, layer in enumerate(self.layers) if hasattr(layer, "self_attn")]

    @modal.method()
    def suppress_links(self, episode: dict) -> dict:
        """Thought-anchors links by blocking attention: for each step i and each later model
        turn j, the mean KL divergence over turn j's tokens between the turn as the model saw
        it and with every later token barred from attending to step i's tokens, in every head
        of every full-attention layer (Qwen: 8 of 32; its linear-attention layers still carry
        the step). Steps as in `step_links`; the request can be blocked too."""
        import torch

        from sika_interp.spans import message_spans, to_tokens

        turns_ = episode["turns"]
        last = turns_[-1]["messages"] + [dict(turns_[-1]["response"], role="assistant")]
        steps = episode_steps(last)
        modules = self.full_attention_modules()
        state: dict = {"mask": None}

        def swap(module, args, kwargs):
            if state["mask"] is not None:
                kwargs["attention_mask"] = state["mask"]
            return args, kwargs

        handles = [m.register_forward_pre_hook(swap, with_kwargs=True) for _, m in modules]
        dtype = next(self.model.parameters()).dtype
        links, missing, checks = [], [], []
        try:
            for j, t in enumerate(turns_):
                prompt, full = self.render_turn(t["messages"], episode["tools"], t["response"])
                T = len(full)
                text = self.tok.decode(full)
                spans = message_spans(text, t["messages"])
                find = self.to_token(full)
                causal = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()

                def logprobs(allowed):
                    mask = torch.zeros(1, 1, T, T, dtype=dtype, device="cuda")
                    state["mask"] = mask.masked_fill_(~allowed, torch.finfo(dtype).min)
                    lp = self.turn_logprobs(t["messages"], episode["tools"], t["response"])
                    state["mask"] = None
                    return lp

                base = self.turn_logprobs(t["messages"], episode["tools"], t["response"])
                if j == 0:  # the hooked path with a plain causal mask must match the unhooked one
                    same = logprobs(causal)
                    checks.append(round(float((base.exp() * (base - same)).sum(-1).mean()), 6))
                for i, group in enumerate(steps):
                    if group[0] >= len(t["messages"]):
                        continue
                    mine = [s for s in spans if s.message in group]
                    if not mine:
                        missing.append([i, j])
                        continue
                    (a, _), (_, b) = to_tokens([mine[0]], find)[0], to_tokens([mine[-1]], find)[0]
                    allowed = causal.clone()
                    allowed[b:, a:b] = False
                    altered = logprobs(allowed)
                    kl = (base.exp() * (base - altered)).sum(-1).mean()
                    links.append([i, j, round(float(kl), 4), b - a])
        finally:
            for h in handles:
                h.remove()
        return {"task_id": episode["task_id"], "links": links, "missing": missing, "check_kl": checks,
                "layers_blocked": [i for i, _ in modules], "turn_step": [next(i for i, g in enumerate(steps) if g[0] == len(t["messages"])) for t in turns_]}

    @modal.method()
    def sentence_attention(self, items: list[dict], proximity: int = 4, block: int = 512) -> list[dict]:
        """For receiver heads (Thought Anchors): each item's full turn (prompt and response)
        split into sentences (system prompt, request, tool results, reasoning, action); per
        full-attention head, the mean attention from each sentence's tokens to each earlier
        sentence's, then each sentence's vertical score (mean attention it receives from
        sentences at least `proximity` later) and the kurtosis of those scores per head.

        Attention is read through the model's own attention interface, so the queries and keys
        are exactly the ones it uses (projected, normed, rotated, scaled). The output comes from
        the usual fused kernel; the weights are recomputed `block` query rows at a time in fp32
        and folded into the sentence grid, so a long episode never needs its whole T×T map
        (18k tokens would be 41 GB per layer)."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import AttentionInterface

        from sika_interp.spans import message_spans, sentences, to_tokens

        state: dict = {"M": None, "grids": {}}

        def sentence_grid(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
            heads, kv_heads = query.shape[1], key.shape[1]
            out = F.scaled_dot_product_attention(query, key, value, is_causal=query.shape[2] > 1, scale=scaling,
                                                 enable_gqa=heads != kv_heads)
            M = state["M"]
            if M is not None and query.shape[2] == M.shape[0]:
                T = query.shape[2]
                q = query[0].float()
                k = key[0].float().repeat_interleave(heads // kv_heads, dim=0)
                grid = torch.zeros(heads, M.shape[1], M.shape[1], device=query.device)
                cols = torch.arange(T, device=query.device)
                for q0 in range(0, T, block):
                    q1 = min(q0 + block, T)
                    scores = (q[:, q0:q1] @ k.transpose(1, 2)) * (scaling if scaling is not None else q.shape[-1] ** -0.5)
                    scores.masked_fill_(cols[None, None, :] > torch.arange(q0, q1, device=query.device)[None, :, None], float("-inf"))
                    probs = scores.softmax(-1)  # [heads, rows, T]
                    grid += torch.einsum("ta,htk->hak", M[q0:q1], probs) @ M
                state["grids"][module.layer_idx] = grid.cpu()
            return out.transpose(1, 2).contiguous(), None

        AttentionInterface.register("sentence_grid", sentence_grid)
        modules = self.full_attention_modules()
        out = []
        try:
            self.model.set_attn_implementation("sentence_grid")
            for item in items:
                messages = item["messages"] + [dict(item["response"], role="assistant")]
                prompt, full = self.render_turn(item["messages"], item["tools"], item["response"])
                text = self.tok.decode(full)
                spans = [s for sp in message_spans(text, messages) for s in sentences(text, sp)]
                ranges = to_tokens(spans, self.to_token(full))
                C, T = len(spans), len(full)
                M = torch.zeros(T, C, device="cuda")
                for c, (a, b) in enumerate(ranges):
                    M[a:b, c] = 1.0 / (b - a)
                state["M"], state["grids"] = M, {}
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([full], device="cuda"), logits_to_keep=1)
                state["M"] = None
                grid = torch.stack([state["grids"][getattr(m, "layer_idx", i)] for i, m in modules]).numpy()
                rows, cols = np.indices((C, C))
                masked = np.where(rows >= cols + proximity, grid, np.nan)  # from row to column
                with np.errstate(invalid="ignore"):
                    vertical = np.nanmean(masked, axis=2)  # mean over later rows → [layers, heads, C]
                v = vertical[..., : max(C - proximity, 1)]
                mu, sd = np.nanmean(v, -1, keepdims=True), np.nanstd(v, -1, keepdims=True)
                kurt = np.nanmean(((v - mu) / np.where(sd > 0, sd, np.nan)) ** 4, -1) - 3
                out.append(
                    {
                        "task_id": item["task_id"],
                        "language": item["language"],
                        "turn": item.get("index", 0),
                        "layers": [i for i, _ in modules],
                        "sentences": [
                            {"text": text[s.start : s.end][:300], "message": s.message, "part": s.part,
                             "role": messages[s.message]["role"], "tokens": b - a}
                            for s, (a, b) in zip(spans, ranges)
                        ],
                        "vertical": vertical.astype(np.float16),
                        "kurtosis": kurt.astype(np.float32),
                        "prompt_tokens": len(prompt),
                        "tokens": T,
                    }
                )
        finally:
            state["M"] = None
            self.model.set_attn_implementation("sdpa")
        return out


def episode_steps(messages: list[dict]) -> list[list[int]]:
    """An episode's steps: every message after the system prompt, a model turn grouped with
    its tool results. Each step is a list of message indices."""
    steps, k = [], 1
    while k < len(messages):
        group = [k]
        if messages[k]["role"] == "assistant":
            while k + 1 < len(messages) and messages[k + 1]["role"] == "tool":
                k += 1
                group.append(k)
        steps.append(group)
        k += 1
    return steps


class Pushes:
    """One decision for `Model.steer_moves`. The model chooses over a few positions: the token
    before a tool call and a reply part ways (reply or call), then the tool's name, read token
    by token along `decisions.name_trie` until each tool is told apart. The `tail` runs from
    the first of these to the end of the shared prefix; trie `nodes` add one token each.
    Unsteered numbers come from plain forward passes of the whole sequence; pushed ones from
    batched passes after a shared cache of the prefix.

    Ways to batch n pushes:
    - "siblings": the n copies of tail and nodes run as one sequence after the cache, each
      copy masked from the others and given its own positions (a tree inside each copy).
      Needs full attention in every layer: a recurrent layer would carry one copy into the next.
    - "tiled": the cache copied n times along the batch, the tail fed at once, each node then
      fed from a copy of its parent's cache; "tiled-stepwise" feeds the tail a token at a time."""

    def __init__(self, owner: "Model", d: dict, used: list[int], dir_layer: list[int], directions, doses: list[float]):
        import torch

        from sika_interp.decisions import REPLY, name_trie

        self.m, self.d, self.used, self.dir_layer, self.directions, self.doses = owner, d, used, dir_layer, directions, doses
        self.width = directions.shape[1]
        ap = owner.action_prefix(d)
        trie = name_trie(ap["renders"], ap["reply"])
        self.trie = trie
        cut, open_at = trie["cut"], trie["open_at"]
        prefix = ap["renders"][sorted(ap["renders"])[0]][:cut]
        self.L = open_at - 1 if open_at is not None else cut - 1
        self.tail, self.prompt_tokens = prefix[self.L :], cut
        self.S, self.K = len(self.tail), len(trie["nodes"])
        self.nodes = trie["nodes"]
        self.parent = [self.nodes.index(n[:-1]) if len(n) > 1 else -1 for n in self.nodes]
        self.labels = [REPLY] + list(trie["labels"])
        self.groups = {k: v["members"] for k, v in trie["labels"].items()}
        self.cur: dict[int, torch.Tensor] = {}
        # Unsteered: plain passes over the prefix and over the prefix plus each node's tokens,
        # recording the residual norm leaving each used layer at the tail (pushes scale by it).
        self.norms: dict[int, torch.Tensor] = {}
        handles = [owner.layers[i].register_forward_hook(self._measure(i)) for i in used]
        with torch.no_grad():
            # Only the last positions' logits: the whole prompt's would be ~1.5 GB per pass.
            logits = owner.model(input_ids=torch.tensor([prefix], device="cuda"), logits_to_keep=self.S).logits[0]
            for h in handles:
                h.remove()
            at_nodes = [
                owner.model(input_ids=torch.tensor([prefix + list(n)], device="cuda"), logits_to_keep=1).logits[0]
                for n in self.nodes
            ]
            self.truth = self.move_logp(logits[:1], logits[-1:], at_nodes)[0]
            self.head = owner.model(input_ids=torch.tensor([prefix[: self.L]], device="cuda"), use_cache=True).past_key_values
        self.base = self.truth.cpu().numpy()

    def _measure(self, i: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.norms[i] = hidden[0, -self.S :].float().norm(dim=-1)

        return hook

    def _add(self, i: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + self.cur[i].to(hidden.dtype).reshape(hidden.shape)
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

        return hook

    @contextmanager
    def hooked(self):
        handles = [self.m.layers[i].register_forward_hook(self._add(i)) for i in self.used]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def move_logp(self, first, root, at_nodes):
        """Log-probabilities of [reply, each label] from the logits ([b, vocab]) where the tool
        call opens, at the end of the tail, and at each trie node."""
        import torch

        lsm = [torch.log_softmax(x.float(), -1) for x in at_nodes]
        lsm_root = torch.log_softmax(root.float(), -1)
        cols = []
        for label in self.labels[1:]:
            total = 0
            for node, token in self.trie["labels"][label]["steps"]:
                total = total + (lsm_root if node < 0 else lsm[node])[:, token]
            cols.append(total)
        names = torch.stack(cols, 1)
        if self.trie["open_at"] is None:
            return torch.cat([torch.full_like(names[:, :1], float("-inf")), names], 1)
        opened = torch.log_softmax(first.float(), -1)[:, self.trie["opener"]]
        reply = torch.log1p(-opened.exp().clamp(max=1 - 1e-6))
        return torch.cat([reply[:, None], opened[:, None] + names], 1)

    def fill(self, chunk: list[tuple[int, int, int]], n: int) -> dict:
        """Per layer, each of `n` slots' push at the tail and node positions ([n, S+K, width]);
        nodes get the push scaled by the norm at the end of the tail. Unused slots get none."""
        import torch

        full = {i: torch.zeros(n, self.S + self.K, self.width, device="cuda") for i in self.used}
        for b, (j, s, k) in enumerate(chunk):
            i = self.dir_layer[j]
            scale = torch.cat([self.norms[i], self.norms[i][-1:].expand(self.K)])
            full[i][b] = (s * self.doses[k]) * scale[:, None] * self.directions[j][None, :]
        return full

    def batch(self, way: str) -> int:
        """Pushes per pass. Siblings add only n×(S+K) tokens to one shared cache. Tiling holds
        the tiled cache, its copy for the trie nodes (K times over when they run as one pass)
        and the extension each pass adds: about (1 + 2K) copies at the peak."""
        import torch

        total = torch.cuda.get_device_properties(0).total_memory
        free = total - torch.cuda.memory_allocated() - 12e9
        if way.startswith("siblings"):
            return 384 if total > 60e9 else 128
        per = sum(t.numel() * t.element_size() for _, _, t in _tensors(self.head) if t.dim() > 0 and t.shape[0] == 1)
        # The node pass holds the tiled cache, its K-fold repeat and, while a layer appends, a
        # second K-fold copy of that layer: budget 2 + 3K copies (and 4 without the node pass).
        copies = 2 + 3 * max(self.K, 1) if self.flat_nodes else 4
        return max(1, min(256, int(free // (copies * max(per, 1)))))

    @property
    def flat_nodes(self) -> bool:
        """Every trie node hangs off the end of the tail, so they can run as one pass."""
        return all(p < 0 for p in self.parent)

    def pushed(self, way: str, chunk: list[tuple[int, int, int]], n: int):
        """[n, labels] move log-probabilities with the chunk's pushes."""
        import copy

        import torch

        full = self.fill(chunk, n)
        S, K, L, model = self.S, self.K, self.L, self.m.model
        if way.startswith("siblings"):
            B = S + K
            ids = torch.tensor([(self.tail + [n_[-1] for n_ in self.nodes]) * n], device="cuda")
            depth = [L + S - 1 + len(node) for node in self.nodes]
            pos = torch.tensor([L + p for p in range(S)] + depth, device="cuda").repeat(n)[None]
            inner = torch.zeros(B, B, dtype=torch.bool, device="cuda")
            inner[:S, :S] = torch.ones(S, S, dtype=torch.bool, device="cuda").tril()
            for a in range(K):
                inner[S + a, :S] = True
                inner[S + a, S + a] = True
                up = self.parent[a]
                while up >= 0:
                    inner[S + a, S + up] = True
                    up = self.parent[up]
            allowed = torch.zeros(n * B, L + n * B, dtype=torch.bool, device="cuda")
            allowed[:, :L] = True
            for b in range(n):
                allowed[b * B : (b + 1) * B, L + b * B : L + (b + 1) * B] = inner
            dtype = next(model.parameters()).dtype
            mask = torch.zeros(1, 1, n * B, L + n * B, dtype=dtype, device="cuda").masked_fill_(~allowed, torch.finfo(dtype).min)
            kwargs = {"position_ids": pos, "attention_mask": mask}
            if way == "siblings+cache_position":
                kwargs["cache_position"] = torch.arange(L, L + n * B, device="cuda")
            self.cur = full
            logits = model(input_ids=ids, past_key_values=copy.deepcopy(self.head), use_cache=True, **kwargs).logits[0].view(n, B, -1)
            return self.move_logp(logits[:, 0], logits[:, S - 1], [logits[:, S + a] for a in range(K)])
        cache = _tile(self.head, n)
        ids = torch.tensor([self.tail] * n, device="cuda")
        if way == "tiled":
            self.cur = {i: f[:, :S] for i, f in full.items()}
            logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits
            first, root = logits[:, 0], logits[:, -1]
        else:
            for p in range(S):
                self.cur = {i: f[:, p : p + 1] for i, f in full.items()}
                logits = model(input_ids=ids[:, p : p + 1], past_key_values=cache, use_cache=True).logits[:, -1]
                first = logits if p == 0 else first
            root = logits
        if K and self.flat_nodes:
            # All nodes in one pass: each slot's cache repeated K times, one node token per copy.
            wide = _rebatch(cache, n, lambda t: t.repeat_interleave(K, dim=0))
            self.cur = {i: f[:, S:].reshape(n * K, 1, -1) for i, f in full.items()}
            toks = torch.tensor([[node[-1]] for node in self.nodes] * n, device="cuda")
            out = model(input_ids=toks, past_key_values=wide, use_cache=True).logits[:, -1].view(n, K, -1)
            return self.move_logp(first, root, [out[:, a] for a in range(K)])
        at_nodes, caches = [], {}
        for a, node in enumerate(self.nodes):
            # Each node continues a fresh copy of its parent's cache (feeding a token extends it).
            c = copy.deepcopy(cache if self.parent[a] < 0 else caches[self.parent[a]])
            self.cur = {i: f[:, S + a : S + a + 1] for i, f in full.items()}
            tok = torch.full((n, 1), node[-1], device="cuda")
            at_nodes.append(model(input_ids=tok, past_key_values=c, use_cache=True).logits[:, -1])
            if a in self.parent:
                caches[a] = c
        return self.move_logp(first, root, at_nodes)

    def check(self, way: str) -> dict:
        """How far a zero-push batch done this way lands from the plain passes: the largest gap
        over moves the model gives at least e^-5 and at least e^-10; or the error."""
        try:
            got = self.pushed(way, [], min(self.batch(way), 8))
        except Exception as e:  # noqa: BLE001 - a way the model's code doesn't support
            return {"error": f"{type(e).__name__}: {e}"[:300]}
        gap = (got - self.truth[None]).abs().max(0).values
        return {
            "likely": float(gap[self.truth > -5].max()),
            "plausible": float(gap[self.truth > -10].max()),
            "worst_move": self.labels[int(torch_argmax(gap, self.truth > -10))],
        }

    def run(self, way: str, rows: list[tuple[int, int, int]]):
        """Every row's pushed move log-probabilities, and a zero push done the same way: shifts
        are measured from it, so rounding differences between this way and the plain passes
        (bf16; up to ~0.15 nats on minority moves) cancel."""
        import torch

        n = self.batch(way)
        zero = self.pushed(way, [], n)[0].cpu()
        effects = torch.empty(len(rows), len(self.labels))
        for c0 in range(0, len(rows), n):
            chunk = rows[c0 : c0 + n]
            effects[c0 : c0 + len(chunk)] = self.pushed(way, chunk, n)[: len(chunk)].cpu()
        return effects, zero

    def summary(self, effects, zero=None) -> dict:
        """What the local side keeps: moves, the unsteered numbers (plain passes), and for each
        margin the decision's rule defines (`decisions.margin_sides`) its unsteered value and
        each push's shift of it from the zero push ([directions, signs, doses])."""
        import numpy as np

        from sika_interp.decisions import margin_sides

        out = {
            "labels": self.labels,
            "groups": self.groups,
            "base": [round(float(x), 4) for x in self.base],
            "tail": self.m.tok.decode(self.tail),
            "nodes": [self.m.tok.decode(list(n)) for n in self.nodes],
            "prompt_tokens": self.prompt_tokens,
            "tail_tokens": self.S,
        }
        for name, (a, b) in margin_sides(self.labels, self.groups, self.d["rule"]).items():
            base = np.logaddexp.reduce(self.base[a]) - np.logaddexp.reduce(self.base[b])
            out[f"{name}_margin"] = round(float(base), 4)
            if effects is not None:
                z = np.asarray(zero)
                start = np.logaddexp.reduce(z[a]) - np.logaddexp.reduce(z[b])
                pushed = np.logaddexp.reduce(effects[..., a], -1) - np.logaddexp.reduce(effects[..., b], -1)
                out[f"{name}_shift"] = np.round(pushed - start, 3).tolist()  # [directions, signs, doses]
                out[f"{name}_route_gap"] = round(float(start - base), 4)
        return out


def torch_argmax(values, where) -> int:
    """Index of the largest of `values` where `where` holds."""
    import torch

    return int(torch.where(where, values, torch.full_like(values, -1.0)).argmax())


def _tensors(obj, seen: set | None = None):
    """(holder, key, tensor) for every tensor inside a cache object."""
    import torch

    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("transformers"):
        items = list(vars(obj).items())
    else:
        return
    for key, value in items:
        if torch.is_tensor(value):
            yield obj, key, value
        else:
            yield from _tensors(value, seen)


def _rebatch(cache, batch: int, fn):
    """A copy of a cache with `fn` applied to every tensor whose first dimension is the batch
    (size `batch`); the other objects are copied as they are."""
    import copy

    keep = {id(t): t for _, _, t in _tensors(cache) if t.dim() > 0 and t.shape[0] == batch}
    # Seeding deepcopy's memo with None skips copying the batch tensors (they are replaced
    # below), so a large cache is copied once, by `fn`, not twice.
    out = copy.deepcopy(cache, {i: None for i in keep})
    _replace_none(out, cache, keep, fn)
    return out


def _replace_none(copied, original, keep: dict, fn, seen: set | None = None) -> None:
    """Walk `copied` alongside `original`; where `original` holds a batch tensor, put fn(it)."""
    import torch

    seen = set() if seen is None else seen
    if id(original) in seen:
        return
    seen.add(id(original))
    if isinstance(original, dict):
        pairs = [(k, original[k], copied[k]) for k in original]
    elif isinstance(original, list):
        pairs = [(k, original[k], copied[k]) for k in range(len(original))]
    elif hasattr(original, "__dict__") and type(original).__module__.startswith("transformers"):
        pairs = [(k, v, getattr(copied, k)) for k, v in vars(original).items()]
    else:
        return
    for key, o, c in pairs:
        if torch.is_tensor(o) and id(o) in keep:
            value = fn(o)
            if isinstance(copied, (dict, list)):
                copied[key] = value
            else:
                setattr(copied, key, value)
        elif not torch.is_tensor(o):
            _replace_none(c, o, keep, fn, seen)


def _tile(cache, n: int):
    """A copy of a batch-1 cache repeated n times along the batch."""
    return _rebatch(cache, 1, lambda t: t.expand(n, *t.shape[1:]).contiguous())


PAY = ("initiate_transfer", "confirm_transfer")
"""Tools that move money to a person: the harmful choice at a scam decision point."""
PROTECT = ("report_fraud", "ask_user", "cancel_transfer")
"""Tools that protect the user at a scam decision point."""
THREATS = ("scams", "injection")
"""Categories where the right move is to refuse: the target of the threat probe."""


TOKEN_LAYER_STEP = 4
"""Token-level states are kept at every 4th layer: enough to see where things change."""


@app.function(volumes={"/cache": cache}, cpu=16, memory=131072, timeout=7200)
def neuron_map(run: str, model: str, token_run: str, categories: dict[str, str], per_layer: int = 3) -> dict:
    """Every MLP neuron of `model`, for the map: a 2-D place per layer (its write direction's
    top two principal components across the layer), and scores from the study's episodes:
    scam or injection against the rest at the end of the request and at the action, Twi
    against English and Twi–English against English at the end of the request, language
    entropy, and from the token job, switch tokens against other words and English against
    Twi words inside Twi–English requests. Candidates (top `per_layer` per score per layer)
    get a card: logit lens and direct-path wiring. Saved under maps/<run>/<model>/."""
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from sika_interp import neurons

    name = model.split("/")[-1]
    acts = Path("/cache/acts") / run / name
    A_req, family, language, _ = neurons.load_mlp(acts, "request_end")
    A_act, family_act, language_act, _ = neurons.load_mlp(acts, "action_start")
    threat = np.array([categories[f] in THREATS for f in family])
    threat_act = np.array([categories[f] in THREATS for f in family_act])
    en_tw = np.isin(language, ["en", "tw"])
    en_mix = np.isin(language, ["en", "en-tw"])
    pref = neurons.language_preference(A_req, language)
    scores = {
        "scam_at_request": neurons.selectivity(A_req, threat),
        "scam_at_action": neurons.selectivity(A_act, threat_act),
        "twi_over_english": neurons.selectivity(A_req[en_tw], language[en_tw] == "tw"),
        "mixed_over_english": neurons.selectivity(A_req[en_mix], language[en_mix] == "en-tw"),
        "language_specific": np.log(3) - pref["entropy"],
    }
    sums = [np.load(p) for p in sorted((Path("/cache/tokens") / token_run / name).glob("mlp-*.npz"))]
    if sums:
        total = {k: sum(s[k] for s in sums) for k in sums[0].files}

        def d(a: str, b: str) -> np.ndarray:
            ma, mb = total[f"{a}_sum"] / total[f"{a}_n"], total[f"{b}_sum"] / total[f"{b}_n"]
            va = total[f"{a}_sq"] / total[f"{a}_n"] - ma**2
            vb = total[f"{b}_sq"] / total[f"{b}_n"] - mb**2
            return ((ma - mb) / np.sqrt((va + vb) / 2 + 1e-6)).astype(np.float32)

        scores["switch"] = d("switch", "other")
        scores["english_word_in_mix"] = d("en", "tw")
    del A_req, A_act

    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, device_map="cpu")
    layers = next(m for n, m in net.named_modules() if n.endswith("language_model.layers") and isinstance(m, torch.nn.ModuleList))
    n_layers, width = len(layers), layers[0].mlp.down_proj.in_features
    hidden = layers[0].mlp.down_proj.out_features
    # Every neuron's write direction and read (gate) direction, one row each, layer by layer.
    writes = np.empty((n_layers * width, hidden), np.float32)
    reads = np.empty((n_layers * width, hidden), np.float32)
    for k, layer in enumerate(layers):
        writes[k * width : (k + 1) * width] = layer.mlp.down_proj.weight.detach().float().numpy().T
        reads[k * width : (k + 1) * width] = layer.mlp.gate_proj.weight.detach().float().numpy()
    norm = next(m for n, m in net.named_modules() if n.endswith("language_model.norm")).weight.detach().float().numpy()
    if "qwen" in model.lower():
        norm = 1 + norm  # Qwen3.5's RMSNorm scales by 1 + weight
    unembed = net.get_output_embeddings().weight.detach().float().numpy()
    del net

    places = []
    for k in range(n_layers):
        cols = writes[k * width : (k + 1) * width]
        cols = cols - cols.mean(0)
        _, _, vt = np.linalg.svd(cols[:: max(1, width // 4096)], full_matrices=False)
        places.append((cols @ vt[:2].T).astype(np.float16))  # [width, 2]

    cards = []
    for layer, i, reason in neurons.candidates(scores, per_layer):
        row = layer * width + i
        up, down = neurons.logit_lens(writes[row], unembed, norm, k=8)
        feeds = neurons.wiring(writes[row], reads[(layer + 1) * width :], k=6)
        fed_by = neurons.wiring_up(reads[row], writes[: layer * width], k=6)
        cards.append(
            {
                "layer": layer,
                "neuron": i,
                "reason": reason,
                "scores": {k: round(float(v[layer, i]), 3) for k, v in scores.items()},
                "language_mean": [round(float(x), 4) for x in pref["mean"][:, layer, i]],
                "promotes": [[tok.decode([t]), round(v, 3)] for t, v in up],
                "suppresses": [[tok.decode([t]), round(v, 3)] for t, v in down],
                "feeds": [[layer + 1 + r // width, r % width, round(c, 3)] for r, c in feeds],
                "fed_by": [[r // width, r % width, round(c, 3)] for r, c in fed_by],
            }
        )
    base = Path("/cache/maps") / run / name
    base.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        base / "map.npz",
        places=np.stack(places),
        **{k: v.astype(np.float16) for k, v in scores.items()},
        prefers=pref["prefers"].astype(np.int8),
    )
    (base / "cards.json").write_text(json.dumps(cards, ensure_ascii=False))
    cache.commit()
    return {"cards": len(cards), "layers": n_layers, "width": width, "scores": list(scores)}


@app.function(volumes={"/cache": cache}, cpu=8, memory=65536, timeout=7200)
def analyse_tokens(run: str, model: str) -> dict:
    """What the token-level states say about Twi–English, per kept layer:
    - a probe trained to tell English from Twi requests' tokens, read on Twi–English tokens
      by their word's language (are English islands read as English, or as Twi context?);
    - how far the state jumps between consecutive tokens at switch points versus elsewhere;
    - MLP neurons that fire differently at switch tokens, and on English versus Twi words
      inside Twi–English requests (top 20 per layer, by standardised difference)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    base = Path("/cache/tokens") / run / model.split("/")[-1]
    eps = []
    for path in sorted(base.glob("eval_*.npz")):
        with np.load(path) as z:
            eps.append({k: z[k] for k in ("residual", "tags", "opens", "layers")} | {"language": path.stem.split("_")[-2]})
    layers = eps[0]["layers"].tolist()
    out: dict = {"layers": layers, "episodes": {lang: sum(e["language"] == lang for e in eps) for lang in ("en", "tw", "en-tw")}}
    word = lambda e: np.isin(e["tags"], ["en", "tw", "mixed"])  # noqa: E731
    mono = [e for e in eps if e["language"] in ("en", "tw")]
    mixed = [e for e in eps if e["language"] == "en-tw"]
    reading: dict = {}
    jumps: dict = {"switch": [], "other": []}
    for j in range(len(layers)):
        X = np.concatenate([e["residual"][word(e), j] for e in mono]).astype(np.float32)
        y = np.concatenate([np.full(word(e).sum(), e["language"] == "tw") for e in mono])
        probe = LogisticRegression(C=0.1, max_iter=2000).fit((X - X.mean(0)) / (X.std(0) + 1e-6), y)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        for side in ("en", "tw", "mixed"):
            states = [e["residual"][e["tags"] == side, j] for e in mixed]
            S = np.concatenate([s for s in states if len(s)]).astype(np.float32)
            reading.setdefault(side, []).append(float(probe.predict_proba((S - mu) / sd)[:, 1].mean()))
        sw, ot = [], []
        for e in mixed:
            R = e["residual"][:, j].astype(np.float32)
            R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-6)
            step = 1 - (R[1:] * R[:-1]).sum(1)
            is_word = word(e)[1:]
            sw += step[e["opens"][1:] & is_word].tolist()
            ot += step[~e["opens"][1:] & is_word].tolist()
        jumps["switch"].append(float(np.mean(sw)))
        jumps["other"].append(float(np.mean(ot)))
    out["reads_as_twi"] = reading
    out["jump"] = jumps

    sums = [np.load(p) for p in sorted(base.glob("mlp-*.npz"))]
    total = {k: sum(s[k] for s in sums) for k in sums[0].files}

    def top(a: str, b: str) -> list[list[list[float]]]:
        ma, mb = total[f"{a}_sum"] / total[f"{a}_n"], total[f"{b}_sum"] / total[f"{b}_n"]
        va = total[f"{a}_sq"] / total[f"{a}_n"] - ma**2
        vb = total[f"{b}_sq"] / total[f"{b}_n"] - mb**2
        z = (ma - mb) / np.sqrt((va + vb) / 2 + 1e-6)
        return [[[int(i), round(float(z[layer, i]), 2)] for i in np.argsort(-np.abs(z[layer]))[:20]] for layer in range(z.shape[0])]

    out["neurons"] = {
        "switch_vs_other": top("switch", "other"),
        "en_vs_tw_words": top("en", "tw"),
        "counts": {k.removesuffix("_n"): int(np.asarray(v).ravel()[0]) for k, v in total.items() if k.endswith("_n")},
    }
    return out


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
    if out.exists():  # a later batch for the same run: keep earlier episodes, replace repeats
        new = {(r["model"], r["task_id"]) for r in rows}
        rows = [r for r in json.loads(out.read_text()) if (r["model"], r["task_id"]) not in new] + rows
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
def tokens(traces: str, run: str, batch: int = 100):
    """Token-level states for the user's message in every episode of `traces`
    (comma-separated), then `analyse_tokens`; prints the Twi–English summary and saves it."""
    jobs = []
    for model, eps in first_turns(traces.split(",")).items():
        for i in range(0, len(eps), batch):
            jobs.append((model, Model(model_id=model).token_states.spawn(run, i // batch, eps[i : i + batch])))
    rows: dict[str, list[dict]] = {}
    for model, job in jobs:
        rows.setdefault(model, []).extend(job.get())
    for model, mine in rows.items():
        print(f"{model}: {sum(r['found'] for r in mine)}/{len(mine)} messages located")
        res = analyse_tokens.remote(run, model)
        out = Path(f"tokens-{run}-{model.split('/')[-1]}.json")
        out.write_text(json.dumps(res))
        print(f"  layers {res['layers']}  episodes {res['episodes']}")
        for side, xs in res["reads_as_twi"].items():
            print(f"  Twi–English {side:5} words read as Twi: " + " ".join(f"{x:.2f}" for x in xs))
        print("  state jump at switch points: " + " ".join(f"{x:.3f}" for x in res["jump"]["switch"]))
        print("  state jump elsewhere:        " + " ".join(f"{x:.3f}" for x in res["jump"]["other"]))
        print(f"  saved {out}")


@app.local_entrypoint()
def cues(traces: str, run: str, points: str, controls: int = 60, batch: int = 40):
    """Which words carry the scam cue in Twi–English requests: `masked_scores` on every
    Twi–English scam or injection request in `traces`, plus `controls` other requests as a
    false-alarm check. `points`: the read-points JSON from `read` (categories)."""
    categories = {r["family"]: r["category"] for r in json.loads(Path(points).read_text())}
    jobs = []
    for model, eps in first_turns(traces.split(",")).items():
        mixed = [e for e in eps if e["language"] == "en-tw"]
        chosen = [e for e in mixed if categories[e["family"]] in THREATS]
        chosen += [e for e in mixed if categories[e["family"]] not in THREATS][:controls]
        for i in range(0, len(chosen), batch):
            jobs.append((model, Model(model_id=model).masked_scores.spawn(run, categories, chosen[i : i + batch])))
    rows: dict[str, list[dict]] = {}
    for model, job in jobs:
        rows.setdefault(model, []).extend(job.get())
    for model, mine in rows.items():
        out = Path(f"cues-{run}-{model.split('/')[-1]}.json")
        out.write_text(json.dumps(mine))
        layers = len(mine[0]["scores"]["as_written"])
        print(f"{model}: {len(mine)} requests  (scam score = English-trained probe at the end of the request)")
        for threat in (True, False):
            group = [r for r in mine if r["threat"] == threat]
            print(f"  {'scam/injection' if threat else 'other (control)'} n={len(group)}")
            for variant in ("as_written", "no_english", "no_twi", "random"):
                means = [sum(r["scores"][variant][j] for r in group) / len(group) for j in range(4, layers, 4)]
                print(f"    {variant:11} " + " ".join(f"{m:.2f}" for m in means))
        print(f"  saved {out}")


def decision_points(paths: list[str], per_language: int, families: tuple[str, ...] = ()) -> dict[str, list[dict]]:
    """Scam-task turns that follow the user's answer (a later user message, or an ask_user
    result), grouped by model: up to `per_language` per language, spread over episodes."""
    by_model: dict[str, list[dict]] = {}
    for path in paths:
        for trace in load(path):
            data = trace["task"]["data"]
            if families and data["family"] not in families:
                continue
            for t in turns(trace):
                prev = t.messages[-1]
                if (prev["role"] == "user" and len(t.messages) > 2) or (prev["role"] == "tool" and prev.get("name") == "ask_user"):
                    by_model.setdefault(trace["agent"]["config"]["model"], []).append(
                        {"task_id": t.task_id, "language": t.language, "index": t.index, "family": data["family"],
                         "messages": t.messages, "tools": t.tools, "response": t.response}
                    )
                    break  # the first decision of each episode
    return {
        model: [d for lang in ("en", "tw", "en-tw") for d in [x for x in ds if x["language"] == lang][:per_language]]
        for model, ds in by_model.items()
    }


SCAM_FAMILIES = ("fake_refund", "pin_request", "promo_fee", "lender_impersonation")


@app.local_entrypoint()
def steer_test(traces: str, neurons: str = "16:5,20:7", controls: int = 4):
    """A small steer-lens run on the decision points in `traces`, for checking the harness."""
    picked = [[int(a) for a in n.split(":")] for n in neurons.split(",")]
    for model, ds in decision_points(traces.split(","), per_language=2, families=SCAM_FAMILIES).items():
        res = Model(model_id=model).steer.remote(ds, picked, controls)
        print(model, "decisions:", res["decisions"], "skipped:", res.get("skipped"))
        for k, v in sorted(res["effects"].items()):
            print("  ", k, v)


@app.local_entrypoint()
def build_map(run: str, model: str, token_run: str, points: str, per_layer: int = 3):
    """Run `neuron_map` for one model; fetch the result with
    `modal volume get sika-interp-cache maps/<run>/<model name> <local dir>`."""
    categories = {r["family"]: r["category"] for r in json.loads(Path(points).read_text())}
    print(neuron_map.remote(run, model, token_run, categories, per_layer))


@app.function(volumes={"/cache": cache}, timeout=600)
def cards_of(run: str, model: str) -> list[dict]:
    """A built map's neuron cards."""
    return json.loads((Path("/cache/maps") / run / model.split("/")[-1] / "cards.json").read_text())


@app.function(volumes={"/cache": cache}, timeout=600)
def save_steering(run: str, model: str, result: dict) -> None:
    """Keep a steering run's summary beside the model's map."""
    path = Path("/cache/maps") / run / model.split("/")[-1] / "steering.json"
    path.write_text(json.dumps(result))
    cache.commit()


@app.local_entrypoint()
def steer_run(traces: str, run: str, model: str, per_language: int = 20, controls: int = 32, containers: int = 6):
    """The steer lens for every candidate neuron on `model`'s map, at its scam decision
    points in `traces`: each neuron's mean shift in the pay-versus-protect margin for a push
    up and down, overall and per language, and its percentile among random pushes at its
    layer. Saved locally and beside the map."""
    cards = cards_of.remote(run, model)
    picked = sorted({(c["layer"], c["neuron"]) for c in cards})
    decisions = decision_points(traces.split(","), per_language, SCAM_FAMILIES).get(model, [])
    step = -(-len(decisions) // containers)
    calls = [
        Model(model_id=model).steer.spawn(decisions[i : i + step], [list(n) for n in picked], controls)
        for i in range(0, len(decisions), step)
    ]
    effects: dict[str, list[float]] = {}
    langs: dict[str, list[str]] = {}
    kept = []
    for call in calls:
        res = call.get()
        kept += res["decisions"]
        for key, values in res["effects"].items():
            effects.setdefault(key, []).extend(values)
        langs.setdefault("order", []).extend(d["language"] for d in res["decisions"])
    order = langs.get("order", [])

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    controls_by_layer: dict[int, list[float]] = {}
    for key, values in effects.items():
        kind, layer, _, _ = key.split(":")
        if kind == "control":
            controls_by_layer.setdefault(int(layer), []).append(abs(mean(values)))
    neurons_out = []
    for layer, i in picked:
        row = {"layer": layer, "neuron": i}
        for sign in ("+1", "-1"):
            values = effects.get(f"neuron:{layer}:{i}:{sign}", [])
            row[f"push{sign[0]}"] = round(mean(values), 3)
            row[f"push{sign[0]}_by_language"] = {
                lang: round(mean([v for v, l in zip(values, order) if l == lang]), 3) for lang in ("en", "tw", "en-tw")
            }
            pool = controls_by_layer.get(layer, [])
            row[f"push{sign[0]}_percentile"] = round(100 * sum(c < abs(mean(values)) for c in pool) / max(len(pool), 1), 1)
        neurons_out.append(row)
    result = {"model": model, "decisions": kept, "neurons": neurons_out, "controls_per_layer": controls}
    save_steering.remote(run, model, result)
    out = Path(f"steering-{run}-{model.split('/')[-1]}.json")
    out.write_text(json.dumps(result))
    strongest = sorted(neurons_out, key=lambda r: -max(abs(r["push+"]), abs(r["push-"])))[:12]
    print(f"{model}: {len(kept)} decisions, {len(neurons_out)} neurons, {controls} controls per layer")
    print("  baseline margin by language:", {lang: round(mean([d["margin"] for d in kept if d["language"] == lang]), 2) for lang in ("en", "tw", "en-tw")})
    for r in strongest:
        print(f"  L{r['layer']:>2} n{r['neuron']:<6} push up {r['push+']:+.2f} ({r['push+_percentile']:.0f}th pct)  push down {r['push-']:+.2f} ({r['push-_percentile']:.0f}th pct)")
    print(f"saved {out}")


@app.local_entrypoint()
def steer_all(
    traces: str,
    run: str,
    model: str,
    seeds: str = "1000000,1000001,1000006",
    doses: str = "0.2,0.4,0.8",
    controls: int = 32,
    containers: int = 4,
    limit: int = 0,
    out: str = "",
    diagnose: bool = False,
    gpu: str = "L40S",
    retry: str = "",
    skip: str = "",
):
    """`steer_moves` for every candidate neuron on `model`'s map (map run `run`), at the
    decisions of every task family in the episodes with these seeds (all three languages):
    every first move, plus the later decisions that have right and wrong moves. `limit`
    keeps that many decisions, spread over families and languages, for timing a slice;
    `diagnose` only checks the batching ways against plain passes; `retry` (a previous output
    JSON) reruns only its decisions that errored and merges them back into it; `skip` (a text
    file of saved array names) leaves out decisions already done, to be rebuilt from their
    arrays with `packing.rows_from_npz`. Saves a JSON of every decision's margins and shifts
    locally."""
    import time

    from sika_interp.decisions import collect

    started = time.time()
    picked = [list(n) for n in sorted({(c["layer"], c["neuron"]) for c in cards_of.remote(run, model)})]
    found = collect(traces.split(","), model, [int(s) for s in seeds.split(",")])
    chosen = [d for d in found if d["kind"] == "first_move" or d["rule"] is not None]
    previous = None
    if retry:
        previous = json.loads(Path(retry).read_text())
        failed = {(r["task_id"], r["kind"]) for r in previous["decisions"] if "error" in r}
        chosen = [d for d in chosen if (d["task_id"], d["kind"]) in failed]
        print(f"retrying {len(chosen)} decisions that errored")
    if skip:
        done = set(Path(skip).read_text().split())
        chosen = [d for d in chosen if f"{d['task_id'].replace('/', '_')}_{d['kind']}.npz" not in done]
        print(f"{len(done)} decisions already saved; running {len(chosen)}")
    if limit:
        step = max(1, len(chosen) // limit)
        chosen = chosen[::step][:limit]
    doses_ = [float(x) for x in doses.split(",")]
    per = -(-len(chosen) // containers)
    calls = [
        Model.with_options(gpu=gpu)(model_id=model).steer_moves.spawn(run, chosen[i : i + per], picked, controls, doses_, diagnose=diagnose)
        for i in range(0, len(chosen), per)
    ]
    rows = [r for call in calls for r in call.get()]
    directions = next(r.pop("directions") for r in rows if "directions" in r)
    for r in rows:
        r.pop("directions", None)
    if previous is not None:
        redone = {(r["task_id"], r["kind"]): r for r in rows}
        rows = [redone.get((r["task_id"], r["kind"]), r) for r in previous["decisions"]]
    result = {"model": model, "run": run, "doses": doses_, "signs": [-1, 1], "controls_per_layer": controls,
              "directions": directions, "decisions": rows}
    path = Path(out or f"steer-all-{run}-{model.split('/')[-1]}.json")
    path.write_text(json.dumps(result))
    seconds = [r["seconds"] for r in rows]
    print(f"{model}: {len(rows)} decisions in {len(calls)} containers, {time.time() - started:.0f}s wall;"
          f" errors {sum('error' in r for r in rows)}, flagged {sum(bool(r.get('flagged')) for r in rows)}")
    for r in rows:
        if "error" in r:
            print(f"  ERROR {r['task_id']} {r['kind']}: {r['error'][:200]}")
    if diagnose:
        for r in rows:
            print(f"  {r['task_id']} {r['kind']}: labels {len(r['labels'])} nodes {r['nodes']}")
            for w, c in r["checks"].items():
                print(f"    {w:24} {c}")
        print(f"saved {path}")
        return
    ok_rows = [r for r in rows if "error" not in r]
    print(f"  way {ok_rows[0]['way']}  checks {ok_rows[0]['checks']}  batch {ok_rows[0]['batch']}")
    print(f"  per decision: mean {sum(seconds) / len(seconds):.1f}s, max {max(seconds):.1f}s; GPU-seconds {sum(seconds):.0f}")
    for r in ok_rows[:6]:
        print(f"  {r['task_id']} {r['kind']}: right {r.get('right_margin')} harm {r.get('harm_margin')}  prompt {r['prompt_tokens']} tail {r['tail_tokens']} labels {len(r['labels'])} nodes {r['nodes']}")
    print(f"saved {path}")


def anchor_episodes(traces: str, model: str, task_ids: str) -> list[dict]:
    """The episodes `task_ids` (comma-separated) of `model` in `traces`, turn by turn."""
    wanted = task_ids.split(",")
    episodes = []
    for path in traces.split(","):
        for trace in load(path):
            data = trace["task"]["data"]
            if data["id"] in wanted and trace["agent"]["config"]["model"] == model:
                ts = turns(trace)
                episodes.append(
                    {"task_id": data["id"], "language": data["language"], "tools": ts[0].tools,
                     "turns": [{"messages": t.messages, "response": t.response} for t in ts]}
                )
    return episodes


@app.local_entrypoint()
def suppress(traces: str, model: str, task_ids: str, out: str = ""):
    """Attention-blocking step links (`suppress_links`) for the episodes `task_ids` of `model`
    in `traces`, one GPU container per episode; saved locally."""
    results = list(Model(model_id=model).suppress_links.map(anchor_episodes(traces, model, task_ids)))
    path = Path(out or f"anchors-suppress-{model.split('/')[-1]}.json")
    path.write_text(json.dumps(results, ensure_ascii=False))
    for r in results:
        strongest = sorted(r["links"], key=lambda x: -x[2])[:4]
        print(f"{r['task_id']}: {len(r['links'])} links, check KL {r['check_kl']}, unplaced {len(r['missing'])}; strongest {strongest}")
    print(f"layers blocked: {results[0]['layers_blocked'] if results else []}\nsaved {path}")


@app.local_entrypoint()
def receivers(
    traces: str, model: str, task_ids: str, seeds: str = "1000000,1000001,1000006", top: int = 20, batch: int = 30,
    out: str = "", gpu: str = "L40S",
):
    """Receiver heads (Thought Anchors) for `model`: `sentence_attention` on the first turn of
    every episode with these seeds (all families and languages) and the last turn of the
    anchor episodes `task_ids`; heads ranked by mean kurtosis of their vertical scores, the
    top `top` taken as receiver heads, and every sentence scored by them. Saved locally."""
    import numpy as np

    wanted = {int(s) for s in seeds.split(",")}
    items = [dict(e, kind="first", index=0) for e in first_turns(traces.split(",")).get(model, []) if e["seed"] in wanted]
    for ep in anchor_episodes(traces, model, task_ids):
        t = ep["turns"][-1]
        items.append({"task_id": ep["task_id"], "language": ep["language"], "kind": "last", "index": len(ep["turns"]) - 1,
                      "messages": t["messages"], "tools": ep["tools"], "response": t["response"]})
    # Full attention maps of long episodes need ~20 GB per layer: use a larger GPU for those.
    calls = [Model.with_options(gpu=gpu)(model_id=model).sentence_attention.spawn(items[i : i + batch]) for i in range(0, len(items), batch)]
    rows = [r for c in calls for r in c.get()]
    for r, item in zip(rows, items):
        r["kind"] = item["kind"]
    layers = rows[0]["layers"]
    kurt = np.nanmean(np.stack([r["kurtosis"] for r in rows]), 0)  # [layers, heads]
    order = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(kurt, nan=-np.inf), axis=None), kurt.shape))[0]
    heads = [(int(a), int(b)) for a, b in order[:top]]
    result = {
        "model": model,
        "layers": layers,
        "kurtosis": np.round(kurt, 3).tolist(),
        "receivers": [{"layer": layers[a], "head": b, "kurtosis": round(float(kurt[a, b]), 3)} for a, b in heads],
        "items": [],
    }
    for r in rows:
        v = r["vertical"].astype(np.float32)
        by_head = np.stack([v[a, b] for a, b in heads])  # [top, sentences]
        score = np.nanmean(by_head, 0)
        result["items"].append(
            {k: r[k] for k in ("task_id", "language", "kind", "turn", "prompt_tokens")}
            | {"sentences": [dict(s, score=None if np.isnan(x) else round(float(x), 5)) for s, x in zip(r["sentences"], score)],
               "by_head": np.round(np.nan_to_num(by_head, nan=-1), 5).tolist()}
        )
    path = Path(out or f"receivers-{model.split('/')[-1]}.json")
    path.write_text(json.dumps(result, ensure_ascii=False))
    print(f"{model}: {len(rows)} turns; top receiver heads (layer, head, kurtosis): "
          + ", ".join(f"L{h['layer']}H{h['head']} {h['kurtosis']:.1f}" for h in result["receivers"][:8]))
    print(f"saved {path}")


@app.local_entrypoint()
def contexts(traces: str, run: str, model: str, batch: int = 300, out: str = ""):
    """Top-firing request contexts (`neuron_contexts`) for every neuron card on `model`'s map
    (map run `run`), over the first turn of every episode in `traces`; merged and saved."""
    picked = [list(n) for n in sorted({(c["layer"], c["neuron"]) for c in cards_of.remote(run, model)})]
    eps = first_turns(traces.split(",")).get(model, [])
    calls = [(i, Model(model_id=model).neuron_contexts.spawn(eps[i : i + batch], picked)) for i in range(0, len(eps), batch)]
    merged: dict = {"neurons": {}, "episodes": {}}
    for offset, call in calls:
        res = call.get()
        for e, ep in res["episodes"].items():
            merged["episodes"][str(int(e) + offset)] = ep
        for key, langs in res["neurons"].items():
            for lang, rows in langs.items():
                for r in rows:
                    r["episode"] += offset
                merged["neurons"].setdefault(key, {}).setdefault(lang, []).extend(rows)
    for langs in merged["neurons"].values():
        for lang in langs:
            langs[lang] = sorted(langs[lang], key=lambda r: -r["peak"])[:8]
    used = {str(r["episode"]) for langs in merged["neurons"].values() for rows in langs.values() for r in rows}
    merged["episodes"] = {e: v for e, v in merged["episodes"].items() if e in used}
    path = Path(out or f"contexts-{run}-{model.split('/')[-1]}.json")
    path.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"{model}: {len(merged['neurons'])} neurons, {len(merged['episodes'])} requests kept; saved {path}")


@app.local_entrypoint()
def profile_steer(traces: str, model: str, way: str = "tiled", neurons: str = "8:1,16:2,24:3", gpu: str = "L40S"):
    """Time the parts of one batch of pushes at the first decision found in `traces`."""
    from sika_interp.decisions import collect

    d = collect(traces.split(","), model, [1000000], ["send_money"])[0]
    picked = [[int(a) for a in n.split(":")] for n in neurons.split(",")]
    print(json.dumps(Model.with_options(gpu=gpu)(model_id=model).profile_moves.remote(d, picked, way), indent=1))


@app.local_entrypoint()
def anchors(traces: str, model: str, task_ids: str):
    """Step links (`step_links`) for the episodes `task_ids` (comma-separated) of `model` in
    `traces`, one GPU container per episode; saved locally as anchors-links-<model>.json."""
    episodes = anchor_episodes(traces, model, task_ids)
    results = list(Model(model_id=model).step_links.map(episodes))
    out = Path(f"anchors-links-{model.split('/')[-1]}.json")
    out.write_text(json.dumps(results, ensure_ascii=False))
    for r in results:
        strongest = sorted(r["links"], key=lambda x: -x[2])[:4]
        print(f"{r['task_id']}: {len(r['steps'])} steps, strongest links (from step, to turn, KL): {strongest}")
    print(f"saved {out}")


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
