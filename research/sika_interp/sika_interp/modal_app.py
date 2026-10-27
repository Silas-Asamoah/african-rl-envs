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
        # Steps: every message after the system prompt, a model turn grouped with its tool results.
        steps, k = [], 1
        while k < len(last):
            group = [k]
            if last[k]["role"] == "assistant":
                while k + 1 < len(last) and last[k + 1]["role"] == "tool":
                    k += 1
                    group.append(k)
            steps.append(group)
            k += 1
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
def anchors(traces: str, model: str, task_ids: str):
    """Step links (`step_links`) for the episodes `task_ids` (comma-separated) of `model` in
    `traces`, one GPU container per episode; saved locally as anchors-links-<model>.json."""
    wanted = task_ids.split(",")
    episodes = []
    for path in traces.split(","):
        for trace in load(path):
            data = trace["task"]["data"]
            if data["id"] in wanted and trace["agent"]["config"]["model"] == model:
                ts = turns(trace)
                episodes.append(
                    {"task_id": data["id"], "tools": ts[0].tools, "turns": [{"messages": t.messages, "response": t.response} for t in ts]}
                )
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
