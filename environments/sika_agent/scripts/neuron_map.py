"""Map the MLP neurons that tell English from Twi and scams from ordinary requests.

In the spirit of neuron-explorer (github.com/lyramakesmusic/neuron-explorer): every MLP
neuron of every layer is read on matched sika-agent tasks, a few hundred are picked out, and
each picked neuron gets a card: what it reads, what it writes, and what pushing it does.

For every neuron (the input to down_proj, so one value per neuron per token):
- **language preference**: how often it is active (> 0) on the user request's tokens in
  English, Twi and Twi–English (LAPE-style activation probability), and a specificity in
  [0, 1] from the entropy over the three;
- **scam selectivity**: its activation at the last prompt token (where the first action is
  decided) on scams and planted instructions against ordinary tasks, as a z-score.

Picked per layer: the most Twi-specific, most English-specific and most scam-selective
neurons, plus a random background sample for the map. For each picked neuron:
- **logit lens**: tokens its output direction promotes and suppresses (through the final norm
  and the unembedding);
- **steer lens**: add ±`--dose` × its typical maximum to its activation at every position of
  a few decision prompts, and record the change in the log-probability of the first action
  being a transfer, a question to the user, a fraud report or reading messages, and the
  tokens most pushed at the next-token position;
- **wiring**: direct-path links to other picked neurons, from its down_proj column read by
  their gate_proj rows (positive excites, negative inhibits).

    uv run --with torch --with transformers python scripts/neuron_map.py \\
        --model Qwen/Qwen3.5-4B --seeds 6 --out maps/q35-4b/neurons.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_activations import build_tasks, user_token_span  # noqa: E402
from interp_common import mlp_modules, steer_neuron, tool_schemas  # noqa: E402
from sika_agent.scenarios import FAMILIES  # noqa: E402

LANGS = ["en", "tw", "en-tw"]
ACTIONS = {"move_money": "initiate_transfer", "ask_user": "ask_user", "report_fraud": "report_fraud", "read_messages": "read_messages"}


def final_norm(model):
    """The decoder's final norm, if it is a plain scale (RMSNorm-style weight)."""
    best = None
    for name, mod in model.named_modules():
        if name.endswith("norm") and ".layers." not in name and hasattr(mod, "weight") and mod.weight is not None:
            best = mod
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--seeds", type=int, default=6, help="Matched seeds per family (each in every language)")
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--per-layer", type=int, default=4, help="Neurons picked per layer for each of Twi, English, scam")
    ap.add_argument("--background", type=int, default=320, help="Random neurons per layer drawn as the map's cloud")
    ap.add_argument("--steer-top", type=int, default=120, help="Picked neurons that get a steer lens (most selective)")
    ap.add_argument("--dose", type=float, default=4.0, help="Steering amount, in units of the neuron's typical maximum")
    ap.add_argument("--steer-families", nargs="+", default=["promo_fee", "fake_refund", "send_money"])
    ap.add_argument("--call-prefix", default='<tool_call>\n{"name": "', help="Text before a tool name in a call")
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device).eval()
    mods = mlp_modules(model)
    layers = list(mods)
    d_mlp = mods[layers[0]].down_proj.in_features
    tools = tool_schemas()
    kwargs = json.loads(args.template_kwargs)
    tasks = list(build_tasks(args.families, LANGS, args.seeds, args.split))[: args.limit]

    def render(task):
        msgs = [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["user"]}]
        return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, **kwargs)

    # -- pass 1: statistics over every neuron ------------------------------------------
    z = lambda: torch.zeros(len(layers), d_mlp, device=device, dtype=torch.float64)  # noqa: E731
    active = {la: z() for la in LANGS}
    ntok = {la: 0 for la in LANGS}
    peak_sum, ntask = z(), 0
    last = {k: (z(), z(), [0]) for k in ("unsafe", "safe")}  # sum, sum of squares, count
    ctx: dict = {}

    def hook(_m, a, li):
        x = a[0][0].float()  # (seq, d_mlp)
        u = x[ctx["u0"] : ctx["u1"]]
        active[ctx["lang"]][li] += (u > 0).double().sum(0)
        peak_sum[li] += u.abs().amax(0).double()
        s, sq, _ = last[ctx["kind"]]
        s[li] += x[-1].double()
        sq[li] += x[-1].double() ** 2

    handles = [mods[la].down_proj.register_forward_pre_hook(lambda m, a, li=i: hook(m, a, li)) for i, la in enumerate(layers)]
    t0 = time.time()
    with torch.no_grad():
        for n, task in enumerate(tasks, 1):
            text = render(task)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
            u0, u1 = user_token_span(enc.pop("offset_mapping")[0].tolist(), text, task["user"])
            ctx.update(u0=u0, u1=u1, lang=task["language"], kind="unsafe" if task["unsafe"] else "safe")
            model(enc["input_ids"].to(device))
            ntok[task["language"]] += u1 - u0
            last[ctx["kind"]][2][0] += 1
            ntask += 1
            if n % 20 == 0 or n == len(tasks):
                print(f"stats {n}/{len(tasks)} ({n / (time.time() - t0):.1f} tasks/s)", flush=True)
    for h in handles:
        h.remove()

    p = torch.stack([active[la] / max(1, ntok[la]) for la in LANGS])  # (3, L, d)
    q = p / p.sum(0).clamp_min(1e-9)
    entropy = -(q * q.clamp_min(1e-12).log()).sum(0)
    spec = (1 - entropy / math.log(3)) * (p.max(0).values > 0.02)
    ratio = ((p[1] + 1e-3) / (p[0] + 1e-3)).log()  # Twi vs English
    scale = (peak_sum / max(1, ntask)).clamp_min(1e-4)
    stats = {}
    for k, (s, sq, c) in last.items():
        mean = s / max(1, c[0])
        stats[k] = (mean, (sq / max(1, c[0]) - mean**2).clamp_min(0))
    scam_z = (stats["unsafe"][0] - stats["safe"][0]) / (0.5 * (stats["unsafe"][1] + stats["safe"][1])).sqrt().clamp_min(1e-6)

    # -- pick neurons ------------------------------------------------------------------------
    picked: dict[tuple[int, int], set[str]] = {}
    for li in range(len(layers)):
        twi_score = ratio[li] * (p[1, li] > 0.05)
        en_score = -ratio[li] * (p[0, li] > 0.05)
        for kind, score in (("tw", twi_score), ("en", en_score), ("scam", scam_z[li].abs())):
            for idx in score.topk(args.per_layer).indices.tolist():
                picked.setdefault((li, idx), set()).add(kind)

    def card(li, idx, kinds):
        probs = {la: round(float(p[k, li, idx]), 4) for k, la in enumerate(LANGS)}
        return {
            "id": f"L{layers[li]}.N{idx}", "layer": layers[li], "idx": idx, "kinds": sorted(kinds),
            "p": probs, "spec": round(float(spec[li, idx]), 3), "pref": LANGS[int(p[:, li, idx].argmax())],
            "twi_ratio": round(float(ratio[li, idx]), 3), "scam_z": round(float(scam_z[li, idx]), 3),
            "scale": round(float(scale[li, idx]), 4),
        }  # fmt: skip

    neurons = [card(li, idx, kinds) for (li, idx), kinds in sorted(picked.items())]
    print(f"picked {len(neurons)} neurons", flush=True)

    # -- logit lens --------------------------------------------------------------------------
    W_U = model.get_output_embeddings().weight
    norm = final_norm(model)
    with torch.no_grad():
        for nrn in neurons:
            v = mods[nrn["layer"]].down_proj.weight[:, nrn["idx"]].float()
            if norm is not None and norm.weight.shape == v.shape:
                v = v * norm.weight.float()
            logits = W_U.float() @ v
            nrn["logit_up"] = [tok.decode([t]) for t in logits.topk(8).indices.tolist()]
            nrn["logit_down"] = [tok.decode([t]) for t in (-logits).topk(8).indices.tolist()]

    # -- wiring among picked neurons ---------------------------------------------------------
    gate_ok = all(hasattr(mods[n["layer"]], "gate_proj") for n in neurons)
    with torch.no_grad():
        D = torch.stack([mods[n["layer"]].down_proj.weight[:, n["idx"]].float() for n in neurons], 1)  # (h, F)
        R = torch.stack(
            [(mods[n["layer"]].gate_proj if gate_ok else mods[n["layer"]].up_proj).weight[n["idx"]].float() for n in neurons]
        )  # (F, h)
        D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-9)
        R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-9)
        M = (R @ D).cpu()  # M[b, a]: how much b reads what a writes
    lay = torch.tensor([n["layer"] for n in neurons])
    later = lay[:, None] > lay[None, :]  # b after a
    M = M * later
    for a, nrn in enumerate(neurons):
        col = M[:, a]
        top = col.abs().topk(min(8, len(neurons))).indices.tolist()
        nrn["feeds"] = [{"to": neurons[b]["id"], "w": round(float(col[b]), 4)} for b in top if col[b] != 0][:6]
        row = M[a]
        top = row.abs().topk(min(8, len(neurons))).indices.tolist()
        nrn["fed_by"] = [{"from": neurons[b]["id"], "w": round(float(row[b]), 4)} for b in top if row[b] != 0][:6]

    # -- steer lens on the first action ------------------------------------------------------
    prompts = []
    for fam in args.steer_families:
        if fam not in FAMILIES:
            continue
        for task in build_tasks([fam], ["en", "tw"], 1, args.split):
            prompts.append(task)
    order = sorted(neurons, key=lambda n: -(abs(n["scam_z"]) + 3 * n["spec"]))[: args.steer_top]

    def first_action(text, steer=None):
        """Log-prob of each representative tool opening the first call, and next-token log-probs."""
        conts = [args.call_prefix + name + '"' for name in ACTIONS.values()]
        p_ids = tok(text, add_special_tokens=False)["input_ids"]
        seqs = [p_ids + tok(c, add_special_tokens=False)["input_ids"] for c in conts]
        width = max(map(len, seqs))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
        with torch.no_grad():
            if steer is None:
                logits = model(ids, attention_mask=mask).logits.float().log_softmax(-1)
            else:
                with steer_neuron(model, *steer):
                    logits = model(ids, attention_mask=mask).logits.float().log_softmax(-1)
        scores = {}
        for k, (action, s) in enumerate(zip(ACTIONS, seqs)):
            span = range(len(p_ids), len(s))
            scores[action] = float(sum(logits[k, t - 1, s[t]] for t in span))
        return scores, logits[0, len(p_ids) - 1]

    if order and prompts:
        texts = [render(t) for t in prompts]
        base = [first_action(t) for t in texts]
        t0 = time.time()
        for k, nrn in enumerate(order, 1):
            effects, pushed = {}, {}
            for sign in (+1, -1):
                amount = sign * args.dose * nrn["scale"]
                delta = {a: 0.0 for a in ACTIONS}
                tok_delta = None
                for text, (b_scores, b_next) in zip(texts, base):
                    s_scores, s_next = first_action(text, (nrn["layer"], nrn["idx"], amount))
                    for a in ACTIONS:
                        delta[a] += (s_scores[a] - b_scores[a]) / len(texts)
                    d = s_next - b_next
                    tok_delta = d if tok_delta is None else tok_delta + d
                key = "up" if sign > 0 else "down"
                effects[key] = {a: round(v, 3) for a, v in delta.items()}
                pushed[key] = [tok.decode([t]) for t in tok_delta.topk(8).indices.tolist()]
            nrn["steer"] = {"dose": args.dose, "first_action": effects, "tokens": pushed}
            if k % 10 == 0 or k == len(order):
                print(f"steer {k}/{len(order)} ({(time.time() - t0) / k:.1f}s each)", flush=True)

    # -- background cloud --------------------------------------------------------------------
    rng = random.Random(0)
    background = []
    for li in range(len(layers)):
        for idx in rng.sample(range(d_mlp), min(args.background, d_mlp)):
            background.append([layers[li], idx, LANGS.index(LANGS[int(p[:, li, idx].argmax())]), round(float(spec[li, idx]), 3),
                               round(float(scam_z[li, idx]), 2)])  # fmt: skip

    out = {
        "model": args.model, "layers": layers, "d_mlp": d_mlp, "tasks": ntask,
        "tokens": ntok, "languages": LANGS, "dose": args.dose,
        "neurons": neurons, "background": background,
        "background_fields": ["layer", "idx", "pref (0 en, 1 tw, 2 en-tw)", "spec", "scam_z"],
    }  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {args.out}: {len(neurons)} neurons, {len(background)} background points")


if __name__ == "__main__":
    main()
