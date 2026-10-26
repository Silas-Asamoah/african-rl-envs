"""Dump a model's residual-stream activations on matched sika-agent tasks.

Builds tasks in matched mode, so one seed gives the same wallet, people and amounts in
English, Twi and Twi–English and only the wording changes. Each task is rendered exactly as
the agent first sees it (system prompt, tool definitions, user request), run through the
model once, and the hidden state of every layer is saved at two positions:

- `last`: the final prompt token, where the model decides its first action;
- `user_mean`: the mean over the user request's tokens, what the model made of the request.

Optionally it also scores the first action: the log-probability of each tool name as the
start of the model's first tool call (`--tool-logprobs`).

Needs torch and transformers, which the environment itself doesn't depend on:

    uv run --with torch --with transformers python scripts/dump_activations.py \\
        --model Qwen/Qwen3.5-4B --seeds 10 --out acts/q35-4b

Outputs, in `--out`:
- `last.npy`, `user_mean.npy`: float16, shape (tasks, layers + 1, hidden); index 0 is the
  embedding output, index i the output of layer i;
- `meta.jsonl`: one line per task, in the same order: id, family, category, language, seed,
  token counts, fertility (user tokens per word) and, with `--tool-logprobs`, tool scores;
- `config.json`: the arguments, model and layer count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sika_agent import templates  # noqa: E402
from sika_agent.scenarios import FAMILIES, generate  # noqa: E402
from sika_agent.taskset import SYSTEM_PROMPT  # noqa: E402
from interp_common import tool_schemas  # noqa: E402

# Families whose request should not be carried out as asked (scams and planted instructions).
UNSAFE_CATEGORIES = {"scams", "injection"}


def build_tasks(families, languages, seeds, split):
    base = 0 if split == "train" else 1_000_000
    for i in range(seeds):
        for family in families:
            for language in languages:
                seed = base + i
                sc = generate(family, language, seed, split, matched=True)
                s = sc.state
                yield {
                    "id": f"{split}/{family}/{language}/{seed}",
                    "family": family,
                    "category": FAMILIES[family][0],
                    "unsafe": FAMILIES[family][0] in UNSAFE_CATEGORIES,
                    "language": language,
                    "seed": seed,
                    "system": SYSTEM_PROMPT.format(owner=s.owner, number=s.owner_number, today=s.today),
                    "user": sc.prompt,
                }


def user_token_span(offsets, text: str, user: str) -> tuple[int, int]:
    """Token indices [start, end) covering the user request inside the rendered prompt."""
    start_char = text.rfind(user)
    if start_char < 0:
        raise RuntimeError("user request not found verbatim in the rendered prompt")
    end_char = start_char + len(user)
    idx = [i for i, (a, b) in enumerate(offsets) if b > start_char and a < end_char and b > a]
    return idx[0], idx[-1] + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id or local path")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--languages", nargs="+", default=list(templates.LANGUAGES))
    ap.add_argument("--seeds", type=int, default=10, help="Matched seeds per family (each in every language)")
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--no-tools", action="store_true", help="Leave tool definitions out of the prompt")
    ap.add_argument(
        "--template-kwargs",
        default='{"enable_thinking": false}',
        help="JSON passed to apply_chat_template (templates ignore keys they don't use)",
    )
    ap.add_argument(
        "--tool-logprobs",
        action="store_true",
        help="Score each tool name as the start of the first tool call (see --call-prefix)",
    )
    ap.add_argument(
        "--call-prefix",
        default='<tool_call>\n{"name": "',
        help="Text the model writes before a tool name; the default is Qwen's format",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N tasks (for a quick check)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    unknown = set(args.families) - set(FAMILIES)
    if unknown:
        sys.exit(f"unknown families: {sorted(unknown)}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        sys.exit("a fast tokenizer is needed to locate the user request's tokens")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.to(device).eval()

    tools = None if args.no_tools else tool_schemas()
    tool_names = [t["function"]["name"] for t in tool_schemas()]
    kwargs = json.loads(args.template_kwargs)
    tasks = list(build_tasks(args.families, args.languages, args.seeds, args.split))[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    last_all, mean_all = [], []
    t0 = time.time()
    with (args.out / "meta.jsonl").open("w") as meta, torch.no_grad():
        for n, task in enumerate(tasks, 1):
            messages = [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["user"]}]
            text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, **kwargs)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            u0, u1 = user_token_span(offsets, text, task["user"])
            out = model(**enc.to(device), output_hidden_states=True, use_cache=args.tool_logprobs)
            hs = torch.stack(out.hidden_states)[:, 0]  # (layers + 1, seq, hidden)
            last_all.append(hs[:, -1].float().cpu().numpy().astype(np.float16))
            mean_all.append(hs[:, u0:u1].float().mean(1).cpu().numpy().astype(np.float16))

            words = max(1, len(task["user"].split()))
            row = {k: task[k] for k in ("id", "family", "category", "unsafe", "language", "seed")}
            row.update(prompt_tokens=int(enc["input_ids"].shape[1]), user_tokens=u1 - u0, fertility=(u1 - u0) / words)
            if args.tool_logprobs:
                row["tool_logprobs"] = score_tools(model, tok, out, args.call_prefix, tool_names, device)
            meta.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 20 == 0 or n == len(tasks):
                rate = n / (time.time() - t0)
                print(f"{n}/{len(tasks)} tasks, {rate:.2f}/s", flush=True)

    np.save(args.out / "last.npy", np.stack(last_all))
    np.save(args.out / "user_mean.npy", np.stack(mean_all))
    config = vars(args) | {"out": str(args.out), "layers": len(last_all[0]) - 1, "tasks": len(tasks), "device": device}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {len(tasks)} tasks × {config['layers'] + 1} layers to {args.out}")


def score_tools(model, tok, out, prefix, names, device) -> dict[str, float]:
    """Log-probability of the text `prefix + name + '"'` continuing the prompt, per tool name.

    Reuses the prompt's cache, so each name costs a few tokens. Each continuation is
    tokenized whole, as the model would write it. These are relative scores for comparing
    the first action across languages, not a decode."""
    import copy

    import torch

    first = out.logits[0, -1].float().log_softmax(-1)
    scores = {}
    for name in names:
        ids = tok(prefix + name + '"', add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        cache = copy.deepcopy(out.past_key_values)
        logits = model(ids, past_key_values=cache, use_cache=True).logits[0].float().log_softmax(-1)
        steps = torch.cat([first[None], logits[:-1]])  # distribution before each continuation token
        scores[name] = steps.gather(1, ids[0][:, None]).sum().item()
    return scores


if __name__ == "__main__":
    main()
