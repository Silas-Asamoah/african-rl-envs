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
