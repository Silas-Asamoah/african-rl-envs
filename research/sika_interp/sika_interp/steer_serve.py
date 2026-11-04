"""Steer or patch a served model: a vLLM worker extension (`--worker-extension-cls
sika_interp.steer_serve.SteerExtension`) reached through vLLM's development endpoint
(`VLLM_SERVER_DEV_MODE=1`), which passes string arguments.

Steering adds one direction to the residual stream after one decoder layer, at every token of
every request, scaled by that token's residual norm, as the steer lens does at decision points:

    POST /collective_rpc {"method": "steer", "args": ["{\"layer\": 19, \"neuron\": 4818, \"dose\": 0.4, \"sign\": 1}"]}
    POST /collective_rpc {"method": "steer", "args": ["{\"layer\": 19, \"seed\": 3, \"dose\": 0.4, \"sign\": 1}"]}
    POST /collective_rpc {"method": "unsteer"}

A neuron's direction is its column of the layer's MLP down projection; a seed gives a random
unit direction instead, as a control.

Patching sets the residual stream after chosen layers, at the template tokens that follow a
request, to given vectors (the patching lens: `modal_app.Patches`). Each spec names a request
by a signature, the token ids of the end of the request and the template tokens, at their
positions; wherever a batch holds that signature whole, the template tokens are set. Prefix
caching must be off so every turn's prefill re-applies the patch, and prefill chunks must be
long enough to hold a whole prompt. Specs come from a folder of per-decision files on disk
(`patch_specs` below):

    POST /collective_rpc {"method": "patch", "args": ["{\"folder\": \"/cache/patch/specs/study/Qwen3.5-9B/primary\", \"source\": \"en\", \"layers\": [19]}"]}
    POST /collective_rpc {"method": "patch_state"}   -> how many specs are loaded and how many template tokens were set so far
    POST /collective_rpc {"method": "unpatch"}

The server must run uncompiled and without CUDA graphs (`--enforce-eager -O0`), so the hooks
run on every forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path


def find_patches(input_ids, positions, specs: list[dict]) -> dict[int, tuple]:
    """Where each spec's signature appears whole in a flattened batch (`input_ids` and each
    token's position in its own sequence): per layer, the flat indices of the template tokens
    (the last `k` of the signature) and the vectors to set there, as (indices, [n, width])."""
    import torch

    found: dict[int, tuple[list, list]] = {}
    n_tokens = int(input_ids.shape[0])
    for spec in specs:
        sig, first, k = spec["signature"], int(spec["first"]), int(spec["k"])
        n = int(sig.shape[0])
        sig = sig.to(input_ids.device, input_ids.dtype)
        for c in (positions == first).nonzero().flatten().tolist():
            if c + n > n_tokens:
                continue
            if not torch.equal(input_ids[c : c + n], sig):
                continue
            if not torch.equal(positions[c : c + n], torch.arange(first, first + n, device=positions.device, dtype=positions.dtype)):
                continue
            idx = torch.arange(c + n - k, c + n, device=input_ids.device)
            for layer, vec in zip(spec["layers"], spec["vectors"]):
                idxs, vecs = found.setdefault(int(layer), ([], []))
                idxs.append(idx)
                vecs.append(vec)
    return {layer: (torch.cat(idxs), torch.cat(vecs)) for layer, (idxs, vecs) in found.items()}


def patch_specs(folder: str | Path, source: str, layers: list[int], device) -> list[dict]:
    """The specs in `folder` (one `.pt` per decision, as `modal_app.patch_specs` writes them)
    for one source and these layers: each with its signature, first position, k, the layers
    and the vectors ([layers, k, width]) on `device`."""
    import torch

    specs = []
    for path in sorted(Path(folder).glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        states = item["states"][source]  # [saved layers, k, width]
        rows = [item["layers"].index(layer) for layer in layers]
        specs.append(
            {
                "task_id": item["task_id"],
                "signature": item["signature"],
                "first": int(item["first"]),
                "k": int(item["k"]),
                "layers": [int(layer) for layer in layers],
                "vectors": states[rows].to(device=device, dtype=torch.float32),
            }
        )
    return specs


class SteerExtension:
    """Mixed into vLLM's GPU worker; `self.model_runner` is the worker's."""

    def _steer_layers(self):
        """The decoder layers: the longest module list whose entries have an MLP down projection
        (a multimodal model's vision tower has its own, shorter list)."""
        import torch

        model = self.model_runner.get_model()
        best = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) and hasattr(getattr(module[0], "mlp", None), "down_proj"):
                if best is None or len(module) > len(best):
                    best = module
        if best is None:
            raise RuntimeError("no decoder layers with mlp.down_proj found")
        return best

    def _steer_install(self) -> None:
        if getattr(self, "_steer_hooks", None):
            return
        self._steer_state = None
        self._patch_state = None
        self._patch_current: dict = {}
        self._patch_hits = 0
        self._patch_probe: dict = {"forwards": 0, "candidate_passes": 0, "matched_passes": 0, "last": None}

        def make(i: int):
            def hook(module, args, output):
                s = self._steer_state
                p = self._patch_current.get(i)
                if s is None or s["layer"] != i:
                    if p is None:
                        return None
                # vLLM's layers return (hidden, residual): the stream is their sum, and the
                # next layer's fused add folds an extra term on `hidden` into it.
                hidden = output[0] if isinstance(output, tuple) else output
                residual = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if s is not None and s["layer"] == i:
                    stream = hidden if residual is None else hidden + residual
                    push = (s["scale"] * stream.float().norm(dim=-1, keepdim=True) * s["vec"]).to(hidden.dtype)
                    hidden = hidden + push
                if p is not None:
                    idx, vec = p
                    hidden = hidden.clone()
                    # Setting the stream to `vec`: with a separate residual, `hidden` carries the rest.
                    hidden[idx] = (vec if residual is None else vec - residual[idx].float()).to(hidden.dtype)
                    self._patch_hits += int(idx.shape[0])
                if isinstance(output, tuple):
                    return (hidden,) + tuple(output[1:])
                return hidden

            return hook

        def before(module, args, kwargs):
            """Find this pass's patch sites from the batch's token ids and positions."""
            st = self._patch_state
            self._patch_current = {}
            if st is None:
                return None
            import torch

            input_ids, positions = batch_tokens(self.model_runner, args, kwargs)
            probe = self._patch_probe
            probe["forwards"] += 1
            probe["last"] = {
                "input_ids": None if input_ids is None else [list(input_ids.shape), str(input_ids.dtype)],
                "positions": None if positions is None else list(positions.shape),
            }
            if input_ids is None or positions is None:
                return None
            if not bool(torch.isin(positions, st["firsts"].to(positions.device)).any()):
                return None
            probe["candidate_passes"] += 1
            found = find_patches(input_ids, positions, st["specs"])
            if found:
                probe["matched_passes"] += 1
            elif "mismatch" not in probe:
                # The first pass where a spec's first position occurs but nothing matched:
                # keep what the batch holds there, against the spec, for `patch_probe`.
                spec = st["specs"][0]
                at = (positions == spec["first"]).nonzero().flatten()
                if len(at):
                    c = int(at[0])
                    n = int(spec["signature"].shape[0])
                    probe["mismatch"] = {
                        "task_id": spec["task_id"],
                        "batch_tokens": input_ids[c : c + n].tolist(),
                        "batch_positions": positions[c : c + n].tolist(),
                        "signature": spec["signature"].tolist(),
                    }
            self._patch_current = found
            return None

        model = self.model_runner.get_model()
        self._steer_hooks = [layer.register_forward_hook(make(i)) for i, layer in enumerate(self._steer_layers())]
        self._steer_hooks.append(model.register_forward_pre_hook(before, with_kwargs=True))

    def steer(self, spec: str) -> dict:
        import torch

        cfg = json.loads(spec)
        self._steer_install()
        layers = self._steer_layers()
        w = layers[cfg["layer"]].mlp.down_proj.weight  # [hidden, intermediate]
        if cfg.get("neuron") is not None:
            v = w[:, int(cfg["neuron"])].detach().float()
        else:
            g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
            v = torch.randn(w.shape[0], generator=g).to(w.device)
        v = v / v.norm()
        self._steer_state = {"layer": int(cfg["layer"]), "vec": v, "scale": float(cfg["sign"]) * float(cfg["dose"])}
        return {"layer": int(cfg["layer"]), "layers": len(layers), "hidden": int(v.numel()), "scale": self._steer_state["scale"]}

    def unsteer(self) -> dict:
        self._steer_state = None
        return {"steer": None}

    def steer_state(self) -> dict:
        s = getattr(self, "_steer_state", None)
        return {"steer": None} if s is None else {"layer": s["layer"], "scale": s["scale"]}

    def patch(self, spec: str) -> dict:
        """Load the specs of one source at some layers from a folder; see `patch_specs`."""
        import torch

        cfg = json.loads(spec)
        self._steer_install()
        device = next(self.model_runner.get_model().parameters()).device
        specs = patch_specs(cfg["folder"], cfg["source"], [int(x) for x in cfg["layers"]], device)
        self._patch_state = {
            "specs": specs,
            "firsts": torch.tensor(sorted({s["first"] for s in specs}), device=device),
            "source": cfg["source"],
            "layers": [int(x) for x in cfg["layers"]],
        }
        self._patch_hits = 0
        self._patch_probe = {"forwards": 0, "candidate_passes": 0, "matched_passes": 0, "last": None}
        return {"specs": len(specs), "source": cfg["source"], "layers": self._patch_state["layers"],
                "k": sorted({s["k"] for s in specs})}

    def unpatch(self) -> dict:
        self._patch_state = None
        self._patch_current = {}
        return {"patch": None}

    def patch_state(self) -> dict:
        st = getattr(self, "_patch_state", None)
        if st is None:
            return {"patch": None}
        return {"specs": len(st["specs"]), "source": st["source"], "layers": st["layers"], "hits": self._patch_hits}

    def patch_probe(self) -> dict:
        """What the passes since the last `patch` looked like: how many ran, how many held a
        spec's first position, how many matched, the last pass's token and position shapes,
        and, for the first pass that held a first position but matched nothing, the batch's
        tokens there against the spec's signature."""
        return dict(getattr(self, "_patch_probe", None) or {"forwards": 0})


def batch_tokens(runner, args, kwargs):
    """The flattened batch's token ids and each token's position in its sequence, from the
    model forward's arguments, or from the runner's buffers when the model is handed
    embeddings instead of ids (vLLM does that for multimodal-capable models). Positions laid
    out for M-RoPE ([3, tokens]) reduce to their first row, which is the text position."""
    input_ids = kwargs.get("input_ids", args[0] if args else None)
    positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
    if positions is None:
        return None, None
    if positions.dim() == 2:
        positions = positions[0]
    n = int(positions.shape[0])
    if input_ids is None:
        buffer = getattr(runner, "input_ids", None)
        ids = getattr(buffer, "gpu", buffer)
        if ids is not None and hasattr(ids, "shape") and ids.shape[0] >= n:
            input_ids = ids[:n]
    return input_ids, positions
