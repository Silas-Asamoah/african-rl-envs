"""Steer a served model: a vLLM worker extension (`--worker-extension-cls
sika_interp.steer_serve.SteerExtension`) that adds one direction to the residual stream after
one decoder layer, at every token of every request, scaled by that token's residual norm, as
the steer lens does at decision points. It is set and cleared through vLLM's development
endpoint (`VLLM_SERVER_DEV_MODE=1`), which passes string arguments:

    POST /collective_rpc {"method": "steer", "args": ["{\"layer\": 19, \"neuron\": 4818, \"dose\": 0.4, \"sign\": 1}"]}
    POST /collective_rpc {"method": "steer", "args": ["{\"layer\": 19, \"seed\": 3, \"dose\": 0.4, \"sign\": 1}"]}
    POST /collective_rpc {"method": "unsteer"}

A neuron's direction is its column of the layer's MLP down projection; a seed gives a random
unit direction instead, as a control. The server must run uncompiled and without CUDA graphs
(`--enforce-eager -O0`), so the hook runs on every forward pass, and without prefix caching,
so no request reuses keys and values computed under another setting.
"""

from __future__ import annotations

import json


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

        def make(i: int):
            def hook(module, args, output):
                s = self._steer_state
                if s is None or s["layer"] != i:
                    return None
                # vLLM's layers return (hidden, residual): the stream is their sum, and the
                # next layer's fused add folds an extra term on `hidden` into it.
                hidden = output[0] if isinstance(output, tuple) else output
                residual = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                stream = hidden if residual is None else hidden + residual
                push = (s["scale"] * stream.float().norm(dim=-1, keepdim=True) * s["vec"]).to(hidden.dtype)
                if isinstance(output, tuple):
                    return (hidden + push,) + tuple(output[1:])
                return hidden + push

            return hook

        self._steer_hooks = [layer.register_forward_hook(make(i)) for i, layer in enumerate(self._steer_layers())]

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
