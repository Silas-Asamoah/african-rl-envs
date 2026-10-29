import json

import pytest

torch = pytest.importorskip("torch")

from sika_interp.steer_serve import SteerExtension  # noqa: E402


class Layer(torch.nn.Module):
    """vLLM-style decoder layer: returns (hidden, residual), the stream being their sum."""

    def __init__(self, hidden=8, inner=16):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(inner, hidden, bias=False)

    def forward(self, hidden, residual):
        return hidden * 0.5, residual + hidden * 0.5


class Model(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = torch.nn.ModuleList(Layer() for _ in range(n))
        self.vision = torch.nn.ModuleList(Layer() for _ in range(2))  # a shorter list to ignore

    def forward(self, x):
        hidden, residual = x, torch.zeros_like(x)
        outs = []
        for layer in self.layers:
            hidden, residual = layer(hidden, residual)
            outs.append(hidden + residual)
        return outs


class Runner:
    def __init__(self, model):
        self.model = model

    def get_model(self):
        return self.model


def worker():
    w = SteerExtension()
    w.model_runner = Runner(Model())
    return w


def test_push_lands_after_the_chosen_layer_scaled_by_the_stream_norm():
    w = worker()
    x = torch.randn(3, 8)
    before = w.model_runner.model(x)
    info = w.steer(json.dumps({"layer": 1, "neuron": 5, "dose": 0.4, "sign": -1}))
    assert info == {"layer": 1, "layers": 4, "hidden": 8, "scale": -0.4}
    after = w.model_runner.model(x)
    assert torch.allclose(after[0], before[0])  # untouched before the layer
    v = w.model_runner.model.layers[1].mlp.down_proj.weight[:, 5].detach()
    v = v / v.norm()
    expected = before[1] + (-0.4) * before[1].norm(dim=-1, keepdim=True) * v
    assert torch.allclose(after[1], expected, atol=1e-5)


def test_random_direction_and_unsteer():
    w = worker()
    x = torch.randn(2, 8)
    before = w.model_runner.model(x)
    w.steer(json.dumps({"layer": 3, "seed": 7, "dose": 0.8, "sign": 1}))
    assert not torch.allclose(w.model_runner.model(x)[3], before[3])
    assert w.steer_state() == {"layer": 3, "scale": 0.8}
    w.unsteer()
    assert torch.allclose(w.model_runner.model(x)[3], before[3])
    assert w.steer_state() == {"steer": None}
