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


from sika_interp.steer_serve import find_patches, patch_specs  # noqa: E402


class TokenModel(torch.nn.Module):
    """vLLM-style model over a flattened batch: token ids and their positions in their own
    sequences; returns the residual stream after each layer."""

    def __init__(self, n=4, vocab=50, hidden=8):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList(Layer(hidden) for _ in range(n))

    def forward(self, input_ids, positions):
        hidden, residual = self.embed(input_ids), torch.zeros(input_ids.shape[0], 8)
        outs = []
        for layer in self.layers:
            hidden, residual = layer(hidden, residual)
            outs.append(hidden + residual)
        return outs


def token_worker():
    w = SteerExtension()
    w.model_runner = Runner(TokenModel())
    return w


def flattened(*sequences):
    """Token ids and positions of several sequences laid end to end, as vLLM batches them."""
    ids = torch.tensor([t for s in sequences for t in s])
    pos = torch.tensor([p for s in sequences for p in range(len(s))])
    return ids, pos


SIG = [11, 12, 13, 41, 42]  # three request tokens then two template tokens
B = [1, 2, 3, 4] + SIG + [7, 8, 9]  # the signature at positions 4..8


def spec(vectors):
    return {"signature": torch.tensor(SIG), "first": 4, "k": 2, "layers": [1, 3], "vectors": vectors}


def test_find_patches_needs_the_whole_signature_at_its_positions():
    vectors = torch.arange(2 * 2 * 8, dtype=torch.float32).view(2, 2, 8)
    a = [21, 22, 23, 24, 25, 26]
    decoy = [0] + B  # the same tokens, one position later
    cut = [1, 2, 3, 4, 11, 12]  # the signature split off by a chunk boundary
    ids, pos = flattened(a, B, decoy, cut)
    found = find_patches(ids, pos, [spec(vectors)])
    assert sorted(found) == [1, 3]
    idx, vec = found[1]
    assert idx.tolist() == [len(a) + 7, len(a) + 8]  # the template tokens of B only
    assert torch.equal(vec, vectors[0])
    assert torch.equal(found[3][1], vectors[1])
    assert find_patches(*flattened(a, decoy), [spec(vectors)]) == {}


def test_patch_sets_the_stream_at_the_template_tokens_after_the_chosen_layers(tmp_path):
    w = token_worker()
    model = w.model_runner.model
    ids, pos = flattened([21, 22, 23], B)
    before = model(ids, pos)
    states = {"en": torch.randn(2, 2, 8).half(), "own": torch.stack([before[1][3 + 7 : 3 + 9], before[3][3 + 7 : 3 + 9]]).half()}
    torch.save({"task_id": "t", "signature": torch.tensor(SIG), "first": 4, "k": 2, "layers": [1, 3], "states": states}, tmp_path / "t.pt")
    loaded = patch_specs(tmp_path, "en", [3], "cpu")
    assert len(loaded) == 1 and loaded[0]["layers"] == [3] and loaded[0]["vectors"].shape == (1, 2, 8)
    assert torch.equal(loaded[0]["vectors"][0], states["en"][1].float())

    info = w.patch(json.dumps({"folder": str(tmp_path), "source": "en", "layers": [1, 3]}))
    assert info == {"specs": 1, "source": "en", "layers": [1, 3], "k": [2]}
    after = model(ids, pos)
    tpl = [3 + 7, 3 + 8]
    assert torch.allclose(after[0], before[0])  # untouched before the first chosen layer
    assert torch.allclose(after[1][tpl], states["en"][0].float(), atol=1e-2)
    assert torch.allclose(after[3][tpl], states["en"][1].float(), atol=1e-2)
    others = [i for i in range(len(ids)) if i not in tpl]
    assert torch.allclose(after[1][others], before[1][others])
    assert not torch.allclose(after[2][tpl], before[2][tpl])  # the change flows on to the next layer
    assert w.patch_state() == {"specs": 1, "source": "en", "layers": [1, 3], "hits": 4}

    # The decision's own state is a no-op.
    w.patch(json.dumps({"folder": str(tmp_path), "source": "own", "layers": [1, 3]}))
    same = model(ids, pos)
    assert torch.allclose(same[3], before[3], atol=1e-2)
    w.unpatch()
    assert torch.allclose(model(ids, pos)[3], before[3])
    assert w.patch_state() == {"patch": None}
    assert model(ids[:3], pos[:3])  # a batch with no signature runs untouched


class EmbedsModel(TokenModel):
    """As vLLM runs a multimodal-capable model: embeddings in, ids left to the runner's
    buffer, and M-RoPE positions of shape [3, tokens]."""

    def forward(self, input_ids=None, positions=None, inputs_embeds=None):
        hidden, residual = inputs_embeds, torch.zeros(inputs_embeds.shape[0], 8)
        outs = []
        for layer in self.layers:
            hidden, residual = layer(hidden, residual)
            outs.append(hidden + residual)
        return outs


class Buffer:
    def __init__(self, gpu):
        self.gpu = gpu


def test_patch_reads_the_runner_buffers_when_the_model_gets_embeddings(tmp_path):
    w = SteerExtension()
    model = EmbedsModel()
    w.model_runner = Runner(model)
    ids, pos = flattened([21, 22, 23], B)
    w.model_runner.input_ids = Buffer(torch.cat([ids, torch.zeros(5, dtype=ids.dtype)]))  # a longer buffer
    mrope = pos[None].repeat(3, 1)
    embeds = model.embed(ids)
    before = model(input_ids=None, positions=mrope, inputs_embeds=embeds)
    states = {"en": torch.randn(2, 2, 8).half()}
    torch.save({"task_id": "t", "signature": torch.tensor(SIG), "first": 4, "k": 2, "layers": [1, 3], "states": states}, tmp_path / "t.pt")
    w.patch(json.dumps({"folder": str(tmp_path), "source": "en", "layers": [1]}))
    after = model(input_ids=None, positions=mrope, inputs_embeds=embeds)
    tpl = [3 + 7, 3 + 8]
    assert torch.allclose(after[1][tpl], states["en"][0].float(), atol=1e-2)
    assert torch.allclose(after[0], before[0])
    assert w.patch_state()["hits"] == 2
    probe = w.patch_probe()
    assert probe["forwards"] == 1 and probe["candidate_passes"] == 1 and probe["matched_passes"] == 1
    assert probe["last"] == {"input_ids": [[len(ids)], str(ids.dtype)], "positions": [len(ids)]}
    assert "mismatch" not in probe


def test_patch_probe_records_the_first_mismatch():
    w = token_worker()
    model = w.model_runner.model
    spec_ = {"task_id": "t", "signature": torch.tensor(SIG), "first": 4, "k": 2, "layers": [1], "vectors": torch.zeros(1, 2, 8)}
    w._steer_install()
    w._patch_state = {"specs": [spec_], "firsts": torch.tensor([4]), "source": "en", "layers": [1]}
    w._patch_hits = 0
    ids, pos = flattened([1, 2, 3, 4, 11, 12, 49, 41, 42, 7])  # one token differs from the signature
    model(ids, pos)
    probe = w.patch_probe()
    assert probe["candidate_passes"] == 1 and probe["matched_passes"] == 0
    assert probe["mismatch"] == {"task_id": "t", "batch_tokens": [11, 12, 49, 41, 42], "batch_positions": [4, 5, 6, 7, 8], "signature": SIG}
    assert w.patch_state()["hits"] == 0


def test_find_patches_takes_the_template_as_far_as_it_goes():
    vectors = torch.arange(2 * 2 * 8, dtype=torch.float32).view(2, 2, 8)
    later_turn = [1, 2, 3, 4, 11, 12, 13, 41, 30, 31]  # the second template token replaced by a model turn
    found = find_patches(*flattened(later_turn), [spec(vectors)])
    idx, vec = found[1]
    assert idx.tolist() == [7]
    assert torch.equal(vec, vectors[0][:1])
    no_template = [1, 2, 3, 4, 11, 12, 13, 30, 31]  # the request without any template token
    assert find_patches(*flattened(no_template), [spec(vectors)]) == {}
    other_request = [1, 2, 3, 4, 11, 12, 14, 41, 42]  # a request token differs
    assert find_patches(*flattened(other_request), [spec(vectors)]) == {}
