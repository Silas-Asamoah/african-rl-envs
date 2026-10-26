"""Shared pieces for the interpretability scripts.

- `tool_schemas()`: the 17 Sika tools as OpenAI-style schemas, from the toolset itself.
- `Runner`: renders chats exactly as an agent would see them, generates in batches, parses
  tool calls (Hermes/Qwen JSON or Qwen3-Coder XML), runs them on the wallet simulator and
  scores the final wallet with the environment's own grader.
- `mlp_modules()` / `capture_mlp()`: the MLP neurons of each layer (the input to down_proj),
  for activation capture and steering.

Needs torch and transformers:  uv run --with torch --with transformers python scripts/...
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_type_hints

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sika_agent import wallet  # noqa: E402
from sika_agent.checks import evaluate  # noqa: E402
from sika_agent.scenarios import FAMILIES, generate  # noqa: E402
from sika_agent.taskset import SYSTEM_PROMPT  # noqa: E402
from sika_agent.toolset import SikaToolset  # noqa: E402

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
UNSAFE_CATEGORIES = {"scams", "injection"}
READ_TOOLS = {
    "check_balance", "list_contacts", "get_fee_schedule", "list_transactions", "read_messages",
    "list_saved_billers", "get_loan_status",
}  # fmt: skip
MONEY_TOOLS = {"initiate_transfer", "confirm_transfer", "buy_airtime", "pay_bill", "make_loan_payment"}
PROTECT_TOOLS = {"report_fraud", "cancel_transfer"}
THIRD_PARTY_TOOLS = {"read_messages", "list_transactions", "initiate_transfer"}  # results quote other people's text


def tool_schemas() -> list[dict]:
    """OpenAI-style schemas for the Sika tools, from the toolset's signatures and docstrings."""
    schemas = []
    for name, fn in vars(SikaToolset).items():
        if name.startswith("_") or not inspect.isfunction(fn) or name == "run":
            continue
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        props, required = {}, []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            props[pname] = {"type": JSON_TYPES[hints[pname]]}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": inspect.getdoc(fn) or "",
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }
        )
    if len(schemas) != 17:
        raise RuntimeError(f"expected 17 tools, found {len(schemas)}")
    return schemas


TOOL_NAMES = {t["function"]["name"] for t in tool_schemas()}

# -- tool calls -------------------------------------------------------------------------

CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
XML_FUNC = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.S)
XML_PARAM = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)
THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)


@dataclass
class Call:
    name: str
    args: dict
    start: int
    """Character offset of the call's block in the assistant text."""
    end: int


def parse_calls(text: str) -> list[Call]:
    """Tool calls in an assistant message, in either Hermes JSON or Qwen3-Coder XML form."""
    calls = []
    for m in CALL_BLOCK.finditer(text):
        body = m.group(1).strip()
        xml = XML_FUNC.search(body)
        if xml:
            args = {}
            for k, v in XML_PARAM.findall(xml.group(2)):
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    args[k] = v
            calls.append(Call(xml.group(1), args, m.start(), m.end()))
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            calls.append(Call("__unparsed__", {"raw": body}, m.start(), m.end()))
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        calls.append(Call(str(obj.get("name", "__unparsed__")), args if isinstance(args, dict) else {}, m.start(), m.end()))
    return calls


def visible_text(text: str) -> str:
    """What the user reads: the message without thinking or tool-call blocks."""
    return CALL_BLOCK.sub("", THINK.sub("", text)).strip()


def run_tool(state, call: Call) -> str:
    if call.name not in TOOL_NAMES:
        return f"Error: unknown tool {call.name!r}."
    try:
        return getattr(wallet, call.name)(state, **call.args)
    except TypeError as e:
        return f"Error: bad arguments for {call.name}: {e}"
    except Exception as e:  # noqa: BLE001 - the agent sees tool errors as text
        return f"Error: {e}"


# -- episodes ---------------------------------------------------------------------------


@dataclass
class Episode:
    family: str
    language: str
    seed: int
    split: str
    state: object
    expect: object
    messages: list[dict]
    prefill: str = ""
    """Text forced at the start of the next assistant turn (for resampling mid-turn)."""
    user_turns: int = 3
    turns: int = 0
    done: bool = False
    outcome: str = ""
    """success, harmful or fail."""
    tool_names: list[str] = field(default_factory=list)
    """For each tool message, the tool that produced it."""

    def copy(self) -> Episode:
        return Episode(
            self.family, self.language, self.seed, self.split, self.state.model_copy(deep=True), self.expect,
            [dict(m) for m in self.messages], self.prefill, self.user_turns, self.turns, self.done, self.outcome,
            list(self.tool_names),
        )


def new_episode(family: str, language: str, seed: int, split: str = "eval") -> Episode:
    sc = generate(family, language, seed, split, matched=True)
    s = sc.state
    system = SYSTEM_PROMPT.format(owner=s.owner, number=s.owner_number, today=s.today)
    return Episode(
        family, language, seed, split, s, sc.expect,
        [{"role": "system", "content": system}, {"role": "user", "content": sc.prompt}],
    )  # fmt: skip


class Runner:
    """Batched agent episodes on a local Hugging Face model."""

    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str | None = None, template_kwargs=None,
                 max_new_tokens: int = 384, max_turns: int = 12, temperature: float = 0.7, top_p: float = 0.95):  # fmt: skip
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
        self.model.to(self.device).eval()
        self.model_id = model_id
        self.tools = tool_schemas()
        self.kwargs = template_kwargs if template_kwargs is not None else {"enable_thinking": False}
        self.max_new_tokens, self.max_turns = max_new_tokens, max_turns
        self.temperature, self.top_p = temperature, top_p
        stops = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != self.tok.unk_token_id:
                stops.add(tid)
        self.stops = sorted(s for s in stops if s is not None)

    def render(self, messages: list[dict], generation: bool = True) -> str:
        return self.tok.apply_chat_template(
            messages, tools=self.tools, add_generation_prompt=generation, tokenize=False, **self.kwargs
        )

    def generate(self, prompts: list[str], sample: bool = True) -> list[str]:
        torch = self.torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=sample, temperature=self.temperature if sample else None,
                top_p=self.top_p if sample else None, eos_token_id=self.stops, pad_token_id=self.tok.pad_token_id,
            )  # fmt: skip
        new = out[:, enc["input_ids"].shape[1] :]
        return [self.tok.decode(row, skip_special_tokens=True) for row in new]

    def step(self, eps: list[Episode], replies: list[str], limitless: bool = False) -> None:
        """Apply each episode's assistant reply: run its tool calls, or let the user answer, or finish."""
        for ep, gen in zip(eps, replies):
            text = ep.prefill + gen
            ep.prefill = ""
            ep.turns += 1
            ep.messages.append({"role": "assistant", "content": text})
            calls = parse_calls(text)
            if calls:
                for call in calls:
                    ep.messages.append({"role": "tool", "content": run_tool(ep.state, call), "name": call.name})
                    ep.tool_names.append(call.name)
            elif "?" in visible_text(text) and ep.user_turns > 0:
                ep.user_turns -= 1
                ep.state.events.append("ask_user")
                ep.state.questions.append(visible_text(text))
                ep.messages.append({"role": "user", "content": wallet.scripted_reply(ep.state)})
            else:
                ep.done = True
            if not ep.done and ep.turns >= self.max_turns and not limitless:
                ep.done = True
            if ep.done:
                self.finish(ep)

    def run(self, eps: list[Episode], batch: int = 8, sample: bool = True) -> list[Episode]:
        """Play every episode to the end."""
        while active := [e for e in eps if not e.done]:
            for i in range(0, len(active), batch):
                chunk = active[i : i + batch]
                prompts = [self.render(e.messages) + e.prefill for e in chunk]
                self.step(chunk, self.generate(prompts, sample))
        return eps

    def replay(self, ep: Episode, assistant_texts: list[str]) -> Episode:
        """Play an episode with given assistant turns (no model), for scripted demos and tests.

        The turn limit doesn't apply, so a script always plays in full."""
        for text in assistant_texts:
            if ep.done:
                break
            self.step([ep], [text], limitless=True)
        if not ep.done:
            ep.done = True
            self.finish(ep)
        return ep

    def finish(self, ep: Episode) -> None:
        """Score the final wallet with the environment's grader."""
        said = "\n".join(visible_text(m["content"]) for m in ep.messages if m["role"] == "assistant")
        v = evaluate(ep.state, ep.expect, said)
        ep.outcome = "success" if v.success else "harmful" if v.harmful else "fail"


# -- MLP neurons ------------------------------------------------------------------------

LAYER_RE = re.compile(r"layers\.(\d+)\.mlp$")


def mlp_modules(model) -> dict[int, object]:
    """Each layer's MLP module (a dense gate/up/down MLP), by layer index."""
    found = {}
    for name, mod in model.named_modules():
        m = LAYER_RE.search(name)
        if m and hasattr(mod, "down_proj"):
            found[int(m.group(1))] = mod
    if not found:
        raise RuntimeError("no dense MLP layers (…layers.N.mlp.down_proj) found in this model")
    return dict(sorted(found.items()))


@contextmanager
def capture_mlp(model, select: dict[int, list[int]] | None = None):
    """Record the MLP neuron activations (down_proj input) of each forward pass.

    Yields a dict {layer: tensor (batch, seq, neurons)} filled in by the next forward: all
    of a layer's neurons, or with `select` only the listed layers and neuron indices."""
    import torch

    mods = mlp_modules(model)
    store: dict[int, object] = {}
    handles = []
    for layer, mod in mods.items():
        if select is not None and layer not in select:
            continue
        idx = None if select is None else torch.tensor(select[layer], dtype=torch.long)

        def hook(_m, args, layer=layer, idx=idx):
            x = args[0].detach()
            store[layer] = x if idx is None else x[..., idx.to(x.device)]

        handles.append(mod.down_proj.register_forward_pre_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer_neuron(model, layer: int, idx: int, amount: float, positions: slice | None = None):
    """Add `amount` to one neuron's activation (so `amount * W_down[:, idx]` to the MLP output)."""
    mod = mlp_modules(model)[layer]

    def hook(_m, args):
        x = args[0]
        if positions is None:
            x[..., idx] += amount
        else:
            x[:, positions, idx] += amount
        return (x,)

    h = mod.down_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        h.remove()


def unsafe(family: str) -> bool:
    return FAMILIES[family][0] in UNSAFE_CATEGORIES
