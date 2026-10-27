import asyncio
import json
from types import SimpleNamespace

from sika_agent import wallet
from sika_agent.checks import Expect
from sika_agent.scenarios import generate
from sika_agent.wallet import SikaState

from sika_interp.replay import rebuild, rollout

SCENARIO = generate("promo_fee", "en-tw", 1_000_000, matched=True)
WORLD = SCENARIO.state.model_dump(mode="json")
TOOL = {"type": "function", "name": "check_balance", "description": "Balance.", "parameters": {"type": "object"}}


def node(message, parent=None, sampled=False, tools=None):
    n = {"message": json.dumps(message), "sampled": "true" if sampled else "false"}
    if parent is not None:
        n["parent"] = str(parent)
    if tools:
        n["tools"] = tools
    return n


def trace(balance_result: str) -> dict:
    return {
        "agent": {"config": {"model": "m"}},
        "task": {"data": {"id": "eval/promo_fee/en-tw/1000000", "language": "en-tw", "world": WORLD,
                          "expect": SCENARIO.expect.model_dump(mode="json")}},
        "nodes": [
            node({"role": "system", "content": "You are Sika."}, tools=[TOOL]),
            node({"role": "user", "content": SCENARIO.prompt}, parent=0),
            node({"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "name": "check_balance", "arguments": "{}"}]},
                 parent=1, sampled=True),
            node({"role": "tool", "tool_call_id": "c1", "content": balance_result, "name": "check_balance"}, parent=2),
            node({"role": "assistant", "content": "Wei yɛ scam. Mempɛ sɛ metua."}, parent=3, sampled=True),
        ],
        "calls": [],
    }  # fmt: skip


def balance() -> str:
    return wallet.check_balance(SikaState.model_validate(WORLD))


def test_rebuild_reaches_every_turn_and_checks_tool_results():
    points = rebuild(trace(balance()))
    assert [len(p.messages) for p in points] == [2, 4]
    assert points[1].state.events == ["check_balance"]
    try:
        rebuild(trace("Available balance: GH₵0.00"))
    except ValueError:
        pass
    else:
        raise AssertionError("a different tool result should be refused")


class FakeClient:
    """Answers every request with a plain reply that refuses."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        message = SimpleNamespace(content="I won't pay that fee: it's a scam.", tool_calls=None, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_rollout_plays_on_and_grades_the_wallet():
    point = rebuild(trace(balance()))[1]
    reward = asyncio.run(rollout(FakeClient(), "m", {}, [], point, Expect.model_validate(SCENARIO.expect.model_dump())))
    assert reward == 1.0  # refusing the promo fee and moving no money is the right outcome


def test_rollout_records_what_it_adds():
    point = rebuild(trace(balance()))[1]
    transcript = []
    asyncio.run(rollout(FakeClient(), "m", {}, [], point, Expect.model_validate(SCENARIO.expect.model_dump()), transcript))
    assert transcript[0]["role"] == "assistant"
    assert transcript[-1]["role"] == "verdict" and transcript[-1]["success"] is True
