import json

from sika_interp.traces import chat_tool, load, turns

TOOL = {"type": "function", "name": "check_balance", "description": "The balance.", "parameters": {"type": "object"}}


def node(message, parent=None, sampled=False, tools=None):
    n = {"message": json.dumps(message), "sampled": "true" if sampled else "false"}
    if parent is not None:
        n["parent"] = str(parent)
    if tools:
        n["tools"] = tools
    return n


TRACE = {
    "task": {"data": {"id": "eval/promo_fee/en-tw/1000000", "language": "en-tw"}},
    "nodes": [
        node({"role": "system", "content": "You are Sika."}, tools=[TOOL]),
        node({"role": "user", "content": "Ei, m'a-win-i Sika Wallet promo no!"}, parent=0),
        node(
            {
                "role": "assistant",
                "reasoning_content": "Scam?",
                "tool_calls": [{"id": "c1", "type": "function", "name": "check_balance", "arguments": "{}"}],
            },
            parent=1,
            sampled=True,
        ),
        node({"role": "tool", "tool_call_id": "c1", "content": "GH₵1,248.85"}, parent=2),
        node({"role": "assistant", "content": "Mempɛ sɛ metua."}, parent=3, sampled=True),
    ],
    "calls": [{"node": 2, "usage": {"prompt_tokens": 2161}}, {"node": 4, "usage": {"prompt_tokens": 2822}}],
}


def test_flat_tools_go_back_to_the_shape_the_model_was_sent():
    assert chat_tool(TOOL) == {
        "type": "function",
        "function": {"name": "check_balance", "description": "The balance.", "parameters": {"type": "object"}},
    }
    nested = chat_tool(TOOL)
    assert chat_tool(nested) == nested


def test_each_model_turn_gets_the_request_that_produced_it():
    first, second = turns(TRACE)
    assert [m["role"] for m in first.messages] == ["system", "user"]
    assert first.response["reasoning_content"] == "Scam?"
    assert [m["role"] for m in second.messages] == ["system", "user", "assistant", "tool"]
    assert second.messages[2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "check_balance", "arguments": "{}"}}
    ]
    assert second.response["content"] == "Mempɛ sɛ metua."
    assert (first.prompt_tokens, second.prompt_tokens) == (2161, 2822)
    assert first.tools == [chat_tool(TOOL)]
    assert (first.task_id, first.language, second.index) == ("eval/promo_fee/en-tw/1000000", "en-tw", 1)


def test_load_reads_every_rollout_in_a_traces_file(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps({"traces": [TRACE]}) for _ in range(2)) + "\n", encoding="utf-8")
    assert len(list(load(path))) == 2
