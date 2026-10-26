import pytest

from sika_interp.serving import for_ministral, reasoning_system_parts

PROMPT = "# HOW YOU SHOULD THINK\n\nUse this template:[THINK]Your scratch work.[/THINK]Then answer."
CONVERSATION = [
    {"role": "system", "content": "You are Sika."},
    {"role": "user", "content": "Ei, m'a-win-i Sika Wallet promo no!"},
    {"role": "assistant", "reasoning_content": "Sounds like a scam.", "content": "", "tool_calls": [{"id": "1"}]},
    {"role": "tool", "tool_call_id": "1", "content": "Available balance: GH₵1,248.85"},
    {"role": "assistant", "reasoning_content": "Refuse.", "content": "Mempɛ sɛ metua."},
]


def test_system_prompt_gains_the_reasoning_template_as_a_thinking_block():
    parts = reasoning_system_parts("You are Sika.", PROMPT)
    assert parts[0] == {"type": "text", "text": "You are Sika.\n\n# HOW YOU SHOULD THINK\n\nUse this template:"}
    assert parts[1] == {"type": "thinking", "thinking": "Your scratch work.", "closed": True}
    assert parts[2] == {"type": "text", "text": "Then answer."}


def test_earlier_reasoning_stays_in_context_as_thinking_blocks():
    out = for_ministral(CONVERSATION, PROMPT)
    assert out[2]["content"] == [{"type": "thinking", "thinking": "Sounds like a scam.", "closed": True}]
    assert out[2]["tool_calls"] == [{"id": "1"}]
    assert out[4]["content"][1] == {"type": "text", "text": "Mempɛ sɛ metua."}
    assert not any("reasoning_content" in m for m in out)


def test_the_input_is_not_modified():
    before = [dict(m) for m in CONVERSATION]
    for_ministral(CONVERSATION, PROMPT)
    assert CONVERSATION == before


def test_a_prompt_without_the_template_is_refused():
    with pytest.raises(ValueError):
        reasoning_system_parts("You are Sika.", "No template here.")
