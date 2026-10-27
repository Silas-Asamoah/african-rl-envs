import math

import pytest
from sika_agent.scenarios import FAMILIES

from sika_interp.decisions import KINDS, MOVES, REPLY, actions, find, margin, moves
from sika_interp.traces import Turn

SYSTEM = {"role": "system", "content": "You are Sika."}
USER = {"role": "user", "content": "Send GH₵50 to Kwesi."}


def call(name):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": "{}"}}


def turn(index, messages, *names):
    response = {"role": "assistant", "content": "" if names else "Done.", "tool_calls": [call(n) for n in names]}
    return Turn("eval/x/en/1", "en", index, messages, [], response, None)


def result(name):
    return {"role": "tool", "name": name, "content": "ok"}


def test_every_family_is_listed_or_explained():
    """Families without moves are the two explained in the module docstring."""
    assert set(FAMILIES) - set(MOVES) == {"payoff", "reference_injection"}


def test_moves_name_real_tools_or_the_reply():
    from sika_agent import wallet

    tools = {name for name in dir(wallet) if not name.startswith("_")}
    for family, kinds in MOVES.items():
        assert set(kinds) <= set(KINDS), family
        for right, wrong in kinds.values():
            assert not set(right) & set(wrong), family
            assert (set(right) | set(wrong)) - {REPLY} <= tools, family


def test_payoff_depends_on_whether_the_balance_covers_it():
    assert moves("payoff", "first_move", {"loan_payment_min": 120000}) == ({"make_loan_payment"}, {REPLY, "ask_user"})
    assert moves("payoff", "first_move", {"loan_payment_min": None}) == ({REPLY}, {"make_loan_payment"})


def test_reference_injection_has_no_margin():
    assert moves("reference_injection", "first_move") is None


def test_a_turn_without_calls_is_the_reply():
    assert actions(turn(0, [SYSTEM, USER])) == [REPLY]


def test_find_skips_lookups_and_marks_each_kind_once():
    m0 = [SYSTEM, USER]
    t0 = turn(0, m0, "list_contacts")
    m1 = m0 + [t0.response, result("list_contacts")]
    t1 = turn(1, m1, "ask_user")
    m2 = m1 + [t1.response, result("ask_user")]
    t2 = turn(2, m2, "initiate_transfer")
    m3 = m2 + [t2.response, result("initiate_transfer")]
    t3 = turn(3, m3, "confirm_transfer")
    m4 = m3 + [t3.response, result("confirm_transfer")]
    t4 = turn(4, m4)
    assert find([t0, t1, t2, t3, t4]) == {"first_move": 1, "after_answer": 2, "after_initiate": 3}


def test_a_second_user_message_is_an_answer():
    m = [SYSTEM, USER, {"role": "assistant", "content": "Are you sure?"}, {"role": "user", "content": "Yes."}]
    assert find([turn(0, m, "initiate_transfer")])["after_answer"] == 0


def test_margin_from_groups():
    logp = {REPLY: math.log(0.2), "initiate_transfer": math.log(0.5), "report_fraud": math.log(0.3)}
    got = margin(logp, {}, {"report_fraud", REPLY}, {"initiate_transfer"})
    assert got == pytest.approx(math.log(0.5) - math.log(0.5))


def test_a_group_with_both_sides_counts_for_neither():
    logp = {"c": math.log(0.6), REPLY: math.log(0.4)}
    groups = {"c": ["cancel_transfer", "confirm_transfer"]}
    assert margin(logp, groups, {"cancel_transfer", REPLY}, {"confirm_transfer"}) is None
