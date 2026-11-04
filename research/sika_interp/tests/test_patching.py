from sika_interp.patching import (
    ALL,
    arms,
    by_layer,
    choose_layers,
    conditions,
    has_twi,
    other_seed,
    request,
    shared_prefix,
    swap_minus_controls,
    template_span,
)


def test_shared_prefix_counts_leading_matches():
    assert shared_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert shared_prefix([1, 2], [1, 2, 3]) == 2
    assert shared_prefix([1], [2]) == 0
    assert shared_prefix([5, 6, 7], [5, 6, 7], [5, 6]) == 2
    assert shared_prefix() == 0


def test_template_span_is_the_whole_generation_prompt_at_turn_0():
    prompt = [1, 2, 3, 90, 91, 92]  # request ends at 3; then the generation prompt
    assert template_span(prompt, prompt, 3) == 3


def test_template_span_stops_where_an_earlier_model_turn_begins():
    turn0 = [1, 2, 3, 90, 91, 92]
    later = [1, 2, 3, 90, 91, 55, 56, 7, 8]  # the earlier turn's call starts after token 91
    assert template_span(later, turn0, 3) == 2


def test_template_span_is_zero_when_the_request_itself_differs():
    assert template_span([1, 9, 3, 90], [1, 2, 3, 90], 3) == 0
    assert template_span([1, 2], [1, 2, 3, 90], 3) == 0


def test_other_seed_prefers_the_next_seed_then_wraps():
    cands = [{"seed": 1}, {"seed": 3}, {"seed": 5}]
    assert other_seed(cands, 3) == {"seed": 5}
    assert other_seed(cands, 5) == {"seed": 1}
    assert other_seed(cands, 2) == {"seed": 3}
    assert other_seed([{"seed": 2}], 2) is None


def test_request_is_the_first_user_message():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "send 5"}, {"role": "tool", "content": "x"}]
    assert request(msgs) == "send 5"


def test_has_twi_needs_a_twi_word():
    assert has_twi("Please send GH₵20 kɔma me nua Ama.")
    assert not has_twi("Please send GH₵20 to my sister Ama.")


def _decision(family, seed, language, reward, text="Mesrɛ wo, send 20 ma me"):
    return {
        "task_id": f"eval/{family}/{language}/{seed}",
        "family": family,
        "seed": seed,
        "language": language,
        "kind": "first_move",
        "reward": reward,
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": f"{text} [{language}/{seed}]"},
                     {"role": "tool", "content": "fees"}],
    }


def test_arms_pair_failures_with_succeeding_english_twins_and_other_seeds():
    ds = [
        _decision("cashout_exact", 0, "en", 1.0), _decision("cashout_exact", 0, "en-tw", 0.0), _decision("cashout_exact", 0, "tw", 0.0),
        _decision("cashout_exact", 1, "en", 1.0), _decision("cashout_exact", 1, "en-tw", 1.0), _decision("cashout_exact", 1, "tw", -1.0),
        # English failed too: not a twin to patch from.
        _decision("cashout_exact", 2, "en", 0.0), _decision("cashout_exact", 2, "en-tw", 0.0), _decision("cashout_exact", 2, "tw", 0.0),
        # No Twi in the wording: dropped from the primary arm.
        _decision("promo_fee", 0, "en", 1.0), _decision("promo_fee", 0, "en-tw", 0.0, text="Send the fee to them"),
        _decision("promo_fee", 0, "tw", 0.0), _decision("promo_fee", 1, "en", 1.0),
        # A later decision kind is ignored.
        dict(_decision("cashout_exact", 0, "en-tw", 0.0), kind="after_answer"),
    ]
    out = arms(ds)
    assert [i["decision"]["task_id"] for i in out["primary"]] == ["eval/cashout_exact/en-tw/0"]
    item = out["primary"][0]
    assert item["swap"] == "en" and item["arm"] == "primary"
    assert set(item["sources"]) == {"own", "en", "en_other", "tw"}
    assert item["sources"]["en"][1]["content"].endswith("[en/0]")
    assert item["sources"]["en_other"][1]["content"].endswith("[en/1]")  # the next seed with English right
    assert item["sources"]["tw"][1]["content"].endswith("[tw/0]")
    assert all(len(v) == 2 for v in item["sources"].values())  # first-turn prompts only
    # Twi failures need no Twi check; promo_fee's Twi twin counts even though its Twi–English wording was dropped.
    assert [i["decision"]["task_id"] for i in out["twi"]] == [
        "eval/cashout_exact/tw/0", "eval/cashout_exact/tw/1", "eval/promo_fee/tw/0"
    ]
    assert out["twi"][0]["sources"]["tw"][1]["content"].endswith("[en-tw/0]")
    assert [i["decision"]["task_id"] for i in out["reverse"]] == ["eval/cashout_exact/en/0"]
    rev = out["reverse"][0]
    assert rev["swap"] == "en-tw" and set(rev["sources"]) == {"own", "en-tw"}
    assert rev["sources"]["en-tw"][1]["content"].endswith("[en-tw/0]")
    assert out["dropped"] == [{"task_id": "eval/promo_fee/en-tw/0", "why": "no Twi in the wording"}]


def test_arms_drop_a_family_with_no_other_english_seed():
    ds = [_decision("payoff", 0, "en", 1.0), _decision("payoff", 0, "en-tw", 0.0), _decision("payoff", 0, "tw", 0.0)]
    out = arms(ds)
    assert out["primary"] == [] and out["twi"] == []
    assert {d["why"] for d in out["dropped"]} == {"no other seed with English right"}


def test_conditions_cover_every_layer_then_all_layers():
    rows = conditions(3, ("en", "random"))
    assert rows == [("en", 0), ("en", 1), ("en", 2), ("random", 0), ("random", 1), ("random", 2), ("en", ALL), ("random", ALL)]


def _scored(seed, shifts):
    return {"seed": seed, "right_shift": shifts}


def test_layer_summaries_and_choice_on_even_seeds():
    rows = conditions(2, ("en", "random", "en_other"))
    # Layer 1 is where the swap beats the controls on even seeds; the odd seeds disagree.
    even = [_scored(0, [0.1, 1.0, 0.0, 0.2, 0.0, 0.1, 0.5, 0.1, 0.1]), _scored(2, [0.1, 1.2, 0.0, 0.2, 0.0, 0.1, 0.5, 0.1, 0.1])]
    odd = [_scored(1, [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0])]
    means = by_layer(rows, even + odd)
    assert means["en:1"] == round((1.0 + 1.2 + 0.2) / 3, 4)
    assert means["en:all"] == round((0.5 + 0.5 + 0.4) / 3, 4)
    contrast = swap_minus_controls(rows, even, "en", ("random", "en_other"))
    assert contrast == {"0": 0.1, "1": round(1.1 - (0.2 + 0.1) / 2, 4), "all": round(0.5 - 0.1, 4)}
    choice = choose_layers(rows, even + odd, "en", ("random", "en_other"), n=1)
    assert choice["chosen"] == [1]
    assert choice["even"] == {"1": 0.95}
    assert choice["odd_held_out"] == {"1": 0.2}
    assert choice["n_even"] == 2 and choice["n_odd"] == 1
    assert choice["all_layers_even"] == 0.4 and choice["all_layers_odd"] == 0.4


def test_decisions_without_a_margin_are_left_out_of_the_means():
    rows = conditions(1, ("en",))
    assert by_layer(rows, [{"seed": 0}, _scored(1, [0.5, 0.7])]) == {"en:0": 0.5, "en:all": 0.7}


def test_replay_plan_lists_decisions_and_settings_in_order():
    from sika_interp.patching import replay_plan

    items = [
        {"arm": "primary", "swap": "en", "decision": {"task_id": "eval/f/en-tw/0", "index": 2}},
        {"arm": "primary", "swap": "en", "decision": {"task_id": "eval/f/en-tw/1", "index": 0}},
        {"arm": "primary", "swap": "en", "decision": {"task_id": "eval/f/en-tw/2", "index": 1}},
    ]
    plan = replay_plan(items, "/specs", [14, 15], best=15, samples=3, every=2)
    assert plan["arm"] == "primary" and plan["samples"] == 3
    assert plan["decisions"] == [{"task_id": "eval/f/en-tw/0", "turn": 2}, {"task_id": "eval/f/en-tw/2", "turn": 1}]
    assert [s["name"] for s in plan["settings"]] == ["unpatched", "own L15", "en L14", "en L15", "half L15", "random L15", "en_other L15", "tw L15"]
    assert plan["settings"][2] == {"name": "en L14", "folder": "/specs", "source": "en", "layers": [14]}
    assert "folder" not in plan["settings"][0]
    reverse = replay_plan([{"arm": "reverse", "swap": "en-tw", "decision": {"task_id": "eval/f/en/0", "index": 0}}], "/r", [14], 14)
    assert [s["name"] for s in reverse["settings"]] == ["unpatched", "own L14", "en-tw L14", "random L14"]


def test_align_takes_the_request_end_from_the_first_turn_render():
    from sika_interp.patching import align

    req, tpl = [5, 6, 7], [90, 91, 92]
    own = [1, 2] + req + tpl
    # A later turn: the model echoed the request, then a call; the template after the echo differs.
    prompt = own + [40, 41] + req + [55, 56]
    twin = [1, 2, 8, 9, 7] + tpl
    other = [1, 2, 8, 8] + tpl[:2] + [77]  # shares only two template tokens
    assert align(prompt, {"own": (own, 5), "en": (twin, 5)}) == (5, 3)
    assert align(prompt, {"own": (own, 5), "en": (twin, 5), "en_other": (other, 4)}) == (5, 2)
    assert align([9] + own[1:], {"own": (own, 5)}) == (5, 0)  # the prompt doesn't start with the own render
