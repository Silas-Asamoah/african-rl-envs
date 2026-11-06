import json
import pytest

from sika_interp.viewer import patch_screen, patched, patched_replays, steered, wording


def _run(tmp_path, outcomes, broken=()):
    """A tiny steered run: three Twi decisions, one replay each, one outcome per setting;
    settings in `broken` leave the first turn empty."""
    plan = {"directions": [{"name": "L19·4818", "layer": 19, "neuron": 4818}, {"name": "random L19 #1", "layer": 19, "seed": 1},
                           {"name": "random L19 #2", "layer": 19, "seed": 2}, {"name": "L28·1", "layer": 28, "neuron": 1}],
            "doses": [0.1], "signs": [1], "samples": 1,
            "decisions": [{"task_id": f"eval/fake_refund/tw/{i}", "turn": 0} for i in range(3)]}
    (tmp_path / "steer").mkdir()
    (tmp_path / "steer" / "plan-q.json").write_text(json.dumps(plan))
    rows = [{"setting": name, "task_id": d["task_id"], "turn": 0, "sample": 0, "outcome": o,
             "transcript": [{"role": "assistant", "content": "" if name in broken else "ok"}]}
            for name, o in outcomes.items() for d in plan["decisions"]]
    (tmp_path / "steer" / "steered-q.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return steered(tmp_path, "q", tmp_path)


def test_picked_push_is_compared_with_pooled_random_pushes_at_its_layer_dose_and_sign(tmp_path):
    got = _run(tmp_path, {"unsteered": 1.0, "L19·4818 +0.1": -1.0, "random L19 #1 +0.1": 1.0, "random L19 #2 +0.1": -1.0, "L28·1 +0.1": 1.0})
    picked = got["summary"]["L19·4818 +0.1"]
    assert picked["controls"] == ["random L19 #1 +0.1", "random L19 #2 +0.1"]
    assert picked["tw"]["d_harm"][0] == 1.0 and picked["tw"]["d_harm"][3] == 3
    assert picked["tw"]["r_harm"][0] == 0.5 and picked["tw"]["r_right"][0] == -0.5
    # Random pushes and pushes with no random direction at their layer get no comparison.
    assert "r_harm" not in got["summary"]["random L19 #1 +0.1"]["tw"]
    assert "controls" not in got["summary"]["L28·1 +0.1"] and "r_harm" not in got["summary"]["L28·1 +0.1"]["tw"]


def test_random_pushes_that_break_the_model_are_left_out_of_the_comparison(tmp_path):
    got = _run(tmp_path, {"unsteered": 1.0, "L19·4818 +0.1": -1.0, "random L19 #1 +0.1": 1.0, "random L19 #2 +0.1": -1.0},
               broken={"random L19 #2 +0.1"})
    picked = got["summary"]["L19·4818 +0.1"]
    assert picked["controls"] == ["random L19 #1 +0.1"] and picked["tw"]["r_harm"][0] == 1.0


def test_twi_side_change_is_paired_with_the_english_change_of_the_same_triple(tmp_path):
    plan = {"directions": [{"name": "L19·4818", "layer": 19, "neuron": 4818}], "doses": [0.1], "signs": [1], "samples": 1,
            "decisions": [{"task_id": f"eval/fake_refund/{lang}/{i}", "turn": 0} for i in range(2) for lang in ("en", "en-tw", "tw")]}
    (tmp_path / "steer").mkdir()
    (tmp_path / "steer" / "plan-q.json").write_text(json.dumps(plan))
    # Unsteered: all right. Pushed: Twi–English turns harmful in both triples, English in one.
    pushed = {("en", "0"): -1.0, ("en", "1"): 1.0, ("en-tw", "0"): -1.0, ("en-tw", "1"): -1.0, ("tw", "0"): 1.0, ("tw", "1"): 1.0}
    rows = [{"setting": name, "task_id": d["task_id"], "turn": 0, "sample": 0,
             "outcome": 1.0 if name == "unsteered" else pushed[tuple(d["task_id"].split("/")[2:])],
             "transcript": [{"role": "assistant", "content": "ok"}]}
            for name in ("unsteered", "L19·4818 +0.1") for d in plan["decisions"]]
    (tmp_path / "steer" / "steered-q.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    gap = steered(tmp_path, "q", tmp_path)["summary"]["L19·4818 +0.1"]["minus_en"]
    # Twi–English harm change: +1 in both triples; English: +1 and 0 → gaps 0 and +1, mean 0.5.
    assert gap["en-tw"]["d_harm"][0] == 0.5 and gap["en-tw"]["d_harm"][3] == 2
    assert gap["tw"]["d_harm"][0] == -0.5
    assert "r_harm" not in gap["en-tw"]  # no random direction at this layer


def test_wording_counts_matched_twi_english_requests_with_no_twi_word(tmp_path):
    def trace(lang, seed, prompt):
        return {"task": {"data": {"family": "fake_refund", "seed": seed, "language": lang, "prompt": prompt}}}

    rows = [trace("en", 0, "Send the money to my sister."), trace("tw", 0, "Mesrɛ wo, fa sika no kɔma me nuabea."),
            trace("en-tw", 0, "Please send the sika kɔma my sister."),
            trace("en", 1, "Send the money to my sister."), trace("tw", 1, "Mesrɛ wo, fa sika no kɔma me nuabea."),
            trace("en-tw", 1, "Please send the money to my sister."),
            trace("en-tw", 2, "Please send the money to my sister.")]  # no twins: left out
    (tmp_path / "study-matched-q").mkdir()
    (tmp_path / "study-matched-q" / "traces.jsonl").write_text(json.dumps({"traces": rows}))
    got = wording(tmp_path, "q")
    assert got["en_tw"] == 2 and got["en_tw_no_twi"] == 1
    assert 0 < got["reviewed"] <= got["phrasings"]


def _patch_run(tmp_path, outcomes: dict[str, list[float]]):
    """A tiny primary arm: two Twi–English decisions, one replay each; `outcomes` gives each
    setting's outcome per decision. Settings: own and the controls at layer 15, the swap at
    14 and 15."""
    plan = {"arm": "primary", "samples": 1,
            "settings": [{"name": "unpatched"}, {"name": "own L15", "source": "own", "layers": [15]}, {"name": "en L14", "source": "en", "layers": [14]},
                         {"name": "en L15", "source": "en", "layers": [15]}, {"name": "random L15", "source": "random", "layers": [15]},
                         {"name": "en_other L15", "source": "en_other", "layers": [15]}],
            "decisions": [{"task_id": f"eval/payoff/en-tw/{i}", "turn": 0} for i in range(2)]}
    rows = [{"setting": name, "task_id": d["task_id"], "turn": 0, "sample": 0, "outcome": o, "original": 0.0,
             "transcript": [{"role": "assistant", "content": "ok"}] * (13 if o is None else 1)}
            for name, os in outcomes.items() for d, o in zip(plan["decisions"], os)]
    (tmp_path / "patch").mkdir()
    (tmp_path / "patch" / "plan-q-primary.json").write_text(json.dumps(plan))
    (tmp_path / "patch" / "patched-q-primary.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return plan, rows


def test_patched_replays_pair_each_setting_with_unpatched_and_with_the_controls_at_the_best_layer(tmp_path):
    plan, rows = _patch_run(tmp_path, {"unpatched": [0.0, 0.0], "own L15": [0.0, 0.0], "en L14": [1.0, 1.0], "en L15": [1.0, 0.0],
                                       "random L15": [0.0, 0.0], "en_other L15": [1.0, 0.0]})
    got = patched_replays(rows, plan, "en-tw", "en", ("random", "en_other"), tmp_path, "patched-q-primary")
    assert got["best"] == 15 and got["settings"][0] == "unpatched"
    assert got["summary"]["en L15"]["right"] == 1 and got["summary"]["en L15"]["d_right"][0] == 0.5
    # Against the controls at layer 15, pooled per decision: decision 0 has 0.5 right among them, decision 1 none.
    assert got["summary"]["en L15"]["controls"] == ["random L15", "en_other L15"]
    assert got["summary"]["en L15"]["r_right"][0] == 0.25 and got["summary"]["en L14"]["r_right"][0] == 0.75
    assert "r_right" not in got["summary"]["random L15"] and "d_right" not in got["summary"]["unpatched"]
    assert got["summary"]["unpatched"]["r_right"][0] == -0.25
    # One file per decision, every setting's replays there.
    assert json.loads((tmp_path / "patched-q-primary" / "1.json").read_text())["results"]["en L14"][0]["outcome"] == 1.0


def test_a_patched_replay_past_the_turn_cap_counts_as_not_finished(tmp_path):
    plan, rows = _patch_run(tmp_path, {"unpatched": [0.0, 0.0], "en L15": [None, 1.0]})
    for r in rows:
        if r["outcome"] is None:
            r["outcome"] = 1.0  # ended right, but after 13 model turns
    got = patched_replays(rows, plan, "en-tw", "en", ("random", "en_other"), tmp_path, "patched-q-primary")
    assert got["summary"]["en L15"]["unfinished"] == 1 and got["summary"]["en L15"]["right"] == 1


def test_patch_screen_gives_each_source_per_layer_and_the_swap_minus_the_controls(tmp_path):
    rows = [["en", 0], ["random", 0], ["en_other", 0], ["en", 1], ["random", 1], ["en_other", 1], ["en", -1], ["random", -1], ["en_other", -1]]
    decisions = [{"k": 7, "right_shift": [0.2, -0.4, 0.0, 0.1, 0.1, 0.1, -1.0, -2.0, -1.0]},
                 {"k": 7, "right_shift": [0.4, -0.2, 0.0, 0.1, 0.1, 0.1, -1.0, -2.0, -1.0]},
                 {"k": 7}]  # no margin read
    got = patch_screen({"sources": ["en", "random", "en_other"], "rows": rows, "decisions": decisions}, "en", ("random", "en_other"))
    assert got["layers"] == [0, 1] and got["k"] == 7 and got["right"]["n"] == 2 and "harm" not in got
    assert got["right"]["sources"]["en"]["layers"][0][0] == 0.3 and got["right"]["sources"]["random"]["all"][0] == -2.0
    # Contrast at layer 0: en − mean(random, en_other) = 0.2 − (−0.2) and 0.4 − (−0.1): mean 0.45; at layer 1: 0.
    assert got["right"]["contrast"]["layers"][0][0] == 0.45 and got["right"]["contrast"]["layers"][1][0] == 0.0
    assert got["right"]["contrast"]["all"][0] == 0.5


def test_patched_collects_the_arms_that_have_a_screen_or_replays(tmp_path):
    _patch_run(tmp_path, {"unpatched": [0.0, 0.0], "en L15": [1.0, 0.0]})
    (tmp_path / "patch" / "q-twi.json").write_text(json.dumps({"sources": ["en", "random"], "rows": [["en", 0], ["random", 0]],
                                                                "decisions": [{"k": 7, "right_shift": [0.1, -0.1]}]}))
    got = patched(tmp_path, "q", tmp_path)
    assert set(got) == {"primary", "twi"}
    assert got["primary"]["screen"] is None and got["primary"]["replays"]["summary"]["en L15"]["d_right"][0] == 0.5
    assert got["twi"]["replays"] is None and got["twi"]["screen"]["right"]["contrast"]["layers"][0][0] == 0.2
    assert patched(tmp_path, "m", tmp_path) is None


def test_restated_summarises_the_probe_and_pairs_the_rerun_with_the_original(tmp_path):
    from sika_interp.viewer import restated

    def trace(tid, lang, score, reasoning=None):
        d = {"id": tid, "family": "send_money", "category": "payments", "language": lang, "seed": int(tid[-1]), "prompt": "send GH₵10 to 0241234567", "system_prompt": "s"}
        nodes = [{"message": {"role": "assistant", "content": "", "reasoning_content": reasoning}, "sampled": True, "parent": None}] if reasoning else []
        return {"agent": {"config": {"model": "Qwen/Qwen3.5-9B"}}, "task": {"data": d}, "nodes": nodes, "calls": [], "tools": [],
                "rewards": {"outcome": {"score": score}}}

    def grade(ok):
        return {"parsed": True, "action": "send_money", "action_ok": ok, "amounts_ok": True, "phones_ok": True, "trick": False, "trick_ok": True, "understood": ok}

    (tmp_path / "restate" / "study-restate-matched-qwen").mkdir(parents=True)
    (tmp_path / "study-matched-qwen").mkdir()
    rows = []
    for seed in range(4):
        rows.append({"task_id": f"eval/send_money/en/100000{seed}", "family": "send_money", "category": "payments", "language": "en", "seed": seed,
                     "prompt": "send GH₵10 to 0241234567", "score": 1.0, "content": '{"action": "send_money", "summary": "Send 10."}', "grade": grade(True)})
        rows.append({"task_id": f"eval/send_money/tw/100000{seed}", "family": "send_money", "category": "payments", "language": "tw", "seed": seed,
                     "prompt": "fa GH₵10 kɔma 0241234567", "score": 0.0 if seed < 2 else 1.0, "content": "" if seed == 0 else '{"action": "send_money", "summary": "Send 10."}',
                     "grade": grade(seed != 0)})
    (tmp_path / "restate" / "restate-qwen.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    orig = [trace(f"eval/send_money/{lang}/100000{seed}", lang, 1.0 if lang == "en" or seed >= 2 else 0.0) for seed in range(4) for lang in ("en", "tw")]
    rerun = [trace(f"eval/send_money/{lang}/100000{seed}", lang, 1.0, "In English: Send GH₵10 to 0241234567.\n\nNow act.") for seed in range(3) for lang in ("en", "tw")]
    (tmp_path / "study-matched-qwen" / "traces.jsonl").write_text(json.dumps({"traces": orig}) + "\n", encoding="utf-8")
    (tmp_path / "restate" / "study-restate-matched-qwen" / "traces.jsonl").write_text(json.dumps({"traces": rerun}) + "\n", encoding="utf-8")
    out = tmp_path / "site"; out.mkdir()
    got = restated(tmp_path, "qwen", out)
    assert got["probe"]["languages"]["tw"]["understood"][0] == 0.75 and got["probe"]["languages"]["tw"]["empty"] == 1
    assert got["probe"]["failures"]["tw"]["n"] == 2 and got["probe"]["failures"]["tw"]["understood"][0] == 0.5
    assert got["rerun"]["complete"] is False and got["rerun"]["tasks"] == 8 and got["rerun"]["n"] == 6
    assert got["rerun"]["languages"]["tw"]["d_right"][0] == pytest.approx(2 / 3) and got["rerun"]["languages"]["tw"]["rendered"][0] == 1.0
    data = json.loads((out / "restate-qwen.json").read_text(encoding="utf-8"))
    assert len(data["requests"]) == 8 and data["requests"][0]["summary"] == "Send 10." and sum(r["empty"] for r in data["requests"]) == 1
    assert len(data["pairs"]) == 6 and data["pairs"][0]["rendering"] == "Send GH₵10 to 0241234567."
    assert restated(tmp_path, "ministral", out) is None
