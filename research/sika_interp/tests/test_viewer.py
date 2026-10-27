import json

from sika_interp.viewer import steered


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
