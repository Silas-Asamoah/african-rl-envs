from sika_interp.steered import settings


def test_settings_start_unsteered_then_every_direction_dose_and_sign():
    plan = {"directions": [{"name": "L19·4818", "layer": 19, "neuron": 4818}, {"name": "random L19 #1", "layer": 19, "seed": 1000}],
            "doses": [0.4], "signs": [1, -1]}
    got = settings(plan)
    assert [s["name"] for s in got] == ["unsteered", "L19·4818 +0.4", "L19·4818 −0.4", "random L19 #1 +0.4", "random L19 #1 −0.4"]
    assert got[1]["neuron"] == 4818 and got[1]["sign"] == 1 and got[4]["seed"] == 1000 and got[4]["sign"] == -1
