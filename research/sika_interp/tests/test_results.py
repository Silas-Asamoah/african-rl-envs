import json

import pytest

from sika_interp.results import episodes, table, wilson


def trace(language, reward, ok=True, category="scams"):
    return {
        "ok": ok,
        "errors": [] if ok else [{"type": "Timeout"}],
        "agent": {"config": {"model": "Qwen/Qwen3.5-9B"}},
        "task": {
            "data": {
                "id": f"eval/promo_fee/{language}/1000000",
                "family": "promo_fee",
                "category": category,
                "language": language,
                "seed": 1000000,
            }
        },
        "rewards": {"outcome": {"score": reward}} if ok else {},
    }


def test_wilson_matches_known_values():
    low, high = wilson(51, 100)
    assert low == pytest.approx(0.4128, abs=1e-3) and high == pytest.approx(0.6064, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)


def test_errored_rollouts_are_left_out(tmp_path):
    path = tmp_path / "traces.jsonl"
    rows = [trace("en", 1.0), trace("tw", -1.0), trace("en-tw", 0.0), trace("tw", 0.0, ok=False)]
    path.write_text("\n".join(json.dumps({"traces": [r]}) for r in rows), encoding="utf-8")
    eps, errored = episodes([str(path)])
    assert (len(eps), errored) == (3, 1)
    text = table(eps)
    assert "| all | 3 | 33% (6–79) | 1 |" in text
    assert "| en-tw | 1 | 0% (0–79) | 0 |" in text
