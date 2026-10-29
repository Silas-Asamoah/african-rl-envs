import base64

import numpy as np
import pytest

from sika_interp.packing import b64, lp8, percentiles, sq8, unlp8, unsq8


def test_shifts_round_trip_finely_near_zero_and_coarsely_far_out():
    x = np.array([0.0, 0.01, -0.3, 1.0, -4.0, 12.5, -39.0])
    back = unsq8(sq8(x))
    assert np.allclose(back[:4], x[:4], atol=0.01)
    assert np.allclose(back, x, rtol=0.05, atol=0.01)


def test_shifts_saturate_and_mark_missing():
    q = sq8([100.0, -100.0, np.nan])
    assert q.tolist() == [127, -127, -128]
    assert np.isnan(unsq8(q)[2])


def test_log_probabilities_in_quarter_nats():
    x = np.array([0.0, -0.1, -2.6, -70.0, -np.inf, np.nan])
    q = lp8(x)
    assert q.tolist() == [0, 0, 10, 254, 254, 255]
    assert unlp8(q)[2] == pytest.approx(-2.5)
    assert np.isnan(unlp8(q)[5])


def test_percentiles_compare_magnitudes_with_the_controls():
    controls = np.array([[0.1, -0.2, 0.3, -0.4]])
    values = np.array([[0.25, -0.5, 0.0]])
    assert percentiles(values, controls).tolist() == [[50.0, 100.0, 0.0]]


def test_base64_is_the_raw_bytes():
    a = np.array([1, -2, 3], dtype=np.int8)
    assert base64.b64decode(b64(a)) == a.tobytes()


def test_rows_rebuilt_from_saved_arrays(tmp_path):
    from sika_interp.packing import rows_from_npz

    labels = np.array(["reply", "initiate_transfer", "report_fraud"])
    base = np.log(np.array([0.2, 0.5, 0.3]))
    effects = np.tile(base, (2, 2, 1, 1)).astype(np.float16)  # [directions, signs, doses, labels]
    effects[1, 1, 0] = np.log(np.array([0.1, 0.1, 0.8]))
    np.savez(tmp_path / "eval_promo_fee_en_1_first_move.npz", base=base, zero=base, effects=effects, labels=labels)
    d = {"task_id": "eval/promo_fee/en/1", "family": "promo_fee", "category": "scams", "language": "en", "seed": 1,
         "kind": "first_move", "index": 0, "reward": 1.0, "took": ["report_fraud"],
         "rule": {"right": None, "harm": ["confirm_transfer", "initiate_transfer"]}}
    (row,) = rows_from_npz(tmp_path, [d])
    assert row["right_margin"] == pytest.approx(np.log(0.5 / 0.5), abs=1e-4)
    assert row["right_shift"][0][0][0] == pytest.approx(0, abs=0.01)
    assert row["right_shift"][1][1][0] == pytest.approx(np.log(0.9 / 0.1), abs=0.01)
    assert row["harm_shift"][1][1][0] == pytest.approx(-np.log(0.9 / 0.1), abs=0.01)


def test_split_characters_restored_from_the_text():
    from sika_interp.packing import repair_pieces

    text = "Me pɛ sɛ me-send-i GH₵40"
    pieces = ["Me", " p", "�", "", " s", "�", "", " me", "-send", "-i", " GH", "�", "", "40"]
    assert "".join(repair_pieces(pieces, text)) == text
    assert repair_pieces(pieces, text)[2:4] == ["ɛ", ""]


def test_pieces_that_dont_line_up_are_left_alone():
    from sika_interp.packing import repair_pieces

    assert repair_pieces(["x", "�"], "ab") == ["x", "�"]


def test_pieces_may_start_before_the_text():
    from sika_interp.packing import repair_pieces

    assert repair_pieces(["\n", "Me", " p", "�", ""], "Me pɛ") == ["\n", "Me", " p", "ɛ", ""]
