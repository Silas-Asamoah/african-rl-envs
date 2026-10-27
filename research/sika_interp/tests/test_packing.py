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
