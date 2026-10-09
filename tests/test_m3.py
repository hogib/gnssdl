import numpy as np
import pytest

from gnssdl.bench.checks import run_checks
from gnssdl.bench.m1_stack import M1Stack
from gnssdl.bench.m3_ridge import M3Pooled, M3Ridge, features
from tests.test_audit import _cube


@pytest.fixture(scope="module")
def cube():
    return _cube(40, 2.0, 2.0)


def score_for(cube):
    val = cube.avail & (cube.split_day == 1)[None, :]
    return lambda pred: float(np.sqrt(np.mean((cube.r - pred)[val] ** 2)))


@pytest.mark.parametrize("cls", [M3Ridge, M3Pooled])
def test_m3_beats_zero_and_matches_m1(cube, cls):
    score = score_for(cube)
    model = cls(radius_km=0.0)
    model.fit(cube, scorer=score)
    pred = model.predict(cube, np.zeros(cube.avail.shape, dtype=bool))
    m1 = M1Stack(radius_km=0.0, length_km=np.inf)
    m1.fit(cube)
    p1 = m1.predict(cube, np.zeros(cube.avail.shape, dtype=bool))
    assert score(pred) < 0.5 * score(np.zeros_like(pred))
    assert score(pred) < 1.05 * score(p1)                       # at least about as good as a plain stack
    assert model.lam is not None and len(model.validation_curve) > 3


@pytest.mark.parametrize("cls", [M3Ridge, M3Pooled])
def test_m3_leak_checks(cube, cls):
    model = cls(radius_km=25.0)
    model.fit(cube, scorer=score_for(cube))
    rng = np.random.default_rng(0)
    hide = cube.avail & (rng.random(cube.avail.shape) < 0.05)
    assert run_checks(model, cube, hide)["passed"]


def test_feature_layout():
    z = np.random.default_rng(0).normal(size=(5, 16, 3))
    a = np.ones((5, 16))
    assert features(z, a, None).shape == (5, 65)
    assert features(z, a, np.ones((16, 3))).shape == (5, 257)
    np.testing.assert_array_equal(features(z, a, None)[:, 64], 1.0)  # intercept
