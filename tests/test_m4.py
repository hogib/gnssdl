import numpy as np
import pytest

from gnssdl.bench.checks import run_checks
from gnssdl.bench.m4_band import M4Band, M4Pooled, bands, fill_own, sigma_for, gaussian_smooth
from tests.test_audit import _cube


@pytest.fixture(scope="module")
def cube():
    return _cube(40, 2.0, 2.0)


def test_bands_sum_to_the_input():
    x = np.random.default_rng(0).normal(size=(3, 2000, 3))
    np.testing.assert_allclose(sum(bands(x)), x, atol=1e-4)


def test_smoother_passes_long_periods_and_halves_power_at_the_cutoff():
    t = np.arange(4000)
    for period in (60.0, 400.0):
        x = np.sin(2 * np.pi * t / period)[None, :, None] * np.ones((1, 1, 3))
        y = gaussian_smooth(x, sigma_for(period))[0, 1000:3000, 0]
        assert abs(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(x[0, 1000:3000, 0] ** 2)) - np.sqrt(0.5)) < 0.02


def test_fill_uses_only_the_station_itself():
    z = np.zeros((2, 10, 3))
    a = np.zeros((2, 10), dtype=bool)
    z[0, [0, 9], :] = [[0, 0, 0], [9, 9, 9]]
    a[0, [0, 9]] = True
    z[1] = 1e6                                                  # another station's values never enter
    out = fill_own(z, a)
    np.testing.assert_allclose(out[0, :, 0], np.arange(10))
    np.testing.assert_allclose(out[1], 0.0)


def score_for(cube):
    val = cube.avail & (cube.split_day == 1)[None, :]
    return lambda pred: float(np.sqrt(np.mean((cube.r - pred)[val] ** 2)))


@pytest.mark.parametrize("cls", [M4Band, M4Pooled])
def test_m4_learns_and_passes_leak_checks(cube, cls):
    model = cls(radius_km=0.0)
    model.fit(cube)
    pred = model.predict(cube, np.zeros(cube.avail.shape, dtype=bool))
    score = score_for(cube)
    assert score(pred) < 0.5 * score(np.zeros_like(pred))
    assert len(model.lams) == 4
    rng = np.random.default_rng(0)
    hide = cube.avail & (rng.random(cube.avail.shape) < 0.05)
    assert run_checks(model, cube, hide)["passed"]
