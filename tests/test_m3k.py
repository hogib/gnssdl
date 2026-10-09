import numpy as np
import pytest

from gnssdl.bench import m3k_kernel
from gnssdl.bench.base import median_sigma
from gnssdl.bench.m1_stack import M1Stack
from gnssdl.bench.m3k_kernel import M3Kernel
from tests.test_audit import _cube


@pytest.fixture(scope="module")
def cube():
    return _cube(40, 2.0, 2.0)


def _scorer(cube):
    val = cube.avail & (cube.split_day == 1)[None, :]

    def score(pred):
        return float(np.sqrt(np.nanmean((cube.r - pred)[val] ** 2)))
    return score


def test_m3k_with_m1_kernel_reproduces_m1(cube):
    m1 = M1Stack(radius_km=0.0, length_km=50.0)
    m1.fit(cube)
    m3k = M3Kernel(radius_km=0.0)
    idx = cube.nbr_idx[0][:, :16]
    d = cube.nbr_dist[0][:, :16].astype(np.float64)
    sbar = median_sigma(cube)
    w = np.exp(-d**2 / (2 * 50.0**2))[..., None] / sbar[np.where(idx >= 0, idx, 0)] ** 2
    m3k._w = np.where((idx >= 0)[..., None], np.nan_to_num(w), 0.0)
    hide = np.zeros(cube.avail.shape, dtype=bool)
    np.testing.assert_allclose(m3k.predict(cube, hide), m1.predict(cube, hide), atol=1e-4)


def test_m3k_trains_saves_and_reloads(cube, tmp_path, monkeypatch):
    monkeypatch.setattr(m3k_kernel, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(m3k_kernel, "MAX_EPOCHS", 3)
    model = M3Kernel(radius_km=0.0)
    model.fit(cube, scorer=_scorer(cube))
    hide = np.zeros(cube.avail.shape, dtype=bool)
    pred = model.predict(cube, hide)
    score = _scorer(cube)
    assert score(pred) < 0.6 * score(np.zeros_like(pred))     # removes most of the common mode
    assert len(model.history) == 3 and (model.kernel() >= 0).all()
    again = M3Kernel(radius_km=0.0)
    again.fit(cube)                                            # no scorer: loads the checkpoint
    np.testing.assert_allclose(again.predict(cube, hide), pred, atol=1e-5)


def test_m3k_never_reads_target_or_hidden(cube, tmp_path, monkeypatch):
    from gnssdl.bench.checks import run_checks
    monkeypatch.setattr(m3k_kernel, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(m3k_kernel, "MAX_EPOCHS", 1)
    model = M3Kernel(radius_km=25.0)
    model.fit(cube, scorer=_scorer(cube))
    rng = np.random.default_rng(1)
    hide = cube.avail & (rng.random(cube.avail.shape) < 0.1)
    assert run_checks(model, cube, hide)["passed"]


def test_kernel_figures_are_written(cube, tmp_path, monkeypatch):
    from gnssdl.bench.kernel_plots import kernel_figures
    monkeypatch.setattr(m3k_kernel, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(m3k_kernel, "MAX_EPOCHS", 1)
    for R in (0.0, 25.0):
        M3Kernel(radius_km=R).fit(cube, scorer=_scorer(cube))
    paths = kernel_figures(M3Kernel, cube, tmp_path, tmp_path / "fig", radii=(0.0, 25.0))
    assert len(paths) == 4 and all(p.exists() and p.stat().st_size > 0 for p in paths)
