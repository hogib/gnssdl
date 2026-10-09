import numpy as np
import pytest

from gnssdl.bench import m5_attention as m5
from gnssdl.bench.checks import run_checks
from tests.test_audit import _cube


@pytest.fixture(scope="module")
def cube():
    return _cube(40, 2.0, 2.0)


def small(**kw):
    return m5.M5Config(**{**dict(d_model=32, heads=4, blocks=2, batch=16, max_steps=60, eval_every=30,
                                 patience=5, k=8, candidates=16, infer_batch=256), **kw})


def scorer_for(cube):
    val = cube.avail & (cube.split_day == 1)[None, :]

    def score(pred):
        return float(np.sqrt(np.mean((cube.r - pred)[val] ** 2)))
    return score


@pytest.fixture(scope="module")
def trained(cube, tmp_path_factory):
    m5.MODEL_DIR = tmp_path_factory.mktemp("models")
    model = m5.M5Aug(radius_km=0.0, config=small(p_aug=0.5))
    model.fit(cube, scorer=scorer_for(cube))
    return model


def test_m5_learns_the_common_mode(cube, trained):
    pred = trained.predict(cube, np.zeros(cube.avail.shape, dtype=bool))
    score = scorer_for(cube)
    assert np.isfinite(pred).all()
    assert score(pred) < 0.7 * score(np.zeros_like(pred))
    assert len(trained.history) == 2


def test_m5_never_reads_target_hidden_or_near_cells(cube, trained):
    rng = np.random.default_rng(0)
    hide = cube.avail & (rng.random(cube.avail.shape) < 0.05)
    assert run_checks(trained, cube, hide)["passed"]
    r25 = m5.M5Attention(radius_km=25.0, config=small(max_steps=30, eval_every=30))
    r25.fit(cube, scorer=scorer_for(cube))
    assert run_checks(r25, cube, hide)["passed"]


def test_checkpoint_reloads(cube, trained):
    again = m5.M5Aug(radius_km=0.0, config=small(p_aug=0.5))
    again.fit(cube)
    hide = np.zeros(cube.avail.shape, dtype=bool)
    np.testing.assert_allclose(again.predict(cube, hide), trained.predict(cube, hide), atol=1e-4)


def test_augmentation_changes_inputs_not_labels(cube, trained):
    import torch
    model = trained
    cells = cube.avail & (cube.split_day == 0)[None, :]
    D = model._inputs(cube, cube.r, cells)
    tg = torch.arange(8, device=model._dev)
    st = torch.full((8,), 100, device=model._dev)
    model.cfg.p_aug = 1.0
    torch.manual_seed(0)
    f_aug = model._batch(D, tg, st, rng=np.random.default_rng(1))[0]
    model.cfg.p_aug = 0.0
    torch.manual_seed(0)
    f_clean = model._batch(D, tg, st, rng=np.random.default_rng(1))[0]
    model.cfg.p_aug = 0.5
    diff = (f_aug - f_clean).abs()
    assert diff[:, 1:, :, 0:2].max() > 0                      # neighbours' horizontal inputs changed
    assert diff[:, 0].max() == 0                               # the target node is untouched
    assert diff[..., 2].max() == 0                             # vertical untouched


def test_affected_includes_far_stack_reach(cube, trained):
    inside = np.zeros(len(cube.sta), dtype=bool)
    inside[0] = True
    assert trained.affected_by(cube, inside).all()             # R = 0: every station averages station 0


def test_station_with_fewer_candidates_than_k_gets_finite_predictions(cube, tmp_path):
    m5.MODEL_DIR = tmp_path
    # 40 stations: every station has 39 candidates, fewer than k = 48, so padded slots are chosen
    model = m5.M5Attention(radius_km=0.0, config=small(k=48, candidates=64, max_steps=10, eval_every=10))
    model.fit(cube, scorer=scorer_for(cube))
    assert np.isfinite(model.predict(cube, np.zeros(cube.avail.shape, dtype=bool))).all()
