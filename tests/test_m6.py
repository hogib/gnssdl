import numpy as np
import pytest

from gnssdl.bench import m6_network as m6
from gnssdl.bench.checks import run_checks
from gnssdl.dataset import distance_azimuth
from tests.test_audit import _cube


@pytest.fixture(scope="module")
def cube():
    return _cube(40, 2.0, 2.0)


def small(**kw):
    return m6.M6Config(**{**dict(d_model=32, heads=4, blocks=2, batch=4, max_steps=60, eval_every=30,
                                 patience=5, warmup=10, infer_hide=0.2, infer_batch=8), **kw})


def scorer_for(cube):
    val = cube.avail & (cube.split_day == 1)[None, :]

    def score(pred):
        return float(np.sqrt(np.mean((cube.r - pred)[val] ** 2)))
    return score


@pytest.fixture(scope="module")
def trained(cube, tmp_path_factory):
    m6.MODEL_DIR = tmp_path_factory.mktemp("models")
    model = m6.M6Aug(radius_km=0.0, config=small(p_aug=0.5))
    model.fit(cube, scorer=scorer_for(cube))
    return model


def test_m6_learns_the_common_mode(cube, trained):
    pred = trained.predict(cube, np.zeros(cube.avail.shape, dtype=bool))
    score = scorer_for(cube)
    assert np.isfinite(pred).all()
    assert score(pred) < 0.7 * score(np.zeros_like(pred))
    assert len(trained.history) == 2


def test_m6_never_reads_target_hidden_or_near_cells(cube, trained):
    rng = np.random.default_rng(0)
    hide = cube.avail & (rng.random(cube.avail.shape) < 0.05)
    assert run_checks(trained, cube, hide)["passed"]
    r25 = m6.M6Network(radius_km=25.0, config=small(max_steps=30, eval_every=30))
    r25.fit(cube, scorer=scorer_for(cube))
    assert run_checks(r25, cube, hide)["passed"]


@pytest.mark.parametrize("n,residual,radius", [(8, True, 0.0), (8, True, 25.0), (8, False, 25.0),
                                               (None, True, 0.0), (None, True, 25.0)])
def test_variants_learn_and_never_read_target_or_near_cells(cube, tmp_path, monkeypatch, n, residual, radius):
    monkeypatch.setattr(m6, "MODEL_DIR", tmp_path)
    model = m6.M6Network(radius_km=radius, config=small(n_neighbours=n, far_residual=residual, p_aug=0.5,
                                                        batch=8 if n else 4, infer_batch=32 if n else 8))
    model.fit(cube, scorer=scorer_for(cube))
    pred = model.predict(cube, np.zeros(cube.avail.shape, dtype=bool))
    assert np.isfinite(pred).all()
    assert scorer_for(cube)(pred) < 0.7 * scorer_for(cube)(np.zeros_like(pred))
    rng = np.random.default_rng(0)
    hide = cube.avail & (rng.random(cube.avail.shape) < 0.05)
    assert run_checks(model, cube, hide)["passed"]


def test_far_stack_excludes_target_and_its_disc(cube):
    import torch
    model = m6.M6Network(radius_km=25.0, config=small(far_residual=True))
    model._dev = "cpu"
    model._tables(cube)
    D = model._inputs(cube.r, cube.avail)
    x, a, _ = model._window(D, torch.as_tensor([200]))
    tg = torch.as_tensor([[3]])
    fs, _ = model._far_stack(x, a, tg)
    x2 = x.clone()
    x2[0, model._T["near"][3]] += 1e3                          # the target and its R-disc
    fs2, _ = model._far_stack(x2, a, tg)
    torch.testing.assert_close(fs, fs2)
    x2[0, ~model._T["near"][3]] += 1.0                         # every station beyond R moves: the stack follows
    assert not torch.allclose(model._far_stack(x2, a, tg)[0], fs)


def test_checkpoint_reloads(cube, trained):
    again = m6.M6Aug(radius_km=0.0, config=small(p_aug=0.5))
    again.fit(cube)
    hide = np.zeros(cube.avail.shape, dtype=bool)
    np.testing.assert_allclose(again.predict(cube, hide), trained.predict(cube, hide), atol=1e-4)


@pytest.mark.parametrize("radius", [0.0, 25.0, 100.0])
def test_groups_cover_every_station_once(cube, radius):
    d, _ = distance_azimuth(cube.lat, cube.lon)
    near = d <= radius
    np.fill_diagonal(near, True)
    groups = m6.hide_groups(near, 0.2, seed=0)
    allg = np.concatenate(groups)
    assert sorted(allg) == list(range(len(cube.sta)))
    for g in groups:                                          # within the cap unless a single target
        assert len(g) == 1 or near[g].any(axis=0).sum() <= round(0.2 * len(cube.sta))


def test_training_hidden_set_contains_targets_and_their_discs(cube):
    model = m6.M6Network(radius_km=25.0, config=small())
    model._dev = "cpu"
    model._tables(cube)
    avail_w = cube.avail[:, :64]
    targets, hidden = model._draw_hidden(avail_w, np.random.default_rng(0))
    assert len(targets) >= 1
    assert hidden[targets].all()
    assert hidden[model._near[targets].any(axis=0)].all()


def test_augmentation_changes_horizontal_values_only(cube, trained):
    import torch
    model = trained
    D = model._inputs(cube.r, cube.avail)
    x, a, _ = model._window(D, torch.as_tensor([100, 400], device=model._dev))
    model.cfg.p_aug = 1.0
    xa = model._augment(x, a, np.tile(np.arange(5), (2, 1)), np.random.default_rng(1))
    model.cfg.p_aug = 0.5
    diff = (xa - x).abs()
    assert diff[..., :2].max() > 0                            # horizontal values changed
    assert diff[..., 2].max() == 0                            # vertical untouched
    assert (diff[..., :2].amax(-1) * (a == 0)).max() == 0     # missing days untouched
    assert (D["x"] == model._inputs(cube.r, cube.avail)["x"]).all()   # labels' source untouched


def test_variant_registry():
    assert set(m6.VARIANTS) == {"m6r", "m6raug", "m6r16", "m6r16aug", "m6r64", "m6r64aug",
                                "m6r256", "m6r256aug"}
    m = m6.VARIANTS["m6r64aug"](radius_km=25.0)
    assert (m.cfg.n_neighbours, m.cfg.far_residual, m.cfg.p_aug) == (64, True, 0.5)
    assert m6.M6Network().cfg.far_residual is False and m6.M6Network().cfg.n_neighbours is None


def test_pair_features():
    c = _cube(6, 2.0, 2.0)
    p = m6.pair_features(c)
    assert p.shape == (7, 7, m6.N_PAIR)
    assert (p[np.arange(7), np.arange(7), 3] == 1).all()
    assert (p[6, :, 4] == 1).all() and (p[:, 6, 4] == 1).all()
