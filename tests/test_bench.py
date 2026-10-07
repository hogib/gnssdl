import math

import numpy as np
import pandas as pd
import pytest

from gnssdl import dataset
from gnssdl.bench import masks as bmasks
from gnssdl.bench.base import Reconstructor, apply_hide, station_scale
from gnssdl.bench.checks import run_checks
from gnssdl.bench.m0_zero import M0Zero
from gnssdl.bench.m1_stack import M1Stack
from gnssdl.bench.run import ScoringContext, run_model, save_results, scoreboard
from gnssdl.bench.score import (
    event_mask, per_station_scores, score_cells, scoring_qc, summarise,
)

N_STA = 30


@pytest.fixture(scope="module")
def toy_cube():
    """Stations that share a 3 mm common mode and have 1 mm own noise."""
    rng = np.random.default_rng(7)
    days = pd.date_range("2008-01-01", "2024-12-31", freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    common = rng.normal(0, 0.003, (len(days), 3))
    st = pd.DataFrame({
        "sta": [f"T{i:03d}" for i in range(N_STA)],
        "lat": 35.0 + rng.uniform(0, 1.0, N_STA),
        "lon": -118.0 + rng.uniform(0, 1.0, N_STA),
    })

    def load(sta):
        r = np.random.default_rng(int(sta[1:]))
        own = r.normal(0, 0.001, (len(days), 3))
        x = common + own
        return pd.DataFrame({"decyear": t, "e": 0.01 * (t - 2008) + x[:, 0], "n": x[:, 1],
                             "u": x[:, 2], "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003},
                            index=days)

    steps = pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    cube, _ = dataset.build_cube(st, load, steps)
    return cube


@pytest.fixture(scope="module")
def toy_masks(toy_cube):
    return bmasks.make_masks(toy_cube)


def test_masks_reproducible_and_only_evaluation_days(toy_cube, toy_masks):
    again = bmasks.make_masks(toy_cube)
    for k in toy_masks:
        np.testing.assert_array_equal(toy_masks[k], again[k])
    train = toy_cube.split_day == 0
    for k in ("scatter", "block", "station"):
        assert not toy_masks[k][:, train].any()
        assert not (toy_masks[k] & ~toy_cube.avail).any()
    seen = toy_cube.split_sta == 0
    assert not toy_masks["station"][seen].any()
    assert not toy_masks["scatter"][~seen].any() and not toy_masks["block"][~seen].any()
    eligible = (toy_cube.avail & ~toy_cube.exclude)[seen][:, ~train].sum()
    assert toy_masks["scatter"].sum() / eligible == pytest.approx(0.10, abs=0.01)
    days = pd.DatetimeIndex(toy_cube.days)
    for year in (2022, 2023):
        per = toy_masks["block"][:, days.year == year].sum(axis=1)
        assert per.max() <= bmasks.BLOCK_DAYS


def test_m0_scores_about_one(toy_cube, toy_masks):
    hide = toy_masks["scatter"]
    pred = M0Zero().predict(apply_hide(toy_cube, hide), hide)
    s = score_cells(toy_cube, pred, hide, station_scale(toy_cube))
    assert 0.8 < s["nrmse_e"] < 1.3 and 0.8 < s["nrmse_n"] < 1.3


def test_m1_removes_common_mode_and_passes_leak_checks(toy_cube, toy_masks):
    hide = toy_masks["scatter"]
    val = hide & (toy_cube.split_day == 1)[None, :]
    scale = station_scale(toy_cube)
    m1 = M1Stack(radius_km=0)
    m1.fit(toy_cube, val_hide=val,
           scorer=lambda p: summarise(per_station_scores(toy_cube, p, val, scale))["nrmse_fast"])
    pred = m1.predict(apply_hide(toy_cube, hide), hide)
    s = score_cells(toy_cube, pred, hide, scale)
    # own noise is 1 mm of ~3.2 mm total, so the floor is ~0.32
    assert s["nrmse_n"] < 0.45
    assert run_checks(m1, toy_cube, hide)["passed"]


class _CopiesHidden(Reconstructor):
    name = "leaky-hidden"

    def predict(self, cube, hide):
        return np.nan_to_num(cube.r, nan=0.0)   # reads hidden cells when they aren't NaN


class _UsesOwnHistory(Reconstructor):
    name = "leaky-own"

    def predict(self, cube, hide):
        own = np.nanmean(np.where(cube.avail[..., None], cube.r, np.nan), axis=1)
        return np.repeat(own[:, None, :], cube.r.shape[1], axis=1).astype(np.float32)


@pytest.mark.parametrize("leaky", [_CopiesHidden, _UsesOwnHistory])
def test_leak_checks_catch_leaky_models(toy_cube, toy_masks, leaky):
    assert not run_checks(leaky(), toy_cube, toy_masks["scatter"])["passed"]


def _no_steps():
    return pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])


def test_run_model_and_scoreboard(toy_cube, toy_masks, tmp_path):
    ctx = ScoringContext.build(toy_cube, _no_steps())
    for name in ("m0", "m1"):
        scores, configs = run_model(name, toy_cube, toy_masks, ctx, radii=[0.0, 25.0], log=lambda m: None)
        save_results(name, scores, configs, tmp_path)
        assert all(c["checks"]["passed"] for c in configs)
    board = scoreboard(tmp_path)
    assert list(board.columns) == ["scatter", "block", "station"]
    assert board.loc[("m1", "0"), "scatter"] < board.loc[("m0", "-"), "scatter"]


def test_offset_goes_to_slow_part_not_fast(toy_cube, toy_masks):
    hide = toy_masks["block"]
    scale = station_scale(toy_cube)
    zero = np.zeros_like(toy_cube.r)
    shifted = zero - 50.0 * scale[:, None, :].astype(np.float32)   # a constant 50-sigma error
    a = summarise(per_station_scores(toy_cube, zero, hide, scale))
    b = summarise(per_station_scores(toy_cube, shifted, hide, scale))
    assert b["nrmse_fast"] == pytest.approx(a["nrmse_fast"], rel=1e-6)
    assert b["nrmse_slow"] > 40 and b["nrmse_total"] > 40


def test_median_over_stations_resists_one_bad_station(toy_cube, toy_masks):
    hide = toy_masks["scatter"]
    scale = station_scale(toy_cube)
    zero = np.zeros_like(toy_cube.r)
    bad = zero.copy()
    i = int(np.flatnonzero(hide.any(axis=1))[0])
    bad[i] += 1e4
    a = summarise(per_station_scores(toy_cube, zero, hide, scale))
    b = summarise(per_station_scores(toy_cube, bad, hide, scale))
    assert b["nrmse_total"] == pytest.approx(a["nrmse_total"], rel=0.1)
    assert score_cells(toy_cube, bad, hide, scale)["nrmse_e"] > 100 * score_cells(toy_cube, zero, hide, scale)["nrmse_e"]


def test_event_mask_marks_window_around_listed_quakes(toy_cube):
    sta = str(toy_cube.sta[3])
    steps = pd.DataFrame([(sta, pd.Timestamp("2023-03-01"), "quake", "x", 100.0, 10.0, 6.5),
                          (sta, pd.Timestamp("2023-06-01"), "quake", "y", 100.0, 10.0, 5.5)],
                         columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    m = event_mask(toy_cube, steps)
    days = pd.DatetimeIndex(toy_cube.days)
    assert m[3, days.get_loc(pd.Timestamp("2023-03-20"))] and not m[3, days.get_loc(pd.Timestamp("2023-06-01"))]
    assert m.sum() == 61


def test_scoring_qc_flags_station_that_degrades(toy_cube):
    from dataclasses import replace
    r = toy_cube.r.copy()
    rng = np.random.default_rng(0)
    late = toy_cube.split_day > 0
    i = int(np.flatnonzero(toy_cube.split_sta == 0)[0])
    r[i, late] += rng.normal(0, 20.0, (late.sum(), 3)).astype(np.float32)
    keep, dropped = scoring_qc(replace(toy_cube, r=r))
    assert not keep[i] and list(dropped.sta) == [str(toy_cube.sta[i])]
    keep0, _ = scoring_qc(toy_cube)
    assert keep0.all()


def test_scoring_qc_flags_unexplained_jump_but_not_listed_one(toy_cube):
    from dataclasses import replace
    days = pd.DatetimeIndex(toy_cube.days)
    i, j = [int(k) for k in np.flatnonzero(toy_cube.split_sta == 0)[:2]]
    r = toy_cube.r.copy()
    r[[i, j], :, 0] += np.where(days >= pd.Timestamp("2023-05-01"), 300.0, 0.0)[None, :].astype(np.float32)
    steps = pd.DataFrame([(str(toy_cube.sta[j]), pd.Timestamp("2023-05-01"), "equipment", "Antenna", None, None, None)],
                         columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    keep, dropped = scoring_qc(replace(toy_cube, r=r), steps)
    assert not keep[i] and keep[j]
    assert dropped.iloc[0]["reason"] == "unexplained jump"


# --------------------------------------------------------------------------- #
# Signal-preservation tests
# --------------------------------------------------------------------------- #

from gnssdl.bench import signal as bsig  # noqa: E402


def test_time_profile_shape():
    g = bsig.time_profile(300, 30)
    assert g[0] == 0 and g[30:30 + bsig.HOLD_DAYS].min() == 1.0
    assert g[2 * 30 + bsig.HOLD_DAYS:].max() == 0.0
    assert np.all(np.diff(g[:30]) > 0)


@pytest.mark.parametrize("radius", [0.0, 100.0])
def test_injections_in_a_slot_never_affect_the_same_station(toy_cube, radius):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    runs = bsig.plan_injections(toy_cube, keep, centres_per_cell=2, period="test", radius_km=radius)
    n = sum(len(r) for r in runs)
    assert n == len(bsig.AMPLITUDES_MM) * len(bsig.FOOTPRINTS_KM) * len(bsig.DURATIONS_DAYS) * 2
    test_days = set(np.flatnonzero(toy_cube.split_day == 2))
    for run in runs:
        assert all(i.t0 in test_days for i in run)
        by_slot = {}
        for i in run:
            by_slot.setdefault(i.t0, []).append(i)
        for group in by_slot.values():
            taken = np.zeros(len(toy_cube.sta), dtype=int)
            for i in group:
                taken += bsig.affected(toy_cube, bsig.footprint(toy_cube, i)[1], radius)
            assert taken.max() <= 1


def _one_run(cube, L, amplitude=5.0, duration=30):
    c = len(cube.sta) // 2
    t0 = int(np.flatnonzero(cube.split_day == 2)[0])
    return [[bsig.Injection(0, amplitude, L, duration, float(cube.lat[c]), float(cube.lon[c]), t0, 1.0, 0.0)]]


def test_m0_keeps_every_transient(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    m0 = M0Zero()
    df = bsig.injection_scores(m0, toy_cube, keep, _one_run(toy_cube, 25.0))
    assert df.rho.iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_m1_absorbs_wide_transients_more_than_narrow(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    m1 = M1Stack(radius_km=0, length_km=math.inf)
    m1.fit(toy_cube)
    wide = bsig.injection_scores(m1, toy_cube, keep, _one_run(toy_cube, 200.0)).rho.iloc[0]
    narrow = bsig.injection_scores(m1, toy_cube, keep, _one_run(toy_cube, 5.0)).rho.iloc[0]
    assert wide < 0.3 < narrow


def test_rho_does_not_depend_on_amplitude_for_a_linear_model(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    m1 = M1Stack(radius_km=0, length_km=50.0)
    m1.fit(toy_cube)
    a = bsig.injection_scores(m1, toy_cube, keep, _one_run(toy_cube, 25.0, amplitude=2.0)).rho.iloc[0]
    b = bsig.injection_scores(m1, toy_cube, keep, _one_run(toy_cube, 25.0, amplitude=10.0)).rho.iloc[0]
    assert a == pytest.approx(b, rel=1e-3)


def test_ridgecrest_retention_is_one_for_m0(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    rc = bsig.ridgecrest_retention(M0Zero(), toy_cube, keep)
    assert len(rc) > 0
    np.testing.assert_allclose(rc.retained, 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# M2
# --------------------------------------------------------------------------- #

from gnssdl.bench.m2_robust import M2Robust, M2Self, weighted_median  # noqa: E402
from gnssdl.bench.run import load_model  # noqa: E402
from gnssdl.dataset import crop_days  # noqa: E402


def test_weighted_median_matches_ordinary_median_and_handles_weights():
    v = np.array([[1.0, 4.0, 2.0, 3.0]])
    assert weighted_median(v, np.ones_like(v))[0] == pytest.approx(2.5)        # mean of middle two
    assert weighted_median(np.array([[5.0, 1.0, 3.0]]), np.ones((1, 3)))[0] == pytest.approx(3.0)
    assert weighted_median(np.array([[7.0]]), np.ones((1, 1)))[0] == 7.0
    heavy = weighted_median(np.array([[0.0, 10.0]]), np.array([[1.0, 9.0]]))[0]
    assert 5.0 < heavy <= 10.0                                                # pulled to the heavy value
    ignored = weighted_median(np.array([[1.0, 2.0, 1000.0]]), np.array([[1.0, 1.0, 0.0]]))[0]
    assert ignored == pytest.approx(1.5)
    assert np.isnan(weighted_median(np.array([[1.0, 2.0]]), np.zeros((1, 2)))[0])


def _fit_m2(cube, **kw):
    m = M2Robust(radius_km=0, length_km=math.inf, k=3.0, **kw)
    m.fit(cube)
    return m


def test_m2_ignores_a_spiking_neighbour_where_m1_does_not(toy_cube):
    from dataclasses import replace
    m1 = M1Stack(radius_km=0, length_km=math.inf)
    m1.fit(toy_cube)
    m2 = _fit_m2(toy_cube)
    target = 0
    nb = int(toy_cube.nbr_idx[0, target, 0])
    t = int(np.flatnonzero(toy_cube.avail[nb] & toy_cube.avail[target] & (toy_cube.split_day == 2))[100])
    r = toy_cube.r.copy()
    r[nb, t, 0] += 300.0
    spiky = replace(toy_cube, r=r)
    no_hide = np.zeros(toy_cube.avail.shape, dtype=bool)
    d1 = abs(m1.predict(spiky, no_hide)[target, t, 0] - m1.predict(toy_cube, no_hide)[target, t, 0])
    d2 = abs(m2.predict(spiky, no_hide)[target, t, 0] - m2.predict(toy_cube, no_hide)[target, t, 0])
    assert d1 > 5.0 and d2 < 1.0


def test_m2_removes_common_mode_and_passes_leak_checks(toy_cube, toy_masks):
    hide = toy_masks["scatter"]
    scale = station_scale(toy_cube)
    m2 = _fit_m2(toy_cube)
    pred = m2.predict(apply_hide(toy_cube, hide), hide)
    assert score_cells(toy_cube, pred, hide, scale)["nrmse_n"] < 0.45
    assert run_checks(m2, toy_cube, hide)["passed"]


def test_m2_tuning_runs_and_stores_hyperparameters(toy_cube, toy_masks, tmp_path):
    ctx = ScoringContext.build(toy_cube, _no_steps())
    scores, configs = run_model("m2", toy_cube, toy_masks, ctx, radii=[0.0], log=lambda m: None)
    save_results("m2", scores, configs, tmp_path)
    assert configs[0]["length_km"] is not None and configs[0]["k"] is not None
    back = load_model("m2", 0.0, toy_cube, tmp_path)
    assert back.length_km == configs[0]["length_km"] and back.k == configs[0]["k"]
    self_ref = load_model("m2self", 0.0, toy_cube, tmp_path)
    assert self_ref.include_self and self_ref.k == back.k


def test_reference_filter_cannot_be_scored_on_masks(toy_cube, toy_masks):
    ctx = ScoringContext.build(toy_cube, _no_steps())
    with pytest.raises(ValueError):
        run_model("m2self", toy_cube, toy_masks, ctx, radii=[0.0], log=lambda m: None)


def test_m2self_keeps_less_of_a_narrow_transient_than_m2(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    m2 = _fit_m2(toy_cube)
    m2s = M2Self(radius_km=0, length_km=math.inf, k=3.0)
    m2s.fit(toy_cube)
    run = _one_run(toy_cube, 3.0)   # narrower than the station spacing
    a = bsig.injection_scores(m2, toy_cube, keep, run).rho.iloc[0]
    b = bsig.injection_scores(m2s, toy_cube, keep, run).rho.iloc[0]
    assert b < a


def test_cropping_does_not_change_m1_predictions(toy_cube):
    m1 = M1Stack(radius_km=0, length_km=50.0)
    m1.fit(toy_cube)
    no_hide = np.zeros(toy_cube.avail.shape, dtype=bool)
    full = m1.predict(toy_cube, no_hide)
    lo, hi = 3000, 3500
    small = crop_days(toy_cube, lo, hi)
    part = m1.predict(small, np.zeros(small.avail.shape, dtype=bool))
    np.testing.assert_allclose(part, full[:, lo:hi], atol=1e-5)


def test_noise_removed_is_zero_for_m0_and_positive_for_m1(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    assert bsig.noise_removed(M0Zero(), toy_cube, keep)["noise_removed"] == pytest.approx(0.0, abs=1e-9)
    m1 = M1Stack(radius_km=0, length_km=math.inf)
    m1.fit(toy_cube)
    assert bsig.noise_removed(m1, toy_cube, keep)["noise_removed"] > 0.3


def test_tuning_uses_all_validation_days_for_target_free_models(toy_cube, toy_masks):
    from gnssdl.bench.run import tuning_cells
    ctx = ScoringContext.build(toy_cube, _no_steps())
    hide, cells = tuning_cells(toy_cube, toy_masks, ctx, M1Stack)
    assert not hide.any()
    val = toy_cube.split_day == 1
    assert cells[:, val].sum() > 5 * toy_masks["scatter"][:, val].sum()
    assert not cells[:, ~val].any() and not cells[toy_cube.split_sta == 1].any()


def test_signal_tests_run_on_the_validation_period(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    runs = bsig.plan_injections(toy_cube, keep, centres_per_cell=1, period="validation")
    val_days = set(np.flatnonzero(toy_cube.split_day == 1))
    assert all(i.t0 in val_days for run in runs for i in run)
    assert "noise_removed" in bsig.noise_removed(M0Zero(), toy_cube, keep, period="validation")


# --------------------------------------------------------------------------- #
# Bootstrap comparisons
# --------------------------------------------------------------------------- #

from gnssdl.bench import stats as bstats  # noqa: E402


def _loo_stats(cube, model, period=1):
    cells = cube.avail & (cube.split_day == period)[None, :] & ~cube.exclude
    resid = cube.r - bsig.clean(model, cube)
    return bstats.block_stats(cube, resid, cells, station_scale(cube))


def test_bootstrap_separates_m1_from_m0_and_not_m0_from_itself(toy_cube):
    m1 = M1Stack(radius_km=0, length_km=math.inf)
    m1.fit(toy_cube)
    s0, s1 = _loo_stats(toy_cube, M0Zero()), _loo_stats(toy_cube, m1)
    res = bstats.paired_bootstrap(s0, s1, n_boot=200)
    assert res["ci_high"] < 0                      # M1 clearly lower error than M0
    same = bstats.paired_bootstrap(s0, s0, n_boot=50)
    assert same["ci_low"] == same["ci_high"] == 0.0


def test_rho_bootstrap_pairs_by_injection():
    a = pd.DataFrame({"id": range(40), "footprint_km": 25.0, "rho": np.linspace(0.2, 0.4, 40)})
    b = a.assign(rho=a.rho + 0.3)
    t = bstats.paired_rho_bootstrap(a, b, n_boot=200)
    assert t["diff"].iloc[0] == pytest.approx(0.3) and t["ci_low"].iloc[0] > 0.25


def test_spurious_signal_zero_for_m0_and_positive_for_m1_near_narrow_transient(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    run = _one_run(toy_cube, 8.0, amplitude=10.0)
    s0 = bsig.injection_scores(M0Zero(), toy_cube, keep, run).iloc[0]
    assert s0.spurious_max == 0.0
    m1 = M1Stack(radius_km=0, length_km=math.inf)
    m1.fit(toy_cube)
    s1 = bsig.injection_scores(m1, toy_cube, keep, run).iloc[0]
    assert s1.spurious_stations > 0 and 0.0 < s1.spurious_max <= 1.0
