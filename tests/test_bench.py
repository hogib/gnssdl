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


def test_injections_are_separated_within_a_slot(toy_cube):
    keep = np.ones(len(toy_cube.sta), dtype=bool)
    runs = bsig.plan_injections(toy_cube, keep, centres_per_cell=2)
    n = sum(len(r) for r in runs)
    assert n == len(bsig.AMPLITUDES_MM) * len(bsig.FOOTPRINTS_KM) * len(bsig.DURATIONS_DAYS) * 2
    test_days = np.flatnonzero(toy_cube.split_day == 2)
    for run in runs:
        assert all(i.t0 in test_days for i in run)
        by_slot = {}
        for i in run:
            by_slot.setdefault(i.t0, []).append(i)
        for group in by_slot.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    d = bsig.distance_azimuth(np.r_[group[a].lat, group[b].lat], np.r_[group[a].lon, group[b].lon])[0][0, 1]
                    assert d >= 2 * bsig.FOOTPRINT_CUTOFF * group[a].footprint_km + bsig.SEPARATION_MARGIN_KM


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
