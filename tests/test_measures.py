"""Analytic checks of the audit's measures (added 2026-10-30).

Each test plants something with a known answer and checks that the measure
returns it, so an error in indexing, pairing, windows or normalisation shows
up as a wrong number rather than a plausible one.
"""
import numpy as np
import pandas as pd
import pytest

from gnssdl import dataset, model as traj
from gnssdl.bench import signal as bsig
from gnssdl.bench.audit import FarStackSweep
from gnssdl.bench.base import Reconstructor
from gnssdl.bench.m0_zero import M0Zero
from gnssdl.bench.run import ScoringContext, seeded_label, write_merged
from gnssdl.bench.score import _split_fast_slow

N = 36


def _steps(rows=()):
    df = pd.DataFrame(list(rows), columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build(end="2025-12-31", extra=None, steps=None, seed=5):
    """Stations within ~200 km sharing a 3 mm common mode, 1 mm own noise.
    `extra(sta, days, t)` returns an (T, 3) array in metres added to one or
    more stations' positions."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2008-01-01", end, freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    common = rng.normal(0, 0.003, (len(days), 3))
    st = pd.DataFrame({"sta": [f"A{i:03d}" for i in range(N)],
                       "lat": 36.0 + rng.uniform(0, 1.5, N), "lon": -120.0 + rng.uniform(0, 1.5, N)})

    def load(sta):
        own = np.random.default_rng(int(sta[1:])).normal(0, 0.001, (len(days), 3))
        x = common + own
        if extra is not None:
            x = x + extra(sta, days, t)
        return pd.DataFrame({"decyear": t, "e": x[:, 0], "n": x[:, 1], "u": x[:, 2],
                             "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003}, index=days)

    steps = _steps() if steps is None else steps
    cube, _ = dataset.build_cube(st, load, steps)
    return cube, steps


@pytest.fixture(scope="module")
def plain():
    return _build()[0]


class _Copy(Reconstructor):
    """Reference filter that predicts the target's own value: removes everything."""
    name = "copy"
    reference_filter = True

    def predict(self, cube, hide):
        return np.nan_to_num(cube.r, nan=0.0)


# --------------------------------------------------------------------------- #
# Retention and spurious signal for a linear filter with a closed form
# --------------------------------------------------------------------------- #

def _far_stack_prediction_of(cube, s_full):
    """What the far stack at R = 0 predicts for a field s (S × T × 3): the
    mean over the other available stations, per day."""
    a = cube.avail[..., None]
    tot = np.where(a, s_full, 0.0).sum(axis=0, keepdims=True)
    cnt = a.sum(axis=0, keepdims=True)
    own = np.where(a, s_full, 0.0)
    return (tot - own) / np.maximum(cnt - a, 1)


def test_far_stack_retention_and_spurious_match_their_closed_forms(plain):
    cube = plain
    fs = FarStackSweep(radius_km=0.0); fs.fit(cube)
    keep = np.ones(len(cube.sta), dtype=bool)
    t0 = int(np.flatnonzero(cube.split_day == 1)[0]) + 10
    centre = np.argmin(np.hypot(cube.lat - cube.lat.mean(), cube.lon - cube.lon.mean()))
    inj = bsig.Injection(id=0, amplitude_mm=5.0, footprint_km=15.0, duration_days=30,
                         lat=float(cube.lat[centre]), lon=float(cube.lon[centre]), t0=t0, ue=1.0, un=0.0)
    df = bsig.injection_scores(fs, cube, keep, [[inj]])
    assert len(df) == 1
    # closed form: rho = <s - P(s), s> / <s, s> over stations within 2 L, horizontal
    sta, s, days = bsig.signal_of(cube, inj)
    s_full = np.zeros(cube.r.shape, dtype=np.float64)
    s_full[sta, days] = np.where(cube.avail[sta, days][..., None], s, 0.0)
    p = _far_stack_prediction_of(cube, s_full)
    d_all, inside = bsig.footprint(cube, inj)
    near = d_all <= bsig.SCORE_RADIUS * inj.footprint_km
    num = np.sum(((s_full - p) * s_full)[near][:, days, :2])
    den = np.sum((s_full * s_full)[near][:, days, :2])
    assert df.rho.iloc[0] == pytest.approx(num / den, abs=2e-3)
    assert 0.3 < df.rho.iloc[0] < 0.995       # the share that enters the network mean is lost
    # spurious: outside the footprint the far stack subtracts the network mean of s
    outside = ~inside
    assert outside.any() and inside.sum() >= 2
    ghost = np.hypot(p[outside][:, days, 0], p[outside][:, days, 1]).max(axis=1) / inj.amplitude_mm
    assert df.spurious_stations.iloc[0] == outside.sum()
    assert df.spurious_max.iloc[0] == pytest.approx(ghost.max(), abs=2e-3)
    assert df.spurious_median.iloc[0] == pytest.approx(np.median(ghost), abs=2e-3)
    assert df.spurious_median.iloc[0] > 0.01  # and it is present everywhere, not just at the worst station


def test_retention_of_m0_is_exactly_one_and_copy_filter_exactly_zero(plain):
    cube = plain
    keep = np.ones(len(cube.sta), dtype=bool)
    t0 = int(np.flatnonzero(cube.split_day == 1)[0]) + 10
    inj = bsig.Injection(id=0, amplitude_mm=5.0, footprint_km=25.0, duration_days=10,
                         lat=float(cube.lat[0]), lon=float(cube.lon[0]), t0=t0, ue=0.0, un=1.0)
    m0 = M0Zero(); m0.fit(cube)
    assert bsig.injection_scores(m0, cube, keep, [[inj]]).rho.iloc[0] == pytest.approx(1.0, abs=1e-6)
    cp = _Copy(); cp.fit(cube)
    assert bsig.injection_scores(cp, cube, keep, [[inj]]).rho.iloc[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Noise removed
# --------------------------------------------------------------------------- #

def test_noise_removed_is_one_for_a_filter_that_copies_the_target(plain):
    cp = _Copy(); cp.fit(plain)
    out = bsig.noise_removed(cp, plain, np.ones(len(plain.sta), bool), "validation")
    assert out["noise_removed"] == pytest.approx(1.0, abs=1e-6)
    assert out["stations"] == len(plain.sta)


def test_fast_slow_split_separates_a_slow_oscillation_from_white_noise():
    days = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1.0, (len(days), 3))
    slow = 5.0 * np.sin(2 * np.pi * np.arange(len(days)) / 365.0)[:, None] * np.ones((1, 3))
    f, s = _split_fast_slow(white + slow, days)
    assert np.allclose(np.sqrt(np.nanmean(f**2, axis=0)), 1.0, atol=0.08)   # fast part is the white noise
    assert np.corrcoef(s[:, 0], slow[:, 0])[0, 1] > 0.99                     # slow part is the oscillation


# --------------------------------------------------------------------------- #
# MIDAS velocity and velocity repeatability
# --------------------------------------------------------------------------- #

def test_midas_velocity_recovers_a_slope_despite_outliers_and_skips_pairs_across_steps():
    rng = np.random.default_rng(1)
    T = 1000
    x = np.zeros((1, T, 3))
    x[0, :, 0] = 3.0 * np.arange(T) / 365.25 + rng.normal(0, 0.3, T)
    x[0, rng.choice(T, 40, replace=False), 0] += 50.0                        # outliers
    ok = np.ones((1, T), bool)
    v = bsig.midas_velocity(x, ok, np.zeros((1, T), int))
    assert v[0, 0] == pytest.approx(3.0, abs=0.1)
    assert v[0, 1] == pytest.approx(0.0, abs=0.1)
    # a 20 mm step at day 500: 57% of the 365-day pairs span it, so without
    # skipping the median slope is the biased one (about 23 mm/yr)
    x[0, 500:, 0] += 20.0
    steps_before = np.zeros((1, T), int); steps_before[0, 500:] = 1
    assert bsig.midas_velocity(x, ok, steps_before)[0, 0] == pytest.approx(3.0, abs=0.1)
    assert bsig.midas_velocity(x, ok, np.zeros((1, T), int))[0, 0] > 10.0  # the bias the skipping removes
    # fewer than the minimum number of pairs: no result
    ok2 = ok.copy(); ok2[0, 365 + 50:] = False
    assert np.isnan(bsig.midas_velocity(x, ok2, steps_before)[0]).all()


def test_velocity_repeatability_recovers_a_planted_velocity_change_and_a_step_does_not_bias_it():
    def extra(sta, days, t):
        out = np.zeros((len(days), 3))
        if sta == "A000":                   # +5 mm/yr north from 2024 on
            out[:, 1] = 0.005 * np.clip(t - 2024.0, 0, None)
        if sta == "A001":                   # a 20 mm earthquake step mid-2024, catalogued
            out[:, 0] = 0.020 * (days >= pd.Timestamp("2024-07-01"))
        return out
    steps = _steps([("A001", "2024-07-01", "quake", "ev1", 100.0, 10.0, 6.5)])
    cube, steps = _build(extra=extra, steps=steps)
    ctx = ScoringContext.build(cube, steps)
    m0 = M0Zero(); m0.fit(cube)
    df = bsig.velocity_repeatability(m0, cube, ctx.keep_sta, steps, ctx.skip).set_index("sta")
    # the 3 mm common mode stays in M0's series, so each dv carries about 0.4 mm/yr of noise
    assert df.loc["A000", "dv_n"] == pytest.approx(5.0, abs=1.0)
    others = df.drop(index=["A000", "A001"], errors="ignore")
    assert (others[["dv_e", "dv_n", "dv_u"]].abs().median() < 0.8).all()
    if "A001" in df.index:                  # scored only if enough pairs avoid the step
        assert abs(df.loc["A001", "dv_e"]) < 0.5


# --------------------------------------------------------------------------- #
# Real earthquake offsets
# --------------------------------------------------------------------------- #

def test_quake_offsets_kept_is_one_for_m0_and_zero_for_a_copy_filter():
    def extra(sta, days, t):
        out = np.zeros((len(days), 3))
        if sta == "A002":
            out[:, 0] = 0.008 * (days >= pd.Timestamp("2020-06-01"))
        return out
    steps = _steps([("A002", "2020-06-01", "quake", "ev2", 100.0, 10.0, 6.1)])
    cube, steps = _build(end="2022-12-31", extra=extra, steps=steps)
    keep = np.ones(len(cube.sta), bool)
    m0 = M0Zero(); m0.fit(cube)
    df = bsig.quake_offsets(m0, cube, keep, steps, "validation")
    assert list(df.sta) == ["A002"]
    assert df.offset_mm.iloc[0] == pytest.approx(8.0, abs=2.5)   # 10-day means carry the 3 mm common mode
    assert df.kept.iloc[0] == pytest.approx(1.0, abs=1e-6)
    cp = _Copy(); cp.fit(cube)
    assert bsig.quake_offsets(cp, cube, keep, steps, "validation").kept.iloc[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Seasonal signal, cube fields, result files, seeds
# --------------------------------------------------------------------------- #

def test_seasonal_signal_matches_its_formula():
    coef = np.zeros((2, 3, 4)); coef[0, 0] = [1.0, 2.0, 0.5, -0.5]; coef[1, 2] = [0, 0, 0, 3.0]
    t = np.array([2020.0, 2020.125, 2020.25])
    s = traj.seasonal_signal(coef, t)
    w = 2 * np.pi * t
    assert np.allclose(s[0, :, 0], np.sin(w) + 2 * np.cos(w) + 0.5 * np.sin(2 * w) - 0.5 * np.cos(2 * w))
    assert np.allclose(s[0, :, 1:], 0.0) and np.allclose(s[1, :, 2], 3 * np.cos(2 * w))


def test_crop_days_keeps_station_level_fields(plain):
    small = dataset.crop_days(plain, 100, 400)
    assert small.r.shape[1] == 300 and small.seasonal.shape == plain.seasonal.shape
    assert np.array_equal(small.nbr_idx, plain.nbr_idx) and np.array_equal(small.qstep_amp, plain.qstep_amp)


def test_write_merged_keeps_other_labels_and_radii(tmp_path):
    p = tmp_path / "x.csv"
    write_merged(pd.DataFrame({"model": ["m5", "m5"], "radius_km": [0.0, 25.0], "v": [1, 2]}), p)
    write_merged(pd.DataFrame({"model": ["m5-s1"], "radius_km": [0.0], "v": [3]}), p)
    write_merged(pd.DataFrame({"model": ["m5"], "radius_km": [0.0], "v": [9]}), p)
    df = pd.read_csv(p)
    got = {(m, r): v for m, r, v in zip(df.model, df.radius_km, df.v)}
    assert got == {("m5", 0.0): 9, ("m5", 25.0): 2, ("m5-s1", 0.0): 3}
    assert seeded_label("m5", None) == "m5" and seeded_label("m5", 0) == "m5" and seeded_label("m5", 3) == "m5-s3"


def test_signal_command_refuses_a_seed_for_a_model_without_one(tmp_path, capsys):
    from gnssdl.cli import main
    rc = main(["bench", "signal", "m1", "--seed", "1", "--cube", str(tmp_path / "none.npz")])
    assert rc == 1 and "no training seed" in capsys.readouterr().err
