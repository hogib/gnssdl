"""Cascadia slow-slip episodes (contract §1.6): tremor clustering and the retention measure."""
import numpy as np
import pandas as pd
import pytest

from gnssdl.bench import episodes as eps
from gnssdl.bench import signal as bsig
from gnssdl.bench.m0_zero import M0Zero


def synthetic_tremor():
    """Two bursts: a 20-day northern episode migrating south, and a 2-day
    blip that must not become an episode; a thin background everywhere."""
    rng = np.random.default_rng(1)
    rows = []
    for k in range(20):                                  # 2023-04-01 .. 04-20, 48.9 -> 47.9
        day = pd.Timestamp("2023-04-01") + pd.Timedelta(days=k)
        lat = 48.9 - k / 20 + rng.normal(0, 0.1, 150)
        rows.append(pd.DataFrame({"lat": lat, "lon": -123.0 + rng.normal(0, 0.1, 150), "day": day}))
    for k in range(2):                                   # too short
        day = pd.Timestamp("2023-06-01") + pd.Timedelta(days=k)
        rows.append(pd.DataFrame({"lat": 45.0 + rng.normal(0, 0.1, 100), "lon": -123.5, "day": day}))
    days = pd.date_range("2022-01-01", "2023-12-31")
    rows.append(pd.DataFrame({"lat": rng.uniform(40, 50, len(days)), "lon": -123.0, "day": days}))
    return pd.concat(rows, ignore_index=True)


def test_cluster_tremor_finds_the_sustained_burst_only():
    out = eps.cluster_tremor(synthetic_tremor())
    assert len(out) == 1
    ep = out.iloc[0]
    assert str(ep.start) == "2023-04-01" and str(ep.end) == "2023-04-20"
    assert 47.7 < ep.lat_min < 48.1 and 48.7 < ep.lat_max < 49.1
    assert ep.n_epicentres >= 0.9 * 20 * 150          # epicentres in cells below MIN_PER_CELL are left out


def test_tremor_band_and_interpolation():
    band = eps.tremor_band(synthetic_tremor())
    assert set(band.columns) == {"lat", "lon"}
    assert np.allclose(band.lon, -123.0, atol=0.05)
    lon = eps.band_longitude(band, np.array([30.0, 48.3, 60.0]))   # beyond the ends: held constant
    assert np.allclose(lon, -123.0, atol=0.05)


def test_read_tremor_skips_malformed_rows(tmp_path):
    p = tmp_path / "tremor.csv"
    p.write_text("lat,lon,depth,starttime,energy,duration \n"
                 " 48.1, -123.2, 20.0, 2023-04-05 01:00:00, 1.0, 300.0 \n"
                 "\n"
                 " 47.9, -123.1, 20.0, 2023-04-06 02:00:00, 1.0, 300.0 \n"
                 " 35.0, -118.0, 20.0, 2023-04-06 02:00:00, 1.0, 300.0 \n")   # outside the margin
    df = eps.read_tremor(p)
    assert len(df) == 2 and str(df.day.iloc[0].date()) == "2023-04-05"


N_BAND, N_FAR = 16, 12


def _cascadia(common_north_mm: float = 0.0):
    """16 stations on the tremor band with a 4 mm east step over an episode
    window, and 12 stations about 370 km east of it without the step. With
    `common_north_mm`, every station also steps north by that much on the
    same day (motion the whole network shares)."""
    from gnssdl import dataset
    rng = np.random.default_rng(3)
    days = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    st = pd.DataFrame({"sta": [f"C{i:03d}" for i in range(N_BAND + N_FAR)],
                       "lat": np.r_[np.linspace(47.0, 49.0, N_BAND), np.linspace(47.0, 49.0, N_FAR)],
                       "lon": np.r_[-123.0 + rng.uniform(-0.3, 0.3, N_BAND), -118.0 + rng.uniform(-0.3, 0.3, N_FAR)]})
    on = days >= pd.Timestamp("2023-04-11")                           # mid-episode
    step = np.where(on, 0.004, 0.0)
    common = np.where(on, common_north_mm / 1000.0, 0.0)

    def load(sta):
        k = int(sta[1:])
        x = np.random.default_rng(k).normal(0, 0.0005, (len(days), 3))
        return pd.DataFrame({"decyear": t, "e": (step if k < N_BAND else 0.0) + x[:, 0], "n": common + x[:, 1],
                             "u": x[:, 2], "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003}, index=days)

    steps = pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    cube, _ = dataset.build_cube(st, load, steps)
    return cube


@pytest.fixture(scope="module")
def cascadia_cube():
    return _cascadia()


BAND = pd.DataFrame({"lat": np.arange(40.125, 50.0, 0.25), "lon": -123.0})
EPISODE = {"id": "T20230401", "source": "test", "timing": "tremor", "start": pd.Timestamp("2023-04-01"),
           "end": pd.Timestamp("2023-04-20"), "lat_min": 47.5, "lat_max": 49.5, "note": ""}


class Copy(M0Zero):
    """Predicts the series itself: nothing is kept."""
    never_reads_target = True

    def predict(self, c, hide):
        return np.nan_to_num(c.r).astype(np.float32)


class FarMean(M0Zero):
    """Predicts, for every station, the daily mean of the stations east of
    118.5°W: removes the shared motion and none of the band's slip."""
    never_reads_target = True

    def predict(self, c, hide):
        far = c.lon > -118.5
        m = np.nanmean(np.where(c.avail[far][..., None], c.r[far], np.nan), axis=0)
        return np.broadcast_to(np.nan_to_num(m), c.r.shape).astype(np.float32)


def test_episode_retention_is_one_for_m0_and_zero_for_a_copy_filter(cascadia_cube):
    cube = cascadia_cube
    episodes = pd.DataFrame([
        EPISODE,
        {"id": "far", "source": "test", "timing": "gnss", "start": pd.Timestamp("2022-02-01"),
         "end": pd.Timestamp("2022-02-20"), "lat_min": 42.0, "lat_max": 44.0, "note": ""},   # no stations
        {"id": "early", "source": "test", "timing": "gnss", "start": pd.Timestamp("2019-12-20"),
         "end": pd.Timestamp("2020-01-05"), "lat_min": 47.0, "lat_max": 49.0, "note": ""},   # window off the cube
    ])
    keep = np.ones(len(cube.sta), dtype=bool)
    df = bsig.episode_retention(M0Zero(), cube, keep, episodes, BAND)
    assert list(df.columns) == bsig.EPISODE_COLUMNS
    assert set(df.episode) == {"T20230401"}
    band_sta = (cube.lon < -120) & (cube.lat >= 47.5)                 # the far stations give no rows
    assert (df.lat >= 47.5).all() and len(df) == band_sta.sum() and set(df.sta) == set(cube.sta[band_sta])
    assert (df.far == N_FAR).all() and (df.floor_mm < 0.5).all()
    # nothing is shared here, so the reference is noise only and the unfiltered series keeps the estimate
    assert np.allclose(df.retained, 1.0, atol=0.08) and (df.obs_mm > 3.5).all() and (df.slip_mm > 3.5).all()
    summ = bsig.episode_summary(df)
    assert summ.loc[summ.timing == "tremor", "retained"].item() == pytest.approx(1.0, abs=0.08)
    assert summ.loc[summ.timing == "tremor", "episodes"].item() == 1
    assert summ.loc[summ.timing == "tremor", "weak"].item() == 0

    assert np.allclose(bsig.episode_retention(Copy(), cube, keep, episodes, BAND).retained, 0.0, atol=1e-6)


def test_episode_retention_sets_the_shared_motion_aside():
    """Every station steps 6 mm north with the band's 4 mm east. A filter
    that removes the shared motion alone keeps the slip estimate (the
    measure before 2026-11-12 gave it 16 / 52), and one that removes
    everything keeps none. The unfiltered series still holds the shared
    motion, which here is at right angles to the slip, so it scores 1 too."""
    cube = _cascadia(common_north_mm=6.0)
    keep = np.ones(len(cube.sta), dtype=bool)
    episodes = pd.DataFrame([EPISODE])
    raw = bsig.episode_retention(M0Zero(), cube, keep, episodes, BAND)
    assert (raw.obs_mm > 6.5).all() and np.allclose(raw.slip_mm, 4.0, atol=0.6)
    for model in (FarMean(), M0Zero()):
        per = bsig.episode_table(bsig.episode_retention(model, cube, keep, episodes, BAND))
        assert per.retained.item() == pytest.approx(1.0, abs=0.08) and per.counted.item()
    assert np.allclose(bsig.episode_retention(Copy(), cube, keep, episodes, BAND).retained, 0.0, atol=1e-6)


def test_episode_retention_needs_far_stations(cascadia_cube):
    cube = cascadia_cube
    keep = cube.lon < -120                                            # no scored far station
    assert bsig.episode_retention(M0Zero(), cube, keep, pd.DataFrame([EPISODE]), BAND).empty


def test_episode_without_slip_is_listed_as_weak(cascadia_cube):
    """A window with no step: the slip estimate is at the noise floor, so the
    episode is not counted."""
    quiet = dict(EPISODE, id="quiet", start=pd.Timestamp("2022-06-01"), end=pd.Timestamp("2022-06-20"))
    keep = np.ones(len(cascadia_cube.sta), dtype=bool)
    df = bsig.episode_retention(M0Zero(), cascadia_cube, keep, pd.DataFrame([EPISODE, quiet]), BAND)
    per = bsig.episode_table(df).set_index("episode")
    assert per.counted.to_dict() == {"T20230401": True, "quiet": False}
    summ = bsig.episode_summary(df)
    assert summ.episodes.item() == 1 and summ.weak.item() == 1


def _rows(episode, n, num, slip_mm=4.0, floor_mm=0.5):
    return pd.DataFrame({"episode": episode, "timing": "gnss", "num": num, "den": 1.0,
                         "slip_mm": slip_mm, "floor_mm": floor_mm, "far": 12, "retained": num}, index=range(n))


def test_episode_summary_pools_and_drops_thin_episodes():
    df = pd.concat([_rows("a", 12, 0.9), _rows("b", 3, 0.1)], ignore_index=True)
    df.loc[0, ["num", "den"]] = [0.0, 13.0]                           # one large station: pooled, not the median
    summ = bsig.episode_summary(df)
    assert summ.episodes.item() == 1 and summ.weak.item() == 0
    assert summ.retained.item() == pytest.approx(11 * 0.9 / 24.0)


def test_episode_summary_refuses_rows_of_the_old_measure():
    old = pd.DataFrame({"episode": ["a"] * 12, "timing": "gnss", "retained": 0.9})
    with pytest.raises(ValueError, match="2026-11-12"):
        bsig.episode_summary(old)


def test_training_exclusion_marks_band_stations_over_the_window_only():
    episodes = pd.DataFrame([
        {"id": "M19-x", "source": "t", "timing": "gnss", "start": pd.Timestamp("2010-08-03"),
         "end": pd.Timestamp("2010-09-25"), "lat_min": 47.2, "lat_max": 48.8, "note": ""},
        {"id": "T2023", "source": "t", "timing": "tremor", "start": pd.Timestamp("2023-04-05"),
         "end": pd.Timestamp("2023-04-30"), "lat_min": 47.0, "lat_max": 49.0, "note": ""},
    ])
    band = pd.DataFrame({"lat": np.arange(40.125, 50.0, 0.25), "lon": -123.0})
    lat = np.array([48.0, 48.0, 46.0, 48.0])
    lon = np.array([-123.0, -122.0, -123.0, -118.0])        # on the band, 74 km east, out of range, 370 km east
    days = pd.date_range("2010-01-01", "2023-12-31", freq="D").to_numpy("datetime64[D]")
    ex = eps.training_exclusion(lat, lon, days, episodes, band)
    d = pd.DatetimeIndex(days)
    window = (d >= "2010-07-24") & (d <= "2010-10-05")       # 10-day margins
    assert ex[0].sum() == window.sum() and ex[0][window].all()
    assert ex[1].sum() == window.sum()                        # within 150 km of the band
    assert not ex[2].any() and not ex[3].any()                # wrong latitude; too far east
    assert not ex[:, d >= "2023-01-01"].any()                 # tremor-timed episodes are not excluded by default
    ex2 = eps.training_exclusion(lat, lon, days, episodes, band, timing=("gnss", "tremor"))
    assert ex2[0, d == "2023-04-10"].all()


def test_build_cube_applies_the_extra_exclusion(cascadia_cube):
    from gnssdl import dataset
    cube = cascadia_cube
    assert not cube.exclude.any()                             # no Ridgecrest station, no hook
    rng = np.random.default_rng(5)
    days = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    st = pd.DataFrame({"sta": [f"E{i:03d}" for i in range(6)], "lat": np.linspace(47, 49, 6), "lon": -123.0})

    def load(sta):
        x = rng.normal(0, 0.001, (len(days), 3))
        return pd.DataFrame({"decyear": t, "e": x[:, 0], "n": x[:, 1], "u": x[:, 2],
                             "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003}, index=days)

    steps = pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    hook = lambda lat, lon, d: ((pd.DatetimeIndex(d) >= "2020-01-01") & (pd.DatetimeIndex(d) < "2020-02-01"))[None, :] & (lat >= 48.0)[:, None]
    cube2, summary = dataset.build_cube(st, load, steps, extra_exclude=hook)
    jan = (pd.DatetimeIndex(cube2.days) >= "2020-01-01") & (pd.DatetimeIndex(cube2.days) < "2020-02-01")
    assert summary["extra_excluded_cells"] == 31 * (cube2.lat >= 48.0).sum()
    assert cube2.exclude[cube2.lat >= 48.0][:, jan].all() and not cube2.exclude[cube2.lat < 48.0].any()
    assert not cube2.exclude[:, ~jan].any()


def test_episodes_in_period_never_returns_test_year_episodes(cascadia_cube):
    from dataclasses import replace
    cube = cascadia_cube
    days = pd.DatetimeIndex(cube.days)
    split = np.where(days < "2022-01-01", 0, np.where(days < "2023-01-01", 1, 2)).astype(np.int8)
    cube = replace(cube, split_day=split)
    episodes = pd.DataFrame([
        {"id": "train", "timing": "gnss", "start": pd.Timestamp("2021-03-01"), "end": pd.Timestamp("2021-03-20"),
         "lat_min": 47, "lat_max": 49, "source": "", "note": ""},
        {"id": "val", "timing": "tremor", "start": pd.Timestamp("2022-05-01"), "end": pd.Timestamp("2022-05-20"),
         "lat_min": 47, "lat_max": 49, "source": "", "note": ""},
        {"id": "straddle", "timing": "tremor", "start": pd.Timestamp("2022-12-20"), "end": pd.Timestamp("2023-01-10"),
         "lat_min": 47, "lat_max": 49, "source": "", "note": ""},
        {"id": "test", "timing": "tremor", "start": pd.Timestamp("2023-04-01"), "end": pd.Timestamp("2023-04-20"),
         "lat_min": 47, "lat_max": 49, "source": "", "note": ""},
    ])
    assert list(bsig.episodes_in_period(cube, episodes, "validation").id) == ["train", "val"]
    assert list(bsig.episodes_in_period(cube, episodes, "test").id) == ["test"]


def test_cli_signal_has_an_episodes_branch():
    import inspect
    from gnssdl import cli
    src = inspect.getsource(cli._bench_signal)
    assert "if args.episodes:" in src and "episodes_in_period" in src


def test_signal_report_prints_seasonal_offsets_and_episodes(tmp_path, capsys):
    """The report reads each signal kind on its own: no planted-transient
    file, an empty file left by an old run, and the seasonal, offset and
    episode tables all print."""
    from gnssdl.cli import _signal_report
    v = tmp_path / "fs" / "signal" / "validation"
    v.mkdir(parents=True)
    (v / "ridgecrest.csv").write_text("model,radius_km\n")         # empty leftover, ignored
    pd.DataFrame({"model": "fs", "radius_km": 0.0, "sta": ["A", "B"], "days": 600,
                  **{c: [0.1, 0.3] for c in ("rho_e", "rho_n", "rho_u", "rho_h", "rho_all")}}
                 ).to_csv(v / "seasonal.csv", index=False)
    pd.DataFrame({"model": "fs", "radius_km": 0.0, "sta": ["A"], "date": "2020-03-31", "event": "e1",
                  "mag": 6.5, "offset_mm": 4.9, "kept": [0.9]}).to_csv(v / "offsets.csv", index=False)
    n = bsig.EPISODE_MIN_STATIONS
    pd.DataFrame({"model": "fs", "radius_km": 0.0, "episode": ["E1"] * n + ["E2"] * 2, "timing": "gnss",
                  "sta": [f"S{i}" for i in range(n + 2)], "lat": 47.0, "band_km": 10.0, "obs_mm": 5.0,
                  "slip_mm": 4.0, "floor_mm": 0.5, "far": 12, "num": [0.2] * n + [0.9] * 2, "den": 1.0,
                  "retained": [0.2] * n + [0.9] * 2}).to_csv(v / "episodes.csv", index=False)
    _signal_report(tmp_path, "validation")
    out = capsys.readouterr().out
    assert "seasonal kept" in out and "0.2" in out
    assert "real earthquake offsets kept" in out and "0.9" in out
    assert "slow-slip episodes kept" in out
    assert "signal kept, validation" not in out and "Ridgecrest" not in out
