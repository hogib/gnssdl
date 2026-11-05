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


@pytest.fixture(scope="module")
def cascadia_cube():
    """Stations on the tremor band with a 4 mm step over an episode window."""
    from gnssdl import dataset
    rng = np.random.default_rng(3)
    n = 16
    days = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    st = pd.DataFrame({"sta": [f"C{i:03d}" for i in range(n)],
                       "lat": np.linspace(47.0, 49.0, n), "lon": -123.0 + rng.uniform(-0.3, 0.3, n)})
    step = np.where(days >= pd.Timestamp("2023-04-11"), 0.004, 0.0)   # mid-episode, 4 mm east

    def load(sta):
        r = np.random.default_rng(int(sta[1:]))
        x = r.normal(0, 0.0005, (len(days), 3))
        return pd.DataFrame({"decyear": t, "e": step + x[:, 0], "n": x[:, 1], "u": x[:, 2],
                             "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003}, index=days)

    steps = pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    cube, _ = dataset.build_cube(st, load, steps)
    return cube


def test_episode_retention_is_one_for_m0_and_zero_for_a_copy_filter(cascadia_cube):
    cube = cascadia_cube
    episodes = pd.DataFrame([
        {"id": "T20230401", "source": "test", "timing": "tremor", "start": pd.Timestamp("2023-04-01"),
         "end": pd.Timestamp("2023-04-20"), "lat_min": 47.5, "lat_max": 49.5, "note": ""},
        {"id": "far", "source": "test", "timing": "gnss", "start": pd.Timestamp("2022-02-01"),
         "end": pd.Timestamp("2022-02-20"), "lat_min": 42.0, "lat_max": 44.0, "note": ""},   # no stations
        {"id": "early", "source": "test", "timing": "gnss", "start": pd.Timestamp("2019-12-20"),
         "end": pd.Timestamp("2020-01-05"), "lat_min": 47.0, "lat_max": 49.0, "note": ""},   # window off the cube
    ])
    band = pd.DataFrame({"lat": np.arange(40.125, 50.0, 0.25), "lon": -123.0})
    keep = np.ones(len(cube.sta), dtype=bool)
    df = bsig.episode_retention(M0Zero(), cube, keep, episodes, band)
    assert set(df.episode) == {"T20230401"}
    assert (df.lat >= 47.5).all() and len(df) == (cube.lat >= 47.5).sum()
    assert np.allclose(df.retained, 1.0) and (df.obs_mm > 3.5).all()
    summ = bsig.episode_summary(df)
    assert summ.loc[summ.timing == "tremor", "retained"].item() == pytest.approx(1.0)
    assert summ.loc[summ.timing == "tremor", "episodes"].item() == 1

    class Copy(M0Zero):
        never_reads_target = True

        def predict(self, c, hide):
            return np.nan_to_num(c.r).astype(np.float32)

    assert np.allclose(bsig.episode_retention(Copy(), cube, keep, episodes, band).retained, 0.0, atol=1e-6)


def test_episode_summary_drops_thin_episodes():
    df = pd.DataFrame({"episode": ["a"] * 12 + ["b"] * 3, "timing": "gnss",
                       "retained": [0.9] * 12 + [0.1] * 3})
    summ = bsig.episode_summary(df)
    assert summ.episodes.item() == 1 and summ.retained.item() == pytest.approx(0.9)
