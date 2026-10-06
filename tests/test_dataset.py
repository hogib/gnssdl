import numpy as np
import pandas as pd
import pytest

from gnssdl import dataset


def _series(start="2008-01-01", end="2024-12-31", seed=0, noise_mm=1.0):
    """Synthetic station in tenv3 layout: 10 mm/yr east, 3 mm annual north."""
    days = pd.date_range(start, end, freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_mm / 1000, (len(days), 3))
    df = pd.DataFrame(
        {
            "decyear": t,
            "e": 0.010 * (t - 2008) + noise[:, 0],
            "n": 0.003 * np.sin(2 * np.pi * t) + noise[:, 1],
            "u": noise[:, 2],
            "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003,
        },
        index=days,
    )
    return df


def _no_steps():
    return pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])


def _steps(rows):
    return pd.DataFrame(
        [(s, pd.Timestamp(d), k, "", None, None, None) for s, d, k in rows],
        columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"],
    )


def test_baseline_ignores_validation_and_test_data():
    a = _series()
    b = a.copy()
    b.loc["2022-01-01":, "e"] += 0.5  # 500 mm, only in the test period
    ra = dataset.process_station(a, _no_steps(), None).r
    rb = dataset.process_station(b, _no_steps(), None).r
    train = ra.index <= dataset.TRAIN_END
    np.testing.assert_allclose(ra[train].to_numpy(), rb[train].to_numpy(), atol=1e-9)
    assert (rb.loc["2022":, "e"] - ra.loc["2022":, "e"]).median() == pytest.approx(500, abs=1e-6)


def test_equipment_step_removed_quake_step_kept():
    s = _series(noise_mm=0.3)
    s.loc["2012-06-01":, "e"] += 0.008   # antenna change: 8 mm
    s.loc["2016-03-01":, "n"] += 0.006   # earthquake: 6 mm
    steps = _steps([("X", "2012-06-01", "equipment"), ("X", "2016-03-01", "quake")])
    r = dataset.process_station(s, steps, None).r
    e_jump = r.loc["2012-06-01":"2012-08-01", "e"].mean() - r.loc["2012-04-01":"2012-05-31", "e"].mean()
    n_jump = r.loc["2016-03-01":"2016-05-01", "n"].mean() - r.loc["2016-01-01":"2016-02-29", "n"].mean()
    assert abs(e_jump) < 0.5
    assert n_jump == pytest.approx(6.0, abs=0.5)


def test_equipment_step_after_training_removed_locally():
    s = _series(noise_mm=0.3)
    s.loc["2023-05-01":, "u"] += 0.020
    steps = _steps([("X", "2023-05-01", "equipment")])
    r = dataset.process_station(s, steps, None).r
    jump = r.loc["2023-05-01":"2023-06-15", "u"].median() - r.loc["2023-03-15":"2023-04-30", "u"].median()
    assert abs(jump) < 1.0


def test_spike_is_screened():
    s = _series()
    s.loc["2015-07-04", "e"] += 0.050
    res = dataset.process_station(s, _no_steps(), None)
    assert pd.Timestamp("2015-07-04") not in res.r.index
    assert res.n_screened >= 1


def test_short_training_record_uses_fallback():
    res = dataset.process_station(_series(start="2019-06-01"), _no_steps(), None)
    assert res.fallback_fit


def test_exclusion_window_left_out_of_fit():
    a = _series()
    b = a.copy()
    b.loc["2019-07-01":"2019-12-31", "e"] += 0.3
    window = dataset.RIDGECREST_WINDOW
    ra = dataset.process_station(a, _no_steps(), window).r
    rb = dataset.process_station(b, _no_steps(), window).r
    np.testing.assert_allclose(ra.loc[:"2019-06-30"].to_numpy(), rb.loc[:"2019-06-30"].to_numpy(), atol=1e-9)


def _network(n=30, seed=1):
    rng = np.random.default_rng(seed)
    lat = 35.0 + rng.uniform(0, 2, n)
    lon = -118.0 + rng.uniform(0, 2, n)
    return pd.DataFrame({"sta": [f"S{i:03d}" for i in range(n)], "lat": lat, "lon": lon})


def test_neighbour_graphs_respect_exclusion_radius():
    st = _network()
    d, az = dataset.distance_azimuth(st.lat.to_numpy(), st.lon.to_numpy())
    idx, dist, _ = dataset.neighbour_graphs(d, az)
    for ri, rad in enumerate(dataset.RADII_KM):
        ok = idx[ri] >= 0
        assert (dist[ri][ok] > rad).all()
        assert (dist[ri][ok] <= dataset.MAX_NEIGHBOUR_KM).all()
        assert not (idx[ri] == np.arange(len(st))[:, None]).any()  # never itself
        d_sorted = np.where(ok, dist[ri], 1e9)
        assert (np.diff(d_sorted, axis=1) >= 0).all()  # nearest first, pads last


def test_heldout_selection_is_deterministic_and_spread():
    st = _network(n=200)
    d, _ = dataset.distance_azimuth(st.lat.to_numpy(), st.lon.to_numpy())
    a = dataset.farthest_point_sample(d, 20)
    b = dataset.farthest_point_sample(d, 20)
    np.testing.assert_array_equal(a, b)
    sub = d[np.ix_(a, a)] + np.eye(len(a)) * 1e9
    rng = np.random.default_rng(5)
    rand = rng.choice(len(st), 20, replace=False)
    sub_rand = d[np.ix_(rand, rand)] + np.eye(20) * 1e9
    assert sub.min() > sub_rand.min()  # spread out, unlike a random pick


def test_build_cube_end_to_end():
    st = _network(n=12)
    st.loc[0, ["lat", "lon"]] = dataset.RIDGECREST_LATLON  # inside the Ridgecrest zone
    st.loc[1, ["lat", "lon"]] = (40.0, -122.0)              # far outside it
    late = {"S005"}
    cube, summary = dataset.build_cube(
        st,
        lambda sta: _series(start="2020-03-01" if sta in late else "2008-01-01",
                            seed=int(sta[1:])),
        _no_steps(),
    )
    S, T = len(cube.sta), len(cube.days)
    assert cube.r.shape == (S, T, 3) and cube.nbr_idx.shape == (5, S, 16)
    assert cube.split_day[cube.days <= np.datetime64("2019-12-31")].max() == 0
    assert set(cube.split_day[cube.days > np.datetime64("2021-12-31")]) == {2}
    i_late = int(np.flatnonzero(cube.sta == "S005")[0])
    assert cube.split_sta[i_late] == 1  # short training record forced out
    rc = cube.exclude.any(axis=1)
    assert rc[np.flatnonzero(cube.sta == "S000")[0]] and not rc[np.flatnonzero(cube.sta == "S001")[0]]
    assert cube.avail.sum() == summary["station_days"]


def test_cube_save_load_roundtrip(tmp_path):
    st = _network(n=6)
    cube, _ = dataset.build_cube(st, lambda sta: _series(seed=int(sta[1:])), _no_steps())
    cube.save(tmp_path / "c.npz")
    back = dataset.Cube.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(back.r, cube.r)
    np.testing.assert_array_equal(back.sta, cube.sta)
    assert back.days.dtype == np.dtype("datetime64[D]")
    assert back.radius_index(25) == 2


def test_unstable_site_fails_qc_but_earthquake_offset_does_not():
    wobbly = _series()
    t = wobbly["decyear"].to_numpy()
    wobbly["e"] += 0.200 * np.sin(2 * np.pi * (t - 2008) / 3.1)  # ±200 mm, non-seasonal
    assert dataset.process_station(wobbly, _no_steps(), None).qc_scale_mm > dataset.QC_MAX_HORIZONTAL_SCALE_MM

    quake = _series()
    quake.loc["2014-08-24":, "n"] += 0.300                       # 300 mm coseismic
    steps = _steps([("X", "2014-08-24", "quake")])
    res = dataset.process_station(quake, steps, None)
    assert res.qc_scale_mm < 3.0
    assert res.r.loc["2015", "n"].median() - res.r.loc["2013", "n"].median() > 250  # still kept


def test_sparse_early_record_recovered_by_fallback():
    s = _series(start="2017-01-01")
    early = (s.index < "2021-06-01") & (np.arange(len(s)) % 10 != 0)
    s = s[~early]  # only every 10th day before mid-2021
    res = dataset.process_station(s, _no_steps(), None)
    assert res is not None and res.fallback_fit


def test_twins_keep_the_longest_record():
    lat = np.array([35.0, 35.0001, 35.5, 36.0])
    lon = np.array([-118.0, -118.0001, -118.0, -118.0])
    d, _ = dataset.distance_azimuth(lat, lon)
    sta = np.array(["AAA1", "AAA2", "BBBB", "CCCC"])
    assert dataset.twin_drops(d, sta, np.array([100, 500, 300, 300])) == {0: 1}
    assert dataset.twin_drops(d, sta, np.array([500, 500, 300, 300])) == {1: 0}  # tie → by id


def test_subsidence_fails_vertical_qc():
    s = _series()
    t = s["decyear"].to_numpy()
    s["u"] -= 0.150 * np.clip(t - 2012, 0, None) ** 1.5   # accelerating subsidence
    res = dataset.process_station(s, _no_steps(), None)
    assert res.qc_scale_mm < dataset.QC_MAX_HORIZONTAL_SCALE_MM
    assert res.qc_scale_u_mm > dataset.QC_MAX_VERTICAL_SCALE_MM
