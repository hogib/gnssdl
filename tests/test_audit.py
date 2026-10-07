import numpy as np
import pandas as pd
import pytest

from gnssdl import dataset
from gnssdl.bench.audit import C1Stack, D1FarStack


@pytest.fixture(scope="module")
def wide_cube():
    """Stations spread over ~1,000 km so that some pairs are > 400 km apart,
    sharing a 3 mm common mode plus 1 mm own noise."""
    rng = np.random.default_rng(3)
    days = pd.date_range("2008-01-01", "2023-12-31", freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    common = rng.normal(0, 0.003, (len(days), 3))
    n = 24
    st = pd.DataFrame({"sta": [f"W{i:03d}" for i in range(n)],
                       "lat": 33.0 + rng.uniform(0, 8.0, n), "lon": -122.0 + rng.uniform(0, 6.0, n)})

    def load(sta):
        own = np.random.default_rng(int(sta[1:])).normal(0, 0.001, (len(days), 3))
        x = common + own
        return pd.DataFrame({"decyear": t, "e": x[:, 0], "n": x[:, 1], "u": x[:, 2],
                             "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003}, index=days)

    steps = pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    cube, _ = dataset.build_cube(st, load, steps)
    return cube


def test_c1_is_the_network_mean_including_the_target(wide_cube):
    c1 = C1Stack()
    pred = c1.predict(wide_cube, np.zeros(wide_cube.avail.shape, dtype=bool))
    t = int(np.flatnonzero(wide_cube.avail.all(axis=0))[0])
    np.testing.assert_allclose(pred[0, t], wide_cube.r[:, t].mean(axis=0), atol=1e-5)
    assert c1.reference_filter and not c1.never_reads_target
    assert c1.affected_by(wide_cube, np.zeros(len(wide_cube.sta), dtype=bool)).all()


def test_d1_uses_only_stations_beyond_400km_and_never_the_target(wide_cube):
    d1 = D1FarStack()
    d1.fit(wide_cube)
    d, _ = dataset.distance_azimuth(wide_cube.lat, wide_cube.lon)
    i = int(np.argmax((d >= 400).sum(axis=1)))
    t = int(np.flatnonzero(wide_cube.avail.all(axis=0))[0])
    pred = d1.predict(wide_cube, np.zeros(wide_cube.avail.shape, dtype=bool))
    far = d[i] >= 400
    np.testing.assert_allclose(pred[i, t], wide_cube.r[far, t].mean(axis=0), atol=1e-5)
    # poisoning the target and every station closer than 400 km changes nothing
    r = wide_cube.r.copy()
    r[d[i] < 400] = 1e6
    from dataclasses import replace
    pred2 = d1.predict(replace(wide_cube, r=r), np.zeros(wide_cube.avail.shape, dtype=bool))
    np.testing.assert_allclose(pred2[i], pred[i], atol=1e-3)


def test_d1_affected_stations(wide_cube):
    d1 = D1FarStack()
    d1.fit(wide_cube)
    d, _ = dataset.distance_azimuth(wide_cube.lat, wide_cube.lon)
    inside = np.zeros(len(wide_cube.sta), dtype=bool)
    inside[0] = True
    aff = d1.affected_by(wide_cube, inside)
    np.testing.assert_array_equal(aff, inside | (d[:, 0] >= 400))


def test_d1_removes_common_mode(wide_cube):
    d1 = D1FarStack()
    d1.fit(wide_cube)
    pred = d1.predict(wide_cube, np.zeros(wide_cube.avail.shape, dtype=bool))
    ok = wide_cube.avail & (d1.no_neighbour == False)  # noqa: E712
    resid = (wide_cube.r - pred)[ok]
    assert resid.std() < 0.6 * wide_cube.r[ok].std()
