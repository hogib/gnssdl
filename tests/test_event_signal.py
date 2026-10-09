import numpy as np
import pytest

from gnssdl.bench import event_signal as es
from gnssdl.bench import events
from gnssdl.bench.audit import C1Stack
from gnssdl.bench.m0_zero import M0Zero
from gnssdl.bench.m1_stack import M1Stack
from gnssdl.bench.signal import affected
from tests.test_audit import _cube
from tests.test_events import vertical_fault


@pytest.fixture(scope="module")
def cube():
    return _cube(40, 2.0, 2.0)          # 33-35 N, 122-120 W


@pytest.fixture(scope="module")
def runs(cube):
    """Events on a synthetic north-striking fault through the middle of the toy network."""
    fault = vertical_fault(length_km=200.0, depth_km=15.0, step_km=2.0)
    fault.lon = fault.lon - 4.0                                 # move it from 117 W to 121 W
    tp = events.EventTemplate("toy afterslip", ("rlss",), 40.0, 2.0, 12.0, 0.3, "log", 120, tau_days=10.0)
    original = events.random_event
    events.random_event = lambda t, cat, rng, lon, lat: events.place_event(t, fault, rng, lon, lat)
    try:
        keep = np.ones(len(cube.sta), dtype=bool)
        return es.plan_events(cube, keep, [tp], [], per_template=6, radius_km=0.0)
    finally:
        events.random_event = original


def test_every_event_is_seen_and_packing_is_exact(cube, runs):
    assert sum(len(r) for r in runs) == 6
    for run in runs:
        taken = np.zeros(len(cube.sta), dtype=int)
        for inj in run:
            assert inj.seen.sum() >= es.MIN_SEEN
            taken += affected(cube, inj.inside, 0.0).astype(int)
        assert taken.max() <= 1


def test_no_filter_keeps_every_event(cube, runs):
    keep = np.ones(len(cube.sta), dtype=bool)
    model = M0Zero()
    ev_rows, sta_rows = es.event_scores(model, cube, keep, runs)
    assert len(ev_rows) == 6
    np.testing.assert_allclose(ev_rows.rho, 1.0, atol=1e-6)
    np.testing.assert_allclose(sta_rows.rho, 1.0, atol=1e-6)
    assert (ev_rows.spurious_max == 0).all()
    assert set(sta_rows.side) <= {-1, 1} and (sta_rows.dist_km >= 0).all()


def test_neighbour_filters_absorb_part_of_the_event(cube, runs):
    keep = np.ones(len(cube.sta), dtype=bool)
    m1 = M1Stack(radius_km=0.0, length_km=50.0)
    m1.fit(cube)
    rho_m1 = es.event_scores(m1, cube, keep, runs)[0].rho
    c1 = C1Stack()
    c1.fit(cube)
    rho_c1 = es.event_scores(c1, cube, keep, runs)[0].rho
    assert (rho_m1 < 0.98).all() and (rho_c1 < 1.0).all()
    assert rho_m1.median() < rho_c1.median()                   # the local stack absorbs more


def test_event_signal_is_permanent(cube, runs):
    inj = runs[0][0]
    sta, s, days = es.event_signal(cube, inj)
    np.testing.assert_allclose(s[:, -1], inj.event.final_mm()[sta], rtol=1e-5, atol=1e-6)
    assert np.abs(s[:, 0]).max() < 0.05 * np.abs(s[:, -1]).max()


def test_templates_load_from_toml(tmp_path):
    (tmp_path / "a.toml").write_text(
        'name = "slow slip"\nslip_senses = ["rlss", "llss"]\nlength_km = 20.0\ntop_km = 0.0\n'
        'bottom_km = 2.0\nslip_m = 0.02\ntime = "propagating"\nduration_days = 0\n'
        'speed_km_day = 9.0\nrise_days = 3.0\nreference = "placeholder"\n')
    (tp,) = es.load_templates(tmp_path)
    assert tp.slip_senses == ("rlss", "llss") and tp.speed_km_day == 9.0
    (tmp_path / "b.toml").write_text('name = "x"\nslip_sense = ["r"]\n')
    with pytest.raises(ValueError, match="unknown keys"):
        es.load_templates(tmp_path)
