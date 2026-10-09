from pathlib import Path

import numpy as np
import pytest

from gnssdl.bench import events

KM = 1000.0


def rect(x0, y0, x1, y1, z_top, z_bot, dx_bottom=0.0):
    """Two triangles of a rectangle from (x0, y0) to (x1, y1) at the top,
    shifted by dx_bottom (east) at the bottom; metres."""
    a, b = [x0, y0, z_top], [x1, y1, z_top]
    c, d = [x1 + dx_bottom, y1, z_bot], [x0 + dx_bottom, y0, z_bot]
    return np.array([[a, b, c], [a, c, d]], dtype=float)


def test_right_lateral_moves_east_side_south():
    tri = rect(0, -50 * KM, 0, 50 * KM, 0, -10 * KM)            # vertical, striking north
    obs = np.array([[5 * KM, 0, 0], [-5 * KM, 0, 0]])
    u = events.displacement(obs, tri, np.ones(2), 180.0)
    assert u[0, 1] < -0.1 and u[1, 1] > 0.1                     # east side south, west side north
    np.testing.assert_allclose(u[0, 1], -u[1, 1], rtol=1e-6)
    assert abs(u[0, 0]) < 1e-3 and abs(u[0, 2]) < 1e-3


def test_long_strike_slip_fault_matches_the_screw_dislocation():
    # slip s from the surface to depth D on a very long vertical fault:
    # u(x) = (s / π) arctan(D / x) at the surface (half-space, 2-D limit)
    D = 10 * KM
    tri = rect(0, -2000 * KM, 0, 2000 * KM, 0, -D)
    xs = np.array([2, 5, 10, 30]) * KM
    obs = np.stack([xs, np.zeros(4), np.zeros(4)], axis=1)
    u = events.displacement(obs, tri, np.ones(2), 0.0)           # left-lateral: east side north
    np.testing.assert_allclose(u[:, 1], np.arctan(D / xs) / np.pi, rtol=1e-3)


def test_reverse_slip_lifts_the_hanging_wall():
    # dips east at 45°: the hanging wall is to the east
    tri = rect(0, -30 * KM, 0, 30 * KM, -1 * KM, -11 * KM, dx_bottom=10 * KM)
    obs = np.array([[3 * KM, 0, 0], [-10 * KM, 0, 0]])
    u = events.displacement(obs, tri, np.ones(2), 90.0)
    assert u[0, 2] > 0.05                                       # uplift above the hanging wall
    assert u[0, 0] < 0                                          # hanging wall moves up-dip, i.e. west
    u_normal = events.displacement(obs, tri, np.ones(2), -90.0)
    np.testing.assert_allclose(u_normal, -u, atol=1e-9)


def test_corner_order_does_not_matter():
    tri = rect(0, -20 * KM, 3 * KM, 20 * KM, 0, -8 * KM, dx_bottom=4 * KM)
    obs = np.array([[6 * KM, 2 * KM, 0], [-7 * KM, -3 * KM, 0]])
    a = events.displacement(obs, tri, np.array([1.0, 0.5]), 150.0)
    b = events.displacement(obs, tri[:, [0, 2, 1]], np.array([1.0, 0.5]), 150.0)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_read_tsurf_merges_blocks_and_atoms(tmp_path):
    p = tmp_path / "f.ts"
    p.write_text("GOCAD TSurf 1\nHEADER {\nname:x\n}\nTFACE\n"
                 "VRTX 1 0 0 0\nVRTX 2 1000 0 0\nVRTX 3 0 0 -1000\nTRGL 1 2 3\n"
                 "TFACE\nATOM 4 2\nPVRTX 5 1000 0 -1000 7\nTRGL 4 5 3\nEND\n")
    xyz, tri = events.read_tsurf(p)
    assert xyz.shape == (4, 3)
    np.testing.assert_array_equal(tri, [[0, 1, 2], [1, 3, 2]])


CFM = Path("data/cfm/extracted/obj/preferred/1000m")


@pytest.mark.skipif(not CFM.exists(), reason="SCEC CFM not downloaded")
def test_cfm_fault_lands_in_california_and_slips():
    path = next(CFM.glob("*Superstition_Hills*"))
    fault = events.load_fault(path)
    assert -117 < fault.lon.mean() < -115 and 32 < fault.lat.mean() < 34
    # 10 cm right-lateral on the whole surface: stations 5 km either side move opposite ways
    lon0, lat0 = fault.lon.mean(), fault.lat.mean()
    u = events.station_displacement(fault, np.full(len(fault.tri), 0.1), 180.0,
                                    [lon0 + 0.06, lon0 - 0.06], [lat0 + 0.03, lat0 - 0.03])
    assert np.sign(u[0, 0]) != np.sign(u[1, 0]) or np.sign(u[0, 1]) != np.sign(u[1, 1])
    assert np.abs(u).max() < 0.1
