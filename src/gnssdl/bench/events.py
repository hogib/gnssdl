"""Event library, part 1: fault surfaces and the displacement they cause
(contract §5.5).

Fault geometry comes from the SCEC Community Fault Model 7.0 (Plesch et al.
2007; Zenodo 10.5281/zenodo.13685611), whose triangulated surfaces (GOCAD
TSurf) are in UTM zone 11, NAD27, metres, elevation positive up. Station
displacements are computed with triangular dislocations in a homogeneous
elastic half-space (Nikkhoo & Walter 2015, via `cutde`), Poisson's ratio
0.25.

Geometry is handled in a local transverse Mercator frame centred on each
event, so no part of the state is far from its projection's meridian.
The half-space surface is z = 0: fault vertices above sea level are clamped
to it, a simplification of the topography.

Slip convention (Aki & Richards): for every triangle the normal is taken to
point up, into the hanging wall; strike is ẑ × n̂ (the fault dips to the right
of strike) and up-dip is n̂ × strike. Rake 0 is left-lateral, 180
right-lateral, 90 reverse, −90 normal; slip is the motion of the hanging
wall relative to the footwall. For a vertical triangle either normal gives
the same physical motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CFM_CRS = "EPSG:26711"            # NAD27 / UTM zone 11N, as stated in the CFM trace files
POISSON = 0.25


@dataclass
class Fault:
    """A triangulated fault surface in geographic coordinates."""
    name: str
    lon: np.ndarray               # V vertex longitudes
    lat: np.ndarray               # V vertex latitudes
    elev: np.ndarray              # V vertex elevations, m (positive up)
    tri: np.ndarray               # M×3 vertex indices


def read_tsurf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V×3: x, y, z) and triangles (M×3, 0-based) of a GOCAD TSurf
    file. Several TFACE blocks are merged into one surface."""
    ids, xyz, tris = {}, [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ("VRTX", "PVRTX"):
            ids[int(parts[1])] = len(xyz)
            xyz.append([float(v) for v in parts[2:5]])
        elif parts[0] == "ATOM":                      # a vertex that repeats an earlier one
            ids[int(parts[1])] = ids[int(parts[2])]
        elif parts[0] == "TRGL":
            tris.append([ids[int(v)] for v in parts[1:4]])
    if not tris:
        raise ValueError(f"{path}: no triangles")
    return np.asarray(xyz, dtype=np.float64), np.asarray(tris, dtype=np.int64)


@lru_cache(maxsize=None)
def _to_lonlat():
    from pyproj import Transformer
    return Transformer.from_crs(CFM_CRS, "EPSG:4326", always_xy=True)


def load_fault(path: Path) -> Fault:
    """Read one CFM surface and convert its vertices to longitude/latitude."""
    xyz, tri = read_tsurf(path)
    lon, lat = _to_lonlat().transform(xyz[:, 0], xyz[:, 1])
    name = Path(path).stem
    return Fault(name=name, lon=np.asarray(lon), lat=np.asarray(lat), elev=xyz[:, 2], tri=tri)


class LocalFrame:
    """Transverse Mercator centred on (lon0, lat0): x east, y north, metres."""

    def __init__(self, lon0: float, lat0: float):
        from pyproj import Transformer
        proj = f"+proj=tmerc +lon_0={lon0} +lat_0={lat0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    def xy(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)


def triangle_frames(tri_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per triangle (M×3×3 corners): unit normal pointing up (into the
    hanging wall), strike and up-dip vectors (each M×3), and the corner
    order that makes the normal point up (M×3×3)."""
    n = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    flip = n[:, 2] < 0
    tri_xyz = tri_xyz.copy()
    tri_xyz[flip] = tri_xyz[flip][:, [0, 2, 1]]
    n[flip] *= -1
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ez = np.array([0.0, 0.0, 1.0])
    strike = np.cross(ez, n)
    flat = np.linalg.norm(strike, axis=1) < 1e-9                 # horizontal triangle: any strike
    strike[flat] = [1.0, 0.0, 0.0]
    strike /= np.linalg.norm(strike, axis=1, keepdims=True)
    updip = np.cross(n, strike)
    return n, strike, updip, tri_xyz


def displacement(obs_xyz: np.ndarray, tri_xyz: np.ndarray, slip_m: np.ndarray,
                 rake_deg: float | np.ndarray) -> np.ndarray:
    """East, north, up displacement (m) at `obs_xyz` (N×3, local metres, z ≤ 0)
    from slip `slip_m` (M, metres) with rake `rake_deg` on triangles
    `tri_xyz` (M×3×3, local metres, z ≤ 0)."""
    import cutde.halfspace as hs
    _, strike, updip, tri_xyz = triangle_frames(np.asarray(tri_xyz, dtype=np.float64))
    rake = np.deg2rad(np.broadcast_to(rake_deg, (len(tri_xyz),)))
    slip_vec = (np.cos(rake)[:, None] * strike + np.sin(rake)[:, None] * updip) * np.asarray(slip_m)[:, None]
    # cutde wants slip in each triangle's own (strike, dip, tensile) frame,
    # built from the corner order exactly as `triangle_frames` builds it
    comp = np.stack([np.einsum("mk,mk->m", slip_vec, strike),
                     np.einsum("mk,mk->m", slip_vec, updip),        # cutde's dip axis points up-dip (tested)
                     np.zeros(len(tri_xyz))], axis=1)
    mat = hs.disp_matrix(np.asarray(obs_xyz, dtype=np.float64), tri_xyz, POISSON)   # N×3×M×3
    return np.einsum("nimj,mj->ni", mat, comp)


def fault_triangles(fault: Fault, frame: LocalFrame, select: np.ndarray | None = None) -> np.ndarray:
    """The fault's triangles (or those in `select`) as M×3×3 local
    coordinates, with vertices above sea level clamped to the surface."""
    x, y = frame.xy(fault.lon, fault.lat)
    z = np.minimum(fault.elev, 0.0)
    v = np.stack([x, y, z], axis=1)
    tri = fault.tri if select is None else fault.tri[select]
    return v[tri]


def station_displacement(fault: Fault, slip_m: np.ndarray, rake_deg, sta_lon, sta_lat,
                         select: np.ndarray | None = None) -> np.ndarray:
    """E, N, U displacement (m) at stations (lon, lat) from slip on
    `fault` (one value per selected triangle), in a frame centred on the
    slipping triangles."""
    tri_idx = np.arange(len(fault.tri)) if select is None else np.flatnonzero(select) if \
        np.asarray(select).dtype == bool else np.asarray(select)
    used = np.unique(fault.tri[tri_idx])
    frame = LocalFrame(float(np.mean(fault.lon[used])), float(np.mean(fault.lat[used])))
    tris = fault_triangles(fault, frame, tri_idx)
    sx, sy = frame.xy(sta_lon, sta_lat)
    obs = np.stack([sx, sy, np.zeros_like(sx)], axis=1)
    return displacement(obs, tris, slip_m, rake_deg)
