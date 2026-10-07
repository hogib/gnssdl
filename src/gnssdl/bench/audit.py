"""Audit filters in common use (docs/08-audit-filters.md).

C1 regional stack (target-inclusive reference filter) and D1, the mean of
stations at least 400 km away (Bachelot et al. 2025). C3 follows.

Each filter provides `affected_by(cube, inside)`: the stations whose
cleaned series a transient inside `inside` can change, which the
planted-signal packing needs (contract §5.1).
"""

from __future__ import annotations

import numpy as np

from gnssdl.bench.base import Reconstructor
from gnssdl.dataset import Cube, distance_azimuth

D_DISTANCE_KM = 400.0


class C1Stack(Reconstructor):
    """Regional stack (Wdowinski et al. 1997): every station's common mode is
    the unweighted mean of all available stations that day, itself
    included."""

    name = "c1"
    reference_filter = True
    never_reads_target = False

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        x = np.where(cube.avail[..., None], cube.r, 0.0).astype(np.float64)
        n = cube.avail.sum(axis=0)                                  # T
        mean = np.divide(x.sum(axis=0), n[:, None], out=np.zeros((len(n), 3)), where=n[:, None] > 0)
        return np.broadcast_to(mean[None], cube.r.shape).astype(np.float32).copy()

    def affected_by(self, cube: Cube, inside: np.ndarray) -> np.ndarray:
        return np.ones(len(cube.sta), dtype=bool)


class D1FarStack(Reconstructor):
    """Mean of every available station at least 400 km away, per day and
    component, unweighted, as described by Bachelot et al. (2025). Never
    uses the target, so it is leave-one-out."""

    name = "d1"
    never_reads_target = True

    def __init__(self, radius_km: float = D_DISTANCE_KM, **kw):
        super().__init__(radius_km=radius_km, **kw)
        self._far: np.ndarray | None = None
        self._lat = self._lon = None

    def fit(self, cube: Cube, val_hide=None, scorer=None) -> None:
        self._far_matrix(cube)

    def _far_matrix(self, cube: Cube) -> np.ndarray:
        if self._far is None or not (np.array_equal(self._lat, cube.lat) and np.array_equal(self._lon, cube.lon)):
            d, _ = distance_azimuth(cube.lat, cube.lon)
            far = d >= self.radius_km
            np.fill_diagonal(far, False)          # never the target itself (matters at R = 0)
            self._far = far.astype(np.float32)
            self._lat, self._lon = cube.lat.copy(), cube.lon.copy()
        return self._far

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        far = self._far_matrix(cube)                                  # S×S
        a = cube.avail.astype(np.float32)                             # S×T
        x = np.where(cube.avail[..., None], cube.r, 0.0).astype(np.float32)
        den = far @ a                                                 # S×T
        out = np.zeros(cube.r.shape, dtype=np.float32)
        for c in range(3):
            num = far @ x[..., c]
            out[..., c] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        self.no_neighbour = den <= 0
        return out

    def affected_by(self, cube: Cube, inside: np.ndarray) -> np.ndarray:
        far = self._far_matrix(cube).astype(bool)
        return inside | far[:, inside].any(axis=1)

    def config(self) -> dict:
        return {**super().config(), "distance_km": self.radius_km}


class FarStackSweep(D1FarStack):
    """The far stack at any exclusion radius: the mean of every available
    station at least R away (all other stations at R = 0). D1 is its
    400 km point."""
    name = "fs"
