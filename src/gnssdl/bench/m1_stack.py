"""M1: distance-weighted neighbour mean (docs/02-m1-weighted-stack.md).

    r̂_i(t) = Σ_j w_ij a_j(t) r_j(t) / Σ_j w_ij a_j(t),   w_ij = exp(-d²/2L²) / σ̄_j²

over the neighbours beyond the exclusion radius. Same-day only; the target's
own values are never read.
"""

from __future__ import annotations

import math

import numpy as np

from gnssdl.bench.base import Reconstructor, apply_hide, median_sigma
from gnssdl.dataset import Cube

LENGTH_SCALES_KM = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0, math.inf)
CHUNK = 64


class M1Stack(Reconstructor):
    name = "m1"

    def __init__(self, radius_km: float = 0.0, length_km: float | None = None, **kw):
        super().__init__(radius_km, **kw)
        self.length_km = length_km
        self.validation_curve: dict[str, float] = {}
        self._sbar: np.ndarray | None = None

    def fit(self, cube: Cube, val_hide: np.ndarray | None = None, scorer=None) -> None:
        self._sbar = median_sigma(cube)
        if self.length_km is not None:
            return
        if val_hide is None or scorer is None:
            raise ValueError("M1 chooses its length scale on validation cells; pass val_hide and scorer")
        masked = apply_hide(cube, val_hide)
        best = None
        for L in LENGTH_SCALES_KM:
            score = float(scorer(self._predict(masked, L)))
            self.validation_curve[f"{L:g}"] = round(score, 5)
            if best is None or score < best[0]:
                best = (score, L)
        self.length_km = best[1]

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        if self._sbar is None or self.length_km is None:
            raise RuntimeError("fit first")
        return self._predict(cube, self.length_km)

    def _predict(self, cube: Cube, length_km: float) -> np.ndarray:
        ri = cube.radius_index(self.radius_km)
        idx = cube.nbr_idx[ri]                     # S×K, -1 pads
        dist = cube.nbr_dist[ri].astype(np.float64)
        valid = idx >= 0
        nb = np.where(valid, idx, 0)

        kernel = np.ones_like(dist) if math.isinf(length_km) else np.exp(-dist**2 / (2 * length_km**2))
        kernel = np.where(valid, kernel, 0.0)
        w = kernel[:, :, None] / self._sbar[nb] ** 2   # S×K×3
        w = np.nan_to_num(w, nan=0.0)

        x = np.nan_to_num(cube.r, nan=0.0)
        a = cube.avail
        S, T, _ = cube.r.shape
        out = np.zeros((S, T, 3), dtype=np.float32)
        self.no_neighbour = np.zeros((S, T), dtype=bool)   # fell back to 0 (M0)
        for lo in range(0, S, CHUNK):
            hi = min(lo + CHUNK, S)
            ww = w[lo:hi, :, None, :] * a[nb[lo:hi]][..., None]   # c×K×T×3
            num = (ww * x[nb[lo:hi]]).sum(axis=1)
            den = ww.sum(axis=1)
            out[lo:hi] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
            self.no_neighbour[lo:hi] = den[..., 0] <= 0
        return out

    def config(self) -> dict:
        return {**super().config(), "length_km": self.length_km,
                "validation_curve": self.validation_curve}
