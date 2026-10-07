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


class M1Stack(Reconstructor):
    name = "m1"
    never_reads_target = True
    hyperparams = ("length_km",)

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
        k = self.n_neighbours
        idx = cube.nbr_idx[ri][:, :k]              # S×K, -1 pads
        dist = cube.nbr_dist[ri][:, :k].astype(np.float64)
        valid = idx >= 0
        nb = np.where(valid, idx, 0)

        kernel = np.ones_like(dist) if math.isinf(length_km) else np.exp(-dist**2 / (2 * length_km**2))
        kernel = np.where(valid, kernel, 0.0)
        w = kernel[:, :, None] / self._sbar[nb] ** 2   # S×K×3
        w = np.nan_to_num(w, nan=0.0)

        # The weights as a sparse station×station matrix, applied to all days at
        # once by a matrix product: same result as gathering the K neighbours,
        # at a cost that does not grow with K.
        S, T, _ = cube.r.shape
        rows = np.repeat(np.arange(S), idx.shape[1])
        a = cube.avail.astype(np.float64)                          # S×T
        out = np.zeros((S, T, 3), dtype=np.float32)
        no_nb = np.zeros((S, T), dtype=bool)
        for c in range(3):
            W = np.zeros((S, S))
            np.add.at(W, (rows, nb.ravel()), w[:, :, c].ravel())
            x = np.where(cube.avail, np.nan_to_num(cube.r[..., c], nan=0.0), 0.0).astype(np.float64)
            num, den = W @ x, W @ a
            out[..., c] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
            if c == 0:
                no_nb = den <= 0
        self.no_neighbour = no_nb                                  # fell back to 0 (M0)
        return out

    def config(self) -> dict:
        return {**super().config(), "length_km": self.length_km,
                "validation_curve": self.validation_curve}


class M1K64(M1Stack):
    """M1 with the 64 nearest neighbours beyond R instead of 16."""
    name = "m1k64"
    n_neighbours = 64


class M1K256(M1Stack):
    """M1 with the 256 nearest neighbours beyond R."""
    name = "m1k256"
    n_neighbours = 256
