"""M3: ridge regression on neighbours (docs/04-m3-ridge.md).

For target i, day t, the features are its K = 16 nearest neighbours beyond
the exclusion radius: all three components of each (zero where missing),
an availability bit per neighbour, and an intercept (65 features). M3b, the
pooled variant shared by all stations, adds each slot's geometry (log
distance, sine and cosine of azimuth) and its products with the slot's
three residuals (257 features). All residuals are divided by the station's
robust noise scale; predictions are scaled back.

- M3a: one weight vector per station and component; stations without
  enough training rows (and held-out stations) use M3b.
- M3b: one weight vector per target component for all stations.

Missing neighbours are zero, and each row's neighbour values are rescaled by
K / (number available), as in a normalised average, so that dropping or
missing neighbours does not bias a linear fit (plain zero-filling made M3a
over- and M3b under-predict by 5-15% on a toy network).

Training uses quiet residuals on training cells (contract §1.5). Each row is
replicated 4× with neighbours dropped at random (p ~ U(0, 0.5)) so the
weights cope with any gap pattern. λ is chosen on the harness's validation
score from a log grid, in units of trace(Gram)/features. The target's own
values never enter its features, so one pass on the unhidden cube is a
leave-one-out prediction (`never_reads_target`).

Deviation from the doc: rows are weighted equally (in noise-scale units, as
M3k and M5), not by 1/σ².
"""

from __future__ import annotations

import numpy as np

from gnssdl.bench.base import Reconstructor, station_scale
from gnssdl.bench.train import quiet_residuals
from gnssdl.dataset import Cube

K = 16
LAMBDAS = tuple(10.0 ** np.arange(-4, 4.01, 1.0))
REPLICAS = 4
MAX_DROP = 0.5
MIN_ROWS = 365
INTERCEPT = 3 * K + K                                            # feature index of the intercept


def _scale(cube: Cube) -> np.ndarray:
    s = station_scale(cube)
    return np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s, axis=0))


def _geometry(cube: Cube, ri: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = cube.nbr_idx[ri][:, :K].astype(np.int64)
    valid = idx >= 0
    d = np.where(valid, cube.nbr_dist[ri][:, :K], 1.0).astype(np.float64)
    az = np.where(valid, cube.nbr_az[ri][:, :K], 0.0).astype(np.float64)
    geo = np.stack([np.log(np.maximum(d, 1.0)) / 5.0, np.sin(az), np.cos(az)], -1) * valid[..., None]
    return idx, valid, geo                                       # S×K, S×K, S×K×3


def features(z: np.ndarray, a: np.ndarray, geo: np.ndarray | None) -> np.ndarray:
    """Rows of features from neighbour values `z` (n×K×3, zero where
    missing), availability `a` (n×K) and, for the pooled model, slot
    geometry `geo` (K×3, or n×K×3)."""
    n = len(z)
    # rescale by K / (neighbours available): equal weights 1/K then give the
    # plain mean whatever is missing, so the weights need not compensate for
    # gaps (zero-filling alone biased the fit towards the average gap count)
    z = z * (K / np.maximum(a.sum(axis=1), 1.0))[:, None, None]
    parts = [z.reshape(n, -1), a, np.ones((n, 1))]
    if geo is not None:
        g = np.broadcast_to(geo, (n, K, 3)) * a[..., None]
        parts += [g.reshape(n, -1), (g[:, :, :, None] * z[:, :, None, :]).reshape(n, -1)]
    return np.concatenate(parts, axis=1)


def _neighbour_rows(Z, A, idx_i, valid_i, days):
    """z (n×K×3) and a (n×K) for one target's neighbours on `days`."""
    nb = np.where(valid_i, idx_i, 0)
    a = A[nb][:, days].T * valid_i[None, :]
    z = np.transpose(Z[nb][:, days], (1, 0, 2)) * a[..., None]
    return z, a.astype(np.float64)


def _drop(z, a, rng):
    """REPLICAS copies of the rows, each neighbour dropped with a row's own
    probability p ~ U(0, MAX_DROP)."""
    n = len(z)
    zz = np.repeat(z, REPLICAS, axis=0)
    aa = np.repeat(a, REPLICAS, axis=0)
    p = rng.uniform(0, MAX_DROP, len(zz))[:, None]
    keep = rng.random(aa.shape) >= p
    return zz * keep[..., None], aa * keep


def _solve(G: np.ndarray, b: np.ndarray, lam_rel: float) -> np.ndarray:
    p = G.shape[0]
    reg = lam_rel * np.trace(G) / p * np.eye(p)
    reg[INTERCEPT, INTERCEPT] = 0.0                              # the intercept is not penalised
    return np.linalg.solve(G + reg + 1e-9 * np.eye(p), b)


class M3Ridge(Reconstructor):
    """M3a by default; `pooled = True` gives M3b."""

    name = "m3a"
    never_reads_target = True
    hyperparams = ("lam",)
    pooled = False

    def __init__(self, radius_km: float = 0.0, lam: float | None = None, seed: int = 0, **kw):
        super().__init__(radius_km, **kw)
        self.lam = lam
        self.seed = seed
        self.event_window: np.ndarray | None = None      # set by the harness
        self.validation_curve: dict[str, float] = {}
        self.fallback_stations: list[str] = []

    # one Gram matrix per station (M3a) and one pooled (M3b)
    def fit(self, cube: Cube, val_hide=None, scorer=None) -> None:
        rng = np.random.default_rng(self.seed)
        ri = cube.radius_index(self.radius_km)
        self._idx, self._valid, self._geo = _geometry(cube, ri)
        self._scale = _scale(cube)
        cells = cube.avail & ~cube.exclude & (cube.split_day == 0)[None, :] & (cube.split_sta == 0)[:, None]
        if self.event_window is not None:
            cells &= ~self.event_window
        Z = np.where(cells[..., None], np.nan_to_num(quiet_residuals(cube), nan=0.0), 0.0) / self._scale[:, None, :]
        S = len(cube.sta)
        p_a, p_b = 3 * K + K + 1, 3 * K + K + 1 + 3 * K + 9 * K
        Ga = np.zeros((S, p_a, p_a))
        ba = np.zeros((S, p_a, 3))
        rows = np.zeros(S, dtype=int)
        Gb = np.zeros((p_b, p_b))
        bb = np.zeros((p_b, 3))
        for i in range(S):
            days = np.flatnonzero(cells[i])
            if not len(days):
                continue
            z, a = _neighbour_rows(Z, cells, self._idx[i], self._valid[i], days)
            z, a = _drop(z, a, rng)
            y = np.repeat(Z[i, days], REPLICAS, axis=0)
            Xa = features(z, a, None)
            Ga[i], ba[i] = Xa.T @ Xa, Xa.T @ y
            rows[i] = len(days)
            Xb = features(z, a, self._geo[i])
            Gb += Xb.T @ Xb
            bb += Xb.T @ y
        self._G, self._rows = (Ga, ba, Gb, bb), rows
        self._own = rows >= MIN_ROWS
        self.fallback_stations = [str(s) for s in cube.sta[~self._own]]
        if self.lam is not None:
            self._set_lambda(self.lam)
            return
        if scorer is None:
            raise ValueError("M3 chooses λ on validation cells; pass a scorer")
        best = None
        for lam in LAMBDAS:
            self._set_lambda(lam)
            score = float(scorer(self.predict(cube, np.zeros(cube.avail.shape, dtype=bool), split=1)))
            self.validation_curve[f"{lam:g}"] = round(score, 5)
            if best is None or score < best[0]:
                best = (score, lam)
        self.lam = best[1]
        self._set_lambda(self.lam)

    def _set_lambda(self, lam: float) -> None:
        Ga, ba, Gb, bb = self._G
        self._beta_b = _solve(Gb, bb, lam)
        self._beta_a = np.zeros_like(ba)
        if not self.pooled:
            for i in np.flatnonzero(self._own):
                self._beta_a[i] = _solve(Ga[i], ba[i], lam)

    def predict(self, cube: Cube, hide: np.ndarray, split: int | None = None) -> np.ndarray:
        S, T = cube.avail.shape
        out = np.zeros((S, T, 3), dtype=np.float32)
        days = np.arange(T) if split is None else np.flatnonzero(cube.split_day == split)
        A = cube.avail & ~hide
        Z = np.where(A[..., None], np.nan_to_num(cube.r, nan=0.0), 0.0) / self._scale[:, None, :]
        for i in range(S):
            z, a = _neighbour_rows(Z, A, self._idx[i], self._valid[i], days)
            if self.pooled or not self._own[i]:
                y = features(z, a, self._geo[i]) @ self._beta_b
            else:
                y = features(z, a, None) @ self._beta_a[i]
            out[i, days] = y * self._scale[i]
        self.no_neighbour = np.zeros((S, T), dtype=bool)
        return out

    def config(self) -> dict:
        return {**super().config(), "lam": self.lam, "validation_curve": self.validation_curve,
                "pooled": self.pooled, "fallback_stations": len(self.fallback_stations)}


class M3Pooled(M3Ridge):
    """M3b: one set of weights for every station, using slot geometry."""
    name = "m3b"
    pooled = True
