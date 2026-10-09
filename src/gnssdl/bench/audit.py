"""Audit filters in common use (docs/08-audit-filters.md).

C1 regional stack and C3 sub-regional PCA (target-inclusive reference
filters), and D1, the mean of stations at least 400 km away (Bachelot et
al. 2025), with its sweep over the distance, fs.

Each filter provides `affected_by(cube, inside)`: the stations whose
cleaned series a transient inside `inside` can change, which the
planted-signal packing needs (contract §5.1).
"""

from __future__ import annotations

import numpy as np

from gnssdl.bench.base import Reconstructor, station_scale
from gnssdl.bench.train import quiet_residuals
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


# C3 settings swept (not tuned: it reads the target, so more components
# always remove more noise; every setting is a point on the trade-off plot)
C3_REGIONS = (1, 4, 8)
C3_COMPONENTS = (1, 2, 3)
C3_MIN_TRAIN_DAYS = 365
C3_FALLBACK_DAYS = 730
PPCA_ITERATIONS = 200
PPCA_TOL = 1e-6


def kmeans_regions(lat: np.ndarray, lon: np.ndarray, k: int, seed: int = 0, restarts: int = 10) -> np.ndarray:
    """Sub-regions: k-means on locally projected coordinates (km), best of
    `restarts` k-means++ starts. Returns a region label per station."""
    lat0 = np.deg2rad(np.mean(lat))
    xy = np.c_[np.deg2rad(lon) * np.cos(lat0), np.deg2rad(lat)] * 6371.0
    if k == 1:
        return np.zeros(len(lat), dtype=int)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        c = xy[[rng.integers(len(xy))]]
        while len(c) < k:                                          # k-means++
            d2 = ((xy[:, None] - c[None]) ** 2).sum(-1).min(1)
            c = np.vstack([c, xy[rng.choice(len(xy), p=d2 / d2.sum())]])
        for _ in range(100):
            lab = ((xy[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([xy[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((xy - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def _outer(W: np.ndarray) -> np.ndarray:
    """n×p² rows of w_i w_iᵀ, flattened, so per-day sums are one matrix product."""
    return (W[:, :, None] * W[:, None, :]).reshape(len(W), -1)


def _scores(W: np.ndarray, s2: float, Y: np.ndarray, A: np.ndarray):
    """pPCA posterior of the daily scores given the available stations:
    mean (T×p) and covariance (T×p×p). Y, A: n×T (Y zero where A is 0)."""
    p = W.shape[1]
    M = (A.T @ _outer(W)).reshape(-1, p, p) + s2 * np.eye(p)
    Minv = np.linalg.inv(M)
    z = np.einsum("tpq,tq->tp", Minv, (A * Y).T @ W)
    return z, s2 * Minv


def fit_ppca(Y: np.ndarray, A: np.ndarray, p: int, iterations: int = PPCA_ITERATIONS, tol: float = PPCA_TOL):
    """Probabilistic PCA with missing values by expectation-maximisation
    (Tipping & Bishop 1999), zero mean (the residuals are already
    detrended). Y, A: n stations × T days. Returns loadings W (n×p) and the
    isotropic noise variance."""
    n, T = Y.shape
    u, sv, _ = np.linalg.svd(A * Y, full_matrices=False)
    W = u[:, :p] * sv[:p] / np.sqrt(max(T, 1))
    s2 = float((A * Y**2).sum() / max(A.sum(), 1)) * 0.1
    prev = np.inf
    for _ in range(iterations):
        z, C = _scores(W, s2, Y, A)
        Ezz = C + np.einsum("tp,tq->tpq", z, z)                     # T×p×p
        num = (A * Y) @ z                                            # n×p
        den = (A @ Ezz.reshape(len(Ezz), -1)).reshape(n, p, p) + 1e-9 * np.eye(p)
        W = np.linalg.solve(den, num[..., None])[..., 0]
        fit = A * (Y - W @ z.T)
        trace = float((A * (_outer(W) @ C.reshape(len(C), -1).T)).sum())
        s2 = float(((fit**2).sum() + trace) / A.sum())
        if abs(prev - s2) < tol * s2:
            break
        prev = s2
    return W, s2


class C3RegionalPCA(Reconstructor):
    """Sub-regional probabilistic PCA (docs/08-audit-filters.md).

    Stations are grouped into `regions` by k-means on their coordinates. In
    each region and per component, pPCA with `components` components is fitted
    by EM on the training-period quiet residuals (normalised per station,
    earthquake windows and the Ridgecrest year left out, missing days latent).
    On any day the component scores are estimated from every available
    station of the region, the target included, and the common mode is the
    reconstruction W·z. Target-inclusive, so a reference filter."""

    name = "c3"
    reference_filter = True
    never_reads_target = False

    def __init__(self, radius_km: float = 0.0, regions: int = 4, components: int = 1, **kw):
        super().__init__(radius_km=radius_km, **kw)
        self.regions, self.components = int(regions), int(components)
        self.event_window: np.ndarray | None = None                # set by the harness
        self.labels = self._W = self._s2 = self._scale = None
        self.fallback_stations: list[str] = []

    def fit(self, cube: Cube, val_hide=None, scorer=None) -> None:
        self.labels = kmeans_regions(cube.lat, cube.lon, self.regions)
        sc = station_scale(cube)
        self._scale = np.where(np.isfinite(sc) & (sc > 0), sc, 1.0)
        q = quiet_residuals(cube).astype(np.float64) / self._scale[:, None, :]
        q = np.nan_to_num(q, nan=0.0)
        cells = cube.avail & ~cube.exclude
        if self.event_window is not None:
            cells &= ~self.event_window
        train = cells & (cube.split_day == 0)[None, :]
        short = train.sum(axis=1) < C3_MIN_TRAIN_DAYS
        self.fallback_stations = [str(s) for s in cube.sta[short]]
        S = len(cube.sta)
        self._W = np.zeros((S, self.components, 3))
        self._s2 = np.zeros((self.regions, 3))
        for g in range(self.regions):
            mem = np.flatnonzero(self.labels == g)
            full, late = mem[~short[mem]], mem[short[mem]]
            for c in range(3):
                tdays = np.flatnonzero(train[full].any(axis=0))
                Y = q[full][:, tdays, c]
                A = train[full][:, tdays].astype(np.float64)
                W, s2 = fit_ppca(Y * A, A, self.components)
                self._W[full, :, c], self._s2[g, c] = W, s2
                if len(late):
                    # stations that started late: regress their first 730
                    # available days on the scores from the full stations
                    z, _ = _scores(W, s2, q[full][:, :, c] * cells[full], cells[full].astype(np.float64))
                    for i in late:
                        days = np.flatnonzero(cells[i])[:C3_FALLBACK_DAYS]
                        zz = z[days]
                        self._W[i, :, c] = np.linalg.lstsq(zz, q[i, days, c], rcond=None)[0]

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("fit first")
        a = (cube.avail & ~hide).astype(np.float64)
        x = np.nan_to_num(cube.r.astype(np.float64) / self._scale[:, None, :], nan=0.0)
        out = np.zeros(cube.r.shape, dtype=np.float32)
        for g in range(self.regions):
            mem = np.flatnonzero(self.labels == g)
            for c in range(3):
                W = self._W[mem, :, c]
                z, _ = _scores(W, self._s2[g, c], x[mem, :, c] * a[mem], a[mem])
                out[mem, :, c] = (W @ z.T) * self._scale[mem, c][:, None]
        return out

    def affected_by(self, cube: Cube, inside: np.ndarray) -> np.ndarray:
        return np.isin(self.labels, np.unique(self.labels[inside]))

    def config(self) -> dict:
        return {**super().config(), "regions": self.regions, "components": self.components,
                "fallback_stations": self.fallback_stations}
