"""M4: band-split ridge regression (docs/05-m4-band-ridge.md).

Every station's series is split into four bands that add up exactly to it,
with differences of Gaussian smoothers (half-power periods 7, 60 and 400
days):

  b1 < 7 days,  b2 7-60 days,  b3 60-400 days,  b4 > 400 days

and an M3-style ridge regression predicts the target's band from its
neighbours' same band; the prediction is the sum of the four. M4a has one
weight vector per station, band and component (M4b for stations without
enough training rows); M4b is pooled with slot geometry, as M3b.

Gaps must be filled before filtering. The design doc fills them with M3b's
prediction, but M3b's prediction of a neighbour can use the neighbour's own
neighbours, the target among them, which would let the target's values into
its own prediction. Each series is therefore filled by linear
interpolation of its own available days, which reads no other station. On
days a neighbour has no data its features are zero (with its availability
bit off), as in M3, so interpolated values are only used through the
filters.

Two-sided: the filters read neighbours' past and future days. Own-history
is off: the target's series is only the regression target during fitting.
λ is chosen per band, on the validation error of that band's prediction.
"""

from __future__ import annotations

import numpy as np

from gnssdl.bench.base import Reconstructor
from gnssdl.bench.m3_ridge import (
    INTERCEPT, K, LAMBDAS, MIN_ROWS, REPLICAS, MAX_DROP, _geometry, _neighbour_rows, _scale, _solve, features,
)
from gnssdl.bench.train import quiet_residuals
from gnssdl.dataset import Cube

CUTOFFS_DAYS = (7.0, 60.0, 400.0)
N_BANDS = len(CUTOFFS_DAYS) + 1
PAD_DAYS = 1024


def fill_own(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """S×T×3 series with each station's gaps filled by linear interpolation
    of its own available days (constant beyond its first and last day);
    zero for stations with no data."""
    out = np.zeros_like(z, dtype=np.float64)
    t = np.arange(z.shape[1])
    for i in range(z.shape[0]):
        d = np.flatnonzero(a[i])
        if not len(d):
            continue
        for c in range(3):
            out[i, :, c] = np.interp(t, d, z[i, d, c])
    return out


CHUNK = 128


def gaussian_smooth(x: np.ndarray, sigma_days: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 1 (FFT, edge padding),
    in chunks of stations to bound memory."""
    n = int(2 ** np.ceil(np.log2(x.shape[1] + 2 * PAD_DAYS)))
    f = np.fft.rfftfreq(n)                                          # cycles per day
    h = np.exp(-2.0 * np.pi**2 * sigma_days**2 * f**2)[None, :, None]
    out = np.empty(x.shape, dtype=np.float32)
    for lo in range(0, x.shape[0], CHUNK):
        xp = np.pad(x[lo:lo + CHUNK], ((0, 0), (PAD_DAYS, PAD_DAYS), (0, 0)), mode="edge")
        y = np.fft.irfft(np.fft.rfft(xp, n=n, axis=1) * h, n=n, axis=1)
        out[lo:lo + CHUNK] = y[:, PAD_DAYS:PAD_DAYS + x.shape[1]]
    return out


def sigma_for(period_days: float) -> float:
    """Gaussian σ whose response falls to half power at `period_days`."""
    return period_days * np.sqrt(np.log(2)) / (2 * np.pi)


def bands(x: np.ndarray) -> list[np.ndarray]:
    """Four bands, shortest first, that sum exactly to `x`."""
    smooth = [x.astype(np.float32)] + [gaussian_smooth(x, sigma_for(p)) for p in CUTOFFS_DAYS]
    return [smooth[k] - smooth[k + 1] for k in range(len(CUTOFFS_DAYS))] + [smooth[-1]]


class M4Band(Reconstructor):
    name = "m4a"
    never_reads_target = True
    supported_contexts = ("two-sided",)
    hyperparams = ()
    pooled = False

    def __init__(self, radius_km: float = 0.0, seed: int = 0, **kw):
        kw.setdefault("context", "two-sided")
        super().__init__(radius_km, **kw)
        self.seed = seed
        self.event_window: np.ndarray | None = None      # set by the harness
        self.lams: list[float] = []
        self.validation_curves: list[dict] = []

    def _bands_of(self, r: np.ndarray, a: np.ndarray) -> list[np.ndarray]:
        z = np.where(a[..., None], np.nan_to_num(r, nan=0.0), 0.0) / self._scale[:, None, :]
        return bands(fill_own(z, a))

    def fit(self, cube: Cube, val_hide=None, scorer=None) -> None:
        rng = np.random.default_rng(self.seed)
        ri = cube.radius_index(self.radius_km)
        self._idx, self._valid, self._geo = _geometry(cube, ri)
        self._scale = _scale(cube)
        cells = cube.avail & ~cube.exclude & (cube.split_day == 0)[None, :] & (cube.split_sta == 0)[:, None]
        if self.event_window is not None:
            cells &= ~self.event_window
        Bq = self._bands_of(quiet_residuals(cube), cells)
        S = len(cube.sta)
        p_a, p_b = INTERCEPT + 1, INTERCEPT + 1 + 3 * K + 9 * K
        Ga = np.zeros((N_BANDS, S, p_a, p_a))
        ba = np.zeros((N_BANDS, S, p_a, 3))
        Gb = np.zeros((N_BANDS, p_b, p_b))
        bb = np.zeros((N_BANDS, p_b, 3))
        rows = np.zeros(S, dtype=int)
        for i in range(S):
            days = np.flatnonzero(cells[i])
            if not len(days):
                continue
            rows[i] = len(days)
            nb = np.where(self._valid[i], self._idx[i], 0)
            a = (cells[nb][:, days].T * self._valid[i][None, :]).astype(np.float64)
            a = np.repeat(a, REPLICAS, axis=0)
            p = rng.uniform(0, MAX_DROP, len(a))[:, None]
            a = a * (rng.random(a.shape) >= p)                       # the same drops for every band
            for b in range(N_BANDS):
                z = np.repeat(np.transpose(Bq[b][nb][:, days], (1, 0, 2)), REPLICAS, axis=0) * a[..., None]
                y = np.repeat(Bq[b][i, days], REPLICAS, axis=0)
                Xa = features(z, a, None)
                Ga[b, i] += Xa.T @ Xa
                ba[b, i] += Xa.T @ y
                Xb = features(z, a, self._geo[i])
                Gb[b] += Xb.T @ Xb
                bb[b] += Xb.T @ y
        self._own = rows >= MIN_ROWS
        # λ per band: validation error of that band's prediction (leave-one-out,
        # every validation day of seen stations)
        val = cube.avail & (cube.split_day == 1)[None, :] & (cube.split_sta == 0)[:, None]
        Bv = self._bands_of(cube.r, cube.avail)
        vdays = np.flatnonzero(cube.split_day == 1)
        self._beta_a = np.zeros((N_BANDS, S, p_a, 3))
        self._beta_b = np.zeros((N_BANDS, p_b, 3))
        for b in range(N_BANDS):
            curve, best = {}, None
            for lam in LAMBDAS:
                beta_b = _solve(Gb[b], bb[b], lam)
                beta_a = None if self.pooled else np.stack(
                    [_solve(Ga[b, i], ba[b, i], lam) if self._own[i] else np.zeros((p_a, 3)) for i in range(S)])
                err, n = 0.0, 0
                for i in np.flatnonzero(val[:, vdays].any(axis=1)):
                    pred = self._band_pred(Bv[b], cube.avail, i, vdays, beta_a, beta_b)
                    m = val[i, vdays]
                    err += float(((pred[m] - Bv[b][i, vdays][m]) ** 2).sum())
                    n += int(m.sum())
                curve[f"{lam:g}"] = round(err / max(n, 1), 6)
                if best is None or curve[f"{lam:g}"] < best[0]:
                    best = (curve[f"{lam:g}"], lam, beta_a, beta_b)
            self.validation_curves.append(curve)
            self.lams.append(best[1])
            if not self.pooled:
                self._beta_a[b] = best[2]
            self._beta_b[b] = best[3]

    def _band_pred(self, Bb, A, i, days, beta_a, beta_b) -> np.ndarray:
        z, a = _neighbour_rows(Bb, A, self._idx[i], self._valid[i], days)
        if self.pooled or not self._own[i]:
            return features(z, a, self._geo[i]) @ beta_b
        return features(z, a, None) @ beta_a[i]

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        S, T = cube.avail.shape
        A = cube.avail & ~hide
        B = self._bands_of(cube.r, A)
        days = np.arange(T)
        out = np.zeros((S, T, 3))
        for b in range(N_BANDS):
            for i in range(S):
                out[i] += self._band_pred(B[b], A, i, days, self._beta_a[b], self._beta_b[b])
        self.no_neighbour = np.zeros((S, T), dtype=bool)
        return (out * self._scale[:, None, :]).astype(np.float32)

    def config(self) -> dict:
        return {**super().config(), "lams": self.lams, "validation_curves": self.validation_curves,
                "pooled": self.pooled, "cutoffs_days": CUTOFFS_DAYS}


class M4Pooled(M4Band):
    name = "m4b"
    pooled = True
