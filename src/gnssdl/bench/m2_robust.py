"""M2: robust neighbour statistic (docs/03-m2-robust-neighbour.md).

For each station, day and component: drop neighbours whose value is a spike
relative to their own 31-day running median, then take the weighted median
of the rest with M1's weights. The weighted median is interpolated as in
Kreemer & Blewitt (2021, §3.6), so equal weights give the ordinary median.
Fewer than N_MIN surviving neighbours falls back to M1, and no neighbour at
all to zero.

M2Self is the labelled reference filter that also puts the target's own
value into its median, as CMC Imaging does. It reads the target, so it is
never scored on the masks; only on signal kept and noise removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnssdl.bench.base import Reconstructor, apply_hide, median_sigma, station_scale
from gnssdl.dataset import Cube, crop_days

LENGTH_SCALES_KM = (25.0, 50.0, 100.0, 200.0, 500.0, math.inf)
SCREEN_K = (2.0, 3.0, 5.0, math.inf)
SCREEN_WINDOW_DAYS = 31
N_MIN = 3
CHUNK = 32


def weighted_median(values: np.ndarray, weights: np.ndarray, axis: int = 1) -> np.ndarray:
    """Interpolated weighted median along `axis`.

    Entries with weight 0 (or NaN value) are ignored. Each kept value gets the
    percentile p_j = c_j - w'_j / 2 (c the cumulative normalised weight), and
    the median is interpolated linearly at p = 0.5. Returns NaN where nothing
    is kept."""
    v = np.moveaxis(values, axis, -1).astype(np.float64)
    w = np.moveaxis(weights, axis, -1).astype(np.float64)
    keep = (w > 0) & np.isfinite(v)
    w = np.where(keep, w, 0.0)
    v = np.where(keep, v, np.inf)            # ignored entries sort to the end
    order = np.argsort(v, axis=-1, kind="stable")
    v = np.take_along_axis(v, order, axis=-1)
    w = np.take_along_axis(w, order, axis=-1)
    total = w.sum(axis=-1, keepdims=True)
    wn = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    p = np.cumsum(wn, axis=-1) - wn / 2
    p = np.where(wn > 0, p, np.inf)          # ignored entries never straddle 0.5
    j = np.argmax(p >= 0.5, axis=-1)[..., None]
    jm = np.maximum(j - 1, 0)
    v_hi, v_lo = np.take_along_axis(v, j, -1), np.take_along_axis(v, jm, -1)
    p_hi, p_lo = np.take_along_axis(p, j, -1), np.take_along_axis(p, jm, -1)
    first = j == 0
    # Rows with nothing kept hold inf - inf here; they are set to NaN below.
    with np.errstate(invalid="ignore"):
        frac = np.divide(0.5 - p_lo, p_hi - p_lo, out=np.zeros_like(p_hi), where=~first & (p_hi > p_lo))
        med = np.where(first, v_hi, v_lo + frac * (v_hi - v_lo))[..., 0]
    return np.where(total[..., 0] > 0, med, np.nan)


@dataclass
class _Prepared:
    x: np.ndarray        # S×T×3 values, NaN where unavailable
    a: np.ndarray        # S×T availability
    spike: np.ndarray    # S×T×3 |x - running median| / scale


class M2Robust(Reconstructor):
    name = "m2"
    never_reads_target = True
    hyperparams = ("length_km", "k")
    include_self = False

    def __init__(self, radius_km: float = 0.0, length_km: float | None = None,
                 k: float | None = None, **kw):
        super().__init__(radius_km, **kw)
        self.length_km = length_km
        self.k = k
        self.validation_curve: dict[str, float] = {}
        self._sbar = self._scale = None

    # -- fitting ---------------------------------------------------------- #

    def fit(self, cube: Cube, val_hide: np.ndarray | None = None, scorer=None) -> None:
        self._sbar = median_sigma(cube)
        self._scale = station_scale(cube)
        if self.length_km is not None and self.k is not None:
            return
        if val_hide is None or scorer is None:
            raise ValueError("M2 tunes L and k on validation cells; pass val_hide and scorer")
        # Tune on the validation years only (plus a margin for the running
        # median); the score only looks at validation cells anyway.
        days = np.flatnonzero(cube.split_day == 1)
        lo, hi = max(0, days[0] - SCREEN_WINDOW_DAYS), min(len(cube.days), days[-1] + SCREEN_WINDOW_DAYS + 1)
        small = crop_days(apply_hide(cube, val_hide), lo, hi)
        prep = self._prepare(small)
        best = None
        for L in LENGTH_SCALES_KM:
            for k in SCREEN_K:
                full = np.zeros(cube.r.shape, dtype=np.float32)
                full[:, lo:hi] = self._predict_prepared(small, prep, L, k)
                score = float(scorer(full))
                self.validation_curve[f"L={L:g},k={k:g}"] = round(score, 5)
                if best is None or score < best[0]:
                    best = (score, L, k)
        _, self.length_km, self.k = best

    # -- prediction ------------------------------------------------------- #

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        if self._sbar is None or self.length_km is None or self.k is None:
            raise RuntimeError("fit first")
        return self._predict_prepared(cube, self._prepare(cube), self.length_km, self.k)

    def _prepare(self, cube: Cube) -> _Prepared:
        """Running-median spike statistic per station, from available cells
        only (hidden cells never enter it)."""
        a = cube.avail
        x = np.where(a[..., None], cube.r, np.nan).astype(np.float32)
        days = pd.DatetimeIndex(cube.days)
        spike = np.full(x.shape, np.inf, dtype=np.float32)
        for i in np.flatnonzero(a.any(axis=1)):
            df = pd.DataFrame(x[i], index=days)
            med = df.rolling(SCREEN_WINDOW_DAYS, center=True, min_periods=5).median().to_numpy()
            dev = np.abs(x[i] - med) / self._scale[i]
            spike[i] = np.where(np.isfinite(dev), dev, 0.0)   # too few days to judge: keep
        return _Prepared(x, a, spike)

    def _predict_prepared(self, cube: Cube, prep: _Prepared, length_km: float, k: float) -> np.ndarray:
        ri = cube.radius_index(self.radius_km)
        idx = cube.nbr_idx[ri]
        dist = cube.nbr_dist[ri].astype(np.float64)
        valid = idx >= 0
        nb = np.where(valid, idx, 0)
        kernel = np.ones_like(dist) if math.isinf(length_km) else np.exp(-dist**2 / (2 * length_km**2))
        w = np.nan_to_num(np.where(valid, kernel, 0.0)[:, :, None] / self._sbar[nb] ** 2, nan=0.0)  # S×K×3

        S, T, _ = cube.r.shape
        out = np.zeros((S, T, 3), dtype=np.float32)
        self.no_neighbour = np.zeros((S, T), dtype=bool)
        self.fell_back_to_m1 = np.zeros((S, T), dtype=bool)
        for lo in range(0, S, CHUNK):
            hi = min(lo + CHUNK, S)
            vals = prep.x[nb[lo:hi]]                                  # c×K×T×3
            avail = prep.a[nb[lo:hi]][..., None] & valid[lo:hi, :, None, None]
            ww = w[lo:hi, :, None, :] * avail                         # c×K×T×3
            if self.include_self:
                own_w = (1.0 / self._sbar[lo:hi] ** 2)[:, None, None, :] * prep.a[lo:hi][:, None, :, None]
                vals = np.concatenate([vals, prep.x[lo:hi][:, None]], axis=1)
                ww = np.concatenate([ww, own_w], axis=1)
                spk = np.concatenate([prep.spike[nb[lo:hi]], prep.spike[lo:hi][:, None]], axis=1)
            else:
                spk = prep.spike[nb[lo:hi]]
            screened = ww * (spk <= k)
            med = weighted_median(vals, screened, axis=1)             # c×T×3
            n_left = (screened > 0).sum(axis=1)
            den = ww.sum(axis=1)
            mean = np.divide((ww * np.nan_to_num(vals)).sum(axis=1), den,
                             out=np.zeros_like(den), where=den > 0)
            use_m1 = n_left < N_MIN
            pred = np.where(use_m1, mean, med)
            out[lo:hi] = np.where(den > 0, np.nan_to_num(pred), 0.0)
            self.no_neighbour[lo:hi] = den[..., 0] <= 0
            self.fell_back_to_m1[lo:hi] = use_m1[..., 0] & (den[..., 0] > 0)
        return out

    def config(self) -> dict:
        return {**super().config(), "length_km": self.length_km, "k": self.k,
                "validation_curve": self.validation_curve}


class M2Self(M2Robust):
    """Reference filter: M2 at R = 0 with the target's own same-day value in
    its median (weight 1/σ̄² at zero distance), as CMC Imaging does. Reads the
    target by design, so it is excluded from the masked scores (contract
    §2.1) and only scored on signal kept and noise removed."""

    name = "m2self"
    never_reads_target = False
    reference_filter = True
    include_self = True
