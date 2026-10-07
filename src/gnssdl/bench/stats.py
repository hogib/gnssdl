"""Uncertainty for model comparisons (contract §4): paired bootstraps.

Noise scores: each station's leave-one-out fast error is summarised per
30-day block. A bootstrap resamples blocks (the same blocks for every
station, keeping the network's shared noise together) and stations, and
recomputes each model's median-over-stations score from the same resample,
so the difference between two models gets a confidence interval.

Signal scores: injections are independent units; resampling them (paired by
injection id) gives an interval for the difference in median retention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnssdl.bench.score import _split_fast_slow
from gnssdl.dataset import Cube

BLOCK_DAYS = 30
N_BOOT = 1000


@dataclass
class BlockStats:
    """Per station and 30-day block: sum of squared normalised fast errors
    (per component) and the number of days."""
    sse: np.ndarray       # S'×B×3
    n: np.ndarray         # S'×B
    stations: np.ndarray  # S' station indices


def block_stats(cube: Cube, resid: np.ndarray, cells: np.ndarray, scale: np.ndarray) -> BlockStats:
    days = pd.DatetimeIndex(cube.days)
    cols_any = np.flatnonzero(cells.any(axis=0))
    if not len(cols_any):
        raise ValueError("no cells to score")
    first = cols_any[0]
    n_blocks = (cols_any[-1] - first) // BLOCK_DAYS + 1
    rows = np.flatnonzero(cells.sum(axis=1) >= BLOCK_DAYS)
    sse = np.zeros((len(rows), n_blocks, 3))
    n = np.zeros((len(rows), n_blocks))
    for k, i in enumerate(rows):
        cols = np.flatnonzero(cells[i])
        fast, _ = _split_fast_slow(resid[i, cols].astype(np.float64), days[cols])
        z = fast / scale[i]
        ok = np.isfinite(z).all(axis=1)
        b = (cols[ok] - first) // BLOCK_DAYS
        np.add.at(sse[k], b, z[ok] ** 2)
        np.add.at(n[k], b, 1)
    return BlockStats(sse, n, rows)


def _score(stats: BlockStats, sta_idx: np.ndarray, blk_idx: np.ndarray) -> float:
    sse = stats.sse[np.ix_(sta_idx, blk_idx)].sum(axis=1)       # S''×3
    n = stats.n[np.ix_(sta_idx, blk_idx)].sum(axis=1)           # S''
    ok = n > 0
    per_station = np.sqrt(sse[ok] / n[ok, None]).mean(axis=1)
    return float(np.median(per_station)) if len(per_station) else np.nan


def paired_bootstrap(a: BlockStats, b: BlockStats, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Median-over-stations fast score of each model, and a 95% interval for
    b - a (negative: b has the lower error)."""
    if not np.array_equal(a.stations, b.stations) or a.n.shape != b.n.shape:
        raise ValueError("block stats must cover the same stations and blocks")
    S, B = a.n.shape
    all_s, all_b = np.arange(S), np.arange(B)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        s_idx = rng.integers(0, S, S)
        b_idx = rng.integers(0, B, B)
        diffs[k] = _score(b, s_idx, b_idx) - _score(a, s_idx, b_idx)
    sa, sb = _score(a, all_s, all_b), _score(b, all_s, all_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"score_a": sa, "score_b": sb, "diff": sb - sa, "ci_low": float(lo), "ci_high": float(hi),
            "stations": S, "blocks": B}


def paired_rho_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Per footprint: median rho of each model and a 95% interval for the
    difference b - a, resampling injections paired by id."""
    m = a.merge(b, on=["id", "footprint_km"], suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    rows = []
    for L, g in m.groupby("footprint_km"):
        ra, rb = g.rho_a.to_numpy(), g.rho_b.to_numpy()
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        d = np.median(rb[idx], axis=1) - np.median(ra[idx], axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"footprint_km": L, "rho_a": float(np.median(ra)), "rho_b": float(np.median(rb)),
                     "diff": float(np.median(rb) - np.median(ra)), "ci_low": float(lo),
                     "ci_high": float(hi), "injections": len(g)})
    return pd.DataFrame(rows)
