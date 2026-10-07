"""Scores (contract §4) computed on hidden cells where the truth exists.

Every score is computed per station first and then summarised by the median
over stations, so a handful of stations with large earthquake offsets cannot
dominate. Each station's error series is split into

    slow  = 61-day running median of the error (trend drift, offsets,
            long postseismic motion)
    fast  = error - slow (day-to-day and week-to-week scatter)

The fast part measures how well the network predicts common-mode noise and
is the primary score. Signal that a model should leave alone (an offset, a
slow transient) lands mostly in the slow part, which is reported but judged
by the signal-preservation tests instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnssdl.dataset import Cube

COMPONENTS = ("e", "n", "u")
SPLITS = {1: "validation", 2: "test"}
SLOW_WINDOW = "61D"
MIN_CELLS_PER_STATION = 20
EVENT_MIN_MAG = 6.0
EVENT_HALF_WINDOW_DAYS = 30
SCORING_QC_RATIO = 3.0
# Undocumented jumps (e.g. a monument moved, WNRA: ~2.2 m in 2025): a change
# between consecutive 30-day medians larger than this, with no NGL-listed step
# of any kind within JUMP_STEP_TOLERANCE_DAYS.
JUMP_LIMIT_MM = (50.0, 50.0, 100.0)
JUMP_STEP_TOLERANCE_DAYS = 45


def _split_fast_slow(err: np.ndarray, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(err, index=days)
    slow = df.rolling(SLOW_WINDOW, center=True, min_periods=3).median().to_numpy()
    return err - slow, slow


def per_station_scores(
    cube: Cube, pred: np.ndarray, cells: np.ndarray, scale: np.ndarray,
    skip: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per station with at least MIN_CELLS_PER_STATION hidden cells.

    `skip` (S×T) removes cells from the fast/slow scores, used for windows
    around earthquakes inside the scoring period. Columns: nrmse_{fast,slow,
    total}_{e,n,u}, rmse_mm_total_{e,n,u}, cells.
    """
    days = pd.DatetimeIndex(cube.days)
    rows = []
    for i in np.flatnonzero(cells.sum(axis=1) >= MIN_CELLS_PER_STATION):
        cols = np.flatnonzero(cells[i])
        truth = cube.r[i, cols].astype(np.float64)
        err = truth - pred[i, cols]
        s = scale[i]
        if not np.all(s > 0):
            continue
        row = {"station": i, "cells": len(cols)}
        for k, c in enumerate(COMPONENTS):
            row[f"nrmse_total_{c}"] = float(np.sqrt(np.nanmean((err[:, k] / s[k]) ** 2)))
            row[f"rmse_mm_total_{c}"] = float(np.sqrt(np.nanmean(err[:, k] ** 2)))
        keep = np.ones(len(cols), dtype=bool) if skip is None else ~skip[i, cols]
        if keep.sum() >= MIN_CELLS_PER_STATION:
            fast, slow = _split_fast_slow(err[keep], days[cols[keep]])
            for k, c in enumerate(COMPONENTS):
                row[f"nrmse_fast_{c}"] = float(np.sqrt(np.nanmean((fast[:, k] / s[k]) ** 2)))
                row[f"nrmse_slow_{c}"] = float(np.sqrt(np.nanmean((slow[:, k] / s[k]) ** 2)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_station: pd.DataFrame) -> dict[str, float]:
    """Median and interquartile range over stations, per part and component,
    plus the mean of the three component medians (`nrmse_fast` etc.)."""
    out: dict[str, float] = {"stations": int(len(per_station))}
    if per_station.empty:
        return out
    out["cells"] = int(per_station["cells"].sum())
    for part in ("fast", "slow", "total"):
        meds = []
        for c in COMPONENTS:
            col = per_station.get(f"nrmse_{part}_{c}")
            if col is None or col.dropna().empty:
                continue
            q25, med, q75 = col.quantile([0.25, 0.5, 0.75])
            out[f"nrmse_{part}_{c}"] = float(med)
            out[f"nrmse_{part}_{c}_q25"] = float(q25)
            out[f"nrmse_{part}_{c}_q75"] = float(q75)
            meds.append(med)
        if meds:
            out[f"nrmse_{part}"] = float(np.mean(meds))
    return out


def event_mask(cube: Cube, steps: pd.DataFrame) -> np.ndarray:
    """S×T: cells within ±EVENT_HALF_WINDOW_DAYS of an M ≥ EVENT_MIN_MAG
    earthquake that NGL lists as possibly offsetting that station."""
    days = pd.DatetimeIndex(cube.days)
    out = np.zeros(cube.avail.shape, dtype=bool)
    pos = {s: i for i, s in enumerate(cube.sta)}
    q = steps[(steps.kind == "quake") & (steps.mag >= EVENT_MIN_MAG)
              & (steps.date >= days[0]) & (steps.date <= days[-1])]
    half = pd.Timedelta(days=EVENT_HALF_WINDOW_DAYS)
    for sta, date in zip(q.sta, q.date):
        if sta in pos:
            out[pos[sta], (days >= date - half) & (days <= date + half)] = True
    return out


def scoring_qc(cube: Cube, steps: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Stations left out of scoring because of site trouble after 2019:

    - their own day-to-day scatter in the validation and test years exceeds
      SCORING_QC_RATIO times their training-years scatter, or
    - their series jumps by more than JUMP_LIMIT_MM between consecutive
      30-day medians in those years with no listed step nearby.

    Uses only each station's own data and the steps file, so it is the same
    for every model. Returns (bool S keep-for-scoring, table of dropped)."""
    days = pd.DatetimeIndex(cube.days)
    train = (cube.split_day == 0)[None, :] & cube.avail & ~cube.exclude
    evald = (cube.split_day > 0)[None, :] & cube.avail & ~cube.exclude
    keep = np.ones(len(cube.sta), dtype=bool)
    dropped = []
    step_dates = {} if steps is None else {s: pd.DatetimeIndex(g.date) for s, g in steps.groupby("sta")}
    tol = pd.Timedelta(days=JUMP_STEP_TOLERANCE_DAYS)
    for i in range(len(cube.sta)):
        cols_eval = np.flatnonzero(evald[i])
        if len(cols_eval) >= 60:
            monthly = pd.DataFrame(cube.r[i, cols_eval], index=days[cols_eval]).resample("30D").median()
            jumps = monthly.diff().abs()
            big = (jumps > np.array(JUMP_LIMIT_MM)).any(axis=1)
            known = step_dates.get(str(cube.sta[i]), pd.DatetimeIndex([]))
            unexplained = [t for t in jumps.index[big]
                           if not ((known >= t - tol - pd.Timedelta(days=30)) & (known <= t + tol)).any()]
            if unexplained:
                keep[i] = False
                dropped.append({"sta": str(cube.sta[i]), "reason": "unexplained jump",
                                "first": str(unexplained[0].date())})
                continue
        if train[i].sum() < 365 or evald[i].sum() < 365:
            continue
        cols = np.flatnonzero(cube.avail[i])
        fast, _ = _split_fast_slow(cube.r[i, cols].astype(np.float64), days[cols])
        f = pd.DataFrame(fast, index=cols)

        def mad(m):
            x = f.loc[np.flatnonzero(m)].to_numpy()
            return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis=0)), axis=0)

        ratio = mad(evald[i]) / mad(train[i])
        if np.nanmax(ratio) > SCORING_QC_RATIO:
            keep[i] = False
            dropped.append({"sta": str(cube.sta[i]), "reason": "scatter increase",
                            "max_ratio": round(float(np.nanmax(ratio)), 2)})
    return keep, pd.DataFrame(dropped)


def score_pattern(
    cube: Cube, pred: np.ndarray, hide: np.ndarray, scale: np.ndarray,
    skip: np.ndarray | None = None, keep_sta: np.ndarray | None = None,
) -> list[dict]:
    """One summary row per split (validation, test)."""
    rows = []
    for code, name in SPLITS.items():
        cells = hide & (cube.split_day == code)[None, :]
        if keep_sta is not None:
            cells = cells & keep_sta[:, None]
        rows.append({"split": name, **summarise(per_station_scores(cube, pred, cells, scale, skip))})
    return rows


def score_cells(cube: Cube, pred: np.ndarray, cells: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    """Pooled scores over all cells at once (secondary; dominated by the
    stations with the largest values)."""
    out: dict[str, float] = {"cells": int(cells.sum())}
    if not cells.any():
        return out
    truth, guess = cube.r[cells], pred[cells]
    s = scale[np.nonzero(cells)[0]]
    for k, c in enumerate(COMPONENTS):
        ok = np.isfinite(truth[:, k]) & np.isfinite(guess[:, k]) & (s[:, k] > 0)
        err = truth[ok, k] - guess[ok, k]
        out[f"nrmse_{c}"] = float(np.sqrt(np.mean((err / s[ok, k]) ** 2)))
        out[f"cmr_{c}"] = float(1 - np.var(err) / np.var(truth[ok, k]))
    return out
