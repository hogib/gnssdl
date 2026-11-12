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
    difference b - a, resampling injections paired by id.

    The pairing is checked where it can be: rows written since 2026-11-12
    record each transient's centre, and an id must then name the same
    transient in both frames, or this raises. Rows written before carry ids
    given in packing order, which differ between filters and radii, so their
    pairs are different transients of the same footprint: `pairing` is then
    "unverified" and the interval is not a paired one."""
    m = a.merge(b, on=["id", "footprint_km"], suffixes=("_a", "_b"))
    checkable = {"lat", "lon", "amplitude_mm", "duration_days"} <= set(a.columns) & set(b.columns)
    if checkable and len(m):
        same = (np.isclose(m.lat_a, m.lat_b) & np.isclose(m.lon_a, m.lon_b)
                & (m.amplitude_mm_a == m.amplitude_mm_b) & (m.duration_days_a == m.duration_days_b))
        if not same.all():
            raise ValueError(f"{int((~same).sum())} of {len(m)} ids name different transients in the two results: "
                             "one of them was written before 2026-11-12; rerun `bench signal` for both")
    rng = np.random.default_rng(seed)
    rows = []
    for L, g in m.groupby("footprint_km"):
        ra, rb = g.rho_a.to_numpy(), g.rho_b.to_numpy()
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        d = np.median(rb[idx], axis=1) - np.median(ra[idx], axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"footprint_km": L, "rho_a": float(np.median(ra)), "rho_b": float(np.median(rb)),
                     "diff": float(np.median(rb) - np.median(ra)), "ci_low": float(lo),
                     "ci_high": float(hi), "injections": len(g),
                     "pairing": "checked" if checkable else "unverified"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Noise removed: bootstrap, and the success criterion (contract §1.6, 2026-11-12)
# --------------------------------------------------------------------------- #

CRITERION_FOOTPRINT_KM = 100.0     # planted-transient width the signal conditions are read at
CRITERION_TOLERANCE = 0.05         # allowed loss of retention / gain of spurious signal against the reference
REFERENCE = ("fs", 0.0)            # the far stack at R = 0


def _noise_arrays(df: pd.DataFrame, stations: list[str], blocks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """S×B×3 sums of squares (cleaned, raw) on a common station and block grid."""
    si = {s: k for k, s in enumerate(stations)}
    bi = {b: k for k, b in enumerate(blocks)}
    clean = np.zeros((len(stations), len(blocks), 3))
    raw = np.zeros_like(clean)
    d = df[df.sta.isin(si) & df.block.isin(bi)]
    rows, cols = d.sta.map(si).to_numpy(), d.block.map(bi).to_numpy()
    clean[rows, cols] = d[["clean_e", "clean_n", "clean_u"]].to_numpy()
    raw[rows, cols] = d[["raw_e", "raw_n", "raw_u"]].to_numpy()
    return clean, raw


def _noise_score(clean: np.ndarray, raw: np.ndarray, s_idx: np.ndarray, b_idx: np.ndarray) -> float:
    c = clean[np.ix_(s_idx, b_idx)].sum(axis=1)                  # S'×3
    r = raw[np.ix_(s_idx, b_idx)].sum(axis=1)
    ok = (r > 0).all(axis=1)
    if not ok.any():
        return np.nan
    return float(np.median(1.0 - np.sqrt(c[ok] / r[ok]).mean(axis=1)))


def noise_score(df: pd.DataFrame) -> float:
    """Noise removed from the block sums of `signal.noise_parts`: equal to
    its summary value."""
    stations, blocks = sorted(df.sta.unique()), sorted(df.block.unique())
    clean, raw = _noise_arrays(df, stations, blocks)
    return _noise_score(clean, raw, np.arange(len(stations)), np.arange(len(blocks)))


def paired_noise_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Noise removed of two filters and a 95% interval for b - a (positive:
    b removes more). Stations and 30-day blocks are resampled, the same
    resample for both filters, as in `paired_bootstrap`."""
    stations = sorted(set(a.sta) & set(b.sta))
    blocks = sorted(set(a.block) & set(b.block))
    if not stations or not blocks:
        raise ValueError("the two filters share no stations or blocks")
    ca, ra = _noise_arrays(a, stations, blocks)
    cb, rb = _noise_arrays(b, stations, blocks)
    S, B = len(stations), len(blocks)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        s_idx, b_idx = rng.integers(0, S, S), rng.integers(0, B, B)
        diffs[k] = _noise_score(cb, rb, s_idx, b_idx) - _noise_score(ca, ra, s_idx, b_idx)
    all_s, all_b = np.arange(S), np.arange(B)
    sa, sb = _noise_score(ca, ra, all_s, all_b), _noise_score(cb, rb, all_s, all_b)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"score_a": sa, "score_b": sb, "diff": sb - sa, "ci_low": float(lo), "ci_high": float(hi),
            "stations": S, "blocks": B}


def _key(model, radius) -> tuple[str, float]:
    return str(model), -1.0 if radius is None or pd.isna(radius) else float(radius)


def criterion_table(noise: pd.DataFrame, injections: pd.DataFrame, reference: tuple[str, float] = REFERENCE,
                    n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """The success criterion of contract §1.6 applied to every (model, radius)
    in `noise` (rows of `signal.noise_parts` with `model` and `radius_km`).

    A filter is `better` than the reference only if its noise removed is
    higher with a bootstrap interval excluding zero AND its median retention
    of 100-km planted transients is not lower than the reference's by more
    than 0.05 AND its median spurious signal for them is not higher by more
    than 0.05. `verdict`: better; noise gain bought with signal (noise
    higher, a signal condition fails); not better (noise not higher, or the
    interval includes zero); no signal test (no 100-km injections saved).
    `best` marks the passing filter with the highest noise removed and
    `same_as_best` the passing filters whose difference from it has an
    interval including zero. If none passes, nothing is marked."""
    groups = {_key(m, r): g for (m, r), g in noise.assign(radius_km=noise.radius_km.fillna(-1.0))
              .groupby(["model", "radius_km"])}
    if reference not in groups:
        raise ValueError(f"no noise blocks for the reference {reference[0]} at R = {reference[1]:g} km")
    wide = injections[np.isclose(injections.footprint_km, CRITERION_FOOTPRINT_KM)]
    wide = wide.assign(radius_km=wide.radius_km.fillna(-1.0))
    sig = {_key(m, r): (float(g.rho.median()), float(g.spurious_max.median()), len(g))
           for (m, r), g in wide.groupby(["model", "radius_km"])}
    if reference not in sig:
        raise ValueError(f"no 100-km planted transients for the reference {reference[0]} at R = {reference[1]:g} km")
    kept_ref, spur_ref, _ = sig[reference]
    rows = []
    for key in sorted(groups):
        res = paired_noise_bootstrap(groups[reference], groups[key], n_boot, seed)
        kept, spur, n_inj = sig.get(key, (np.nan, np.nan, 0))
        row = {"model": key[0], "radius_km": np.nan if key[1] < 0 else key[1],
               "noise_removed": res["score_b"], "diff": res["diff"], "ci_low": res["ci_low"],
               "ci_high": res["ci_high"], "kept_100": kept, "spurious_100": spur, "injections": n_inj,
               "noise_higher": res["ci_low"] > 0,
               "kept_ok": bool(kept >= kept_ref - CRITERION_TOLERANCE) if n_inj else False,
               "spurious_ok": bool(spur <= spur_ref + CRITERION_TOLERANCE) if n_inj else False}
        if key == reference:
            row["verdict"] = "reference"
        elif not n_inj:
            row["verdict"] = "no signal test"
        elif not row["noise_higher"]:
            row["verdict"] = "not better"
        elif row["kept_ok"] and row["spurious_ok"]:
            row["verdict"] = "better"
        else:
            row["verdict"] = "noise gain bought with signal"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["best"], out["same_as_best"] = False, False
    passing = out[out.verdict == "better"]
    if len(passing):
        top = passing.noise_removed.idxmax()
        out.loc[top, "best"] = True
        top_key = _key(out.model[top], out.radius_km[top])
        for i in passing.index.drop(top):
            res = paired_noise_bootstrap(groups[_key(out.model[i], out.radius_km[i])], groups[top_key], n_boot, seed)
            out.loc[i, "same_as_best"] = bool(res["ci_low"] <= 0 <= res["ci_high"])
    return out


# --------------------------------------------------------------------------- #
# An interval for every measure of every filter (added 2026-11-12)
# --------------------------------------------------------------------------- #

def _error_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """S×B×3 sums of squared normalised fast errors and S×B day counts."""
    stations, blocks = sorted(df.sta.unique()), sorted(df.block.unique())
    si = {s: k for k, s in enumerate(stations)}
    bi = {b: k for k, b in enumerate(blocks)}
    sse, n = np.zeros((len(stations), len(blocks), 3)), np.zeros((len(stations), len(blocks)))
    rows, cols = df.sta.map(si).to_numpy(), df.block.map(bi).to_numpy()
    sse[rows, cols] = df[["err_e", "err_n", "err_u"]].to_numpy()
    n[rows, cols] = df.err_days.to_numpy()
    keep = n.sum(axis=1) > 0
    return sse[keep], n[keep]


def _interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def interval_table(noise: pd.DataFrame | None, injections: pd.DataFrame | None, seasonal: pd.DataFrame | None,
                   n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Every measure of every (model, radius) with a 95% bootstrap interval,
    from saved results. One row per model, radius, measure and, for the
    planted transients, footprint:

      noise_removed   stations and 30-day blocks resampled (noise blocks)
      error           the leave-one-out fast error of `bench compare`, same
                      resampling; absent for target-inclusive filters
      kept, spurious  median over planted transients of rho and of the largest
                      spurious amplitude, transients resampled, per footprint
      seasonal        median over stations of rho_all, stations resampled

    `n` is the number of units resampled (stations or transients)."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(model, radius, measure, value, boot, n, footprint=np.nan):
        lo, hi = _interval(boot)
        rows.append({"model": model, "radius_km": np.nan if radius < 0 else radius, "measure": measure,
                     "footprint_km": footprint, "value": value, "ci_low": lo, "ci_high": hi, "n": int(n)})

    def groups(df):
        return df.assign(radius_km=df.radius_km.fillna(-1.0)).groupby(["model", "radius_km"])

    if noise is not None:
        for (model, radius), g in groups(noise):
            stations, blocks = sorted(g.sta.unique()), sorted(g.block.unique())
            clean, raw = _noise_arrays(g, stations, blocks)
            S, B = len(stations), len(blocks)
            draws = [(rng.integers(0, S, S), rng.integers(0, B, B)) for _ in range(n_boot)]
            add(model, radius, "noise_removed", _noise_score(clean, raw, np.arange(S), np.arange(B)),
                np.array([_noise_score(clean, raw, si, bi) for si, bi in draws]), S)
            if "err_days" in g and g.err_days.sum() > 0:
                st = BlockStats(*_error_arrays(g), stations=np.zeros(0))
                S2, B2 = st.n.shape
                add(model, radius, "error", _score(st, np.arange(S2), np.arange(B2)),
                    np.array([_score(st, rng.integers(0, S2, S2), rng.integers(0, B2, B2)) for _ in range(n_boot)]), S2)
    if injections is not None:
        for (model, radius), g in groups(injections):
            for L, h in g.groupby("footprint_km"):
                idx = rng.integers(0, len(h), (n_boot, len(h)))
                for measure, col in (("kept", "rho"), ("spurious", "spurious_max")):
                    v = h[col].to_numpy(float)
                    add(model, radius, measure, float(np.median(v)), np.median(v[idx], axis=1), len(h), float(L))
    if seasonal is not None:
        for (model, radius), g in groups(seasonal):
            v = g.rho_all.dropna().to_numpy(float)
            if len(v):
                idx = rng.integers(0, len(v), (n_boot, len(v)))
                add(model, radius, "seasonal", float(np.median(v)), np.median(v[idx], axis=1), len(v))
    return pd.DataFrame(rows, columns=["model", "radius_km", "measure", "footprint_km", "value", "ci_low",
                                       "ci_high", "n"])

