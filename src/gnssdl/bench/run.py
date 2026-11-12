"""Run a model through the benchmark and collect the scoreboard."""

from __future__ import annotations

import contextlib
import fcntl
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnssdl.bench.base import Reconstructor, apply_hide, station_scale
from gnssdl.bench.checks import run_checks
from gnssdl.bench import registry
from gnssdl.bench.score import (
    event_mask, per_station_scores, score_cells, score_pattern, scoring_qc, summarise,
)
from gnssdl.dataset import Cube
from gnssdl.bench.paths import RUNS, score_files, scores_path

SCORED_PATTERNS = ("scatter", "block", "station")


def variant_label(name: str, settings: dict) -> str:
    """Results label of one swept setting, e.g. c3-k4p1 for 4 regions and 1
    component; the plain name for a filter with no sweep."""
    if name == "c3":
        return f"c3-k{settings['regions']}p{settings['components']}"
    if registry.get(name).sweep:
        return name + "-" + "_".join(f"{k}{v:g}" for k, v in sorted(settings.items()))
    return name


def model_radii(name: str, cube: Cube, requested: list[float] | None = None) -> list[float]:
    """Radii a model is run at: one for radius-free and fixed-radius models,
    otherwise the requested ones or every radius of the cube."""
    spec = registry.get(name)
    if spec.radius_free:
        return [0.0]
    if spec.fixed_radius is not None:
        return [spec.fixed_radius]
    if spec.radii:
        return requested or list(spec.radii)
    return requested or [float(r) for r in cube.radii]
PARTS = ("fast", "slow", "total")


@dataclass
class ScoringContext:
    """Everything model-independent the scores need, computed once."""
    scale: np.ndarray       # S×3 station noise scale
    skip: np.ndarray        # S×T cells near in-period earthquakes (left out of fast/slow)
    keep_sta: np.ndarray    # S, False for stations dropped by scoring QC
    qc_dropped: pd.DataFrame

    @classmethod
    def build(cls, cube: Cube, steps: pd.DataFrame) -> "ScoringContext":
        keep, dropped = scoring_qc(cube, steps)
        return cls(station_scale(cube), event_mask(cube, steps), keep, dropped)


# Neighbour rule (contract §2.2, decided by the author 2026-11-12). A filter over the
# N nearest stations reads up to N of them: the stations beyond R and within the
# cube's search range. Results are reported with the counts actually read; a
# combination of R and N is flagged, not dropped, when more than half the scored
# stations have fewer than N; a station with fewer than MIN_NEIGHBOURS beyond R is
# not scored at that radius, for any filter, so filters stay comparable.
MIN_NEIGHBOURS = 4
NEIGHBOUR_COUNTS = (16, 64, 256)
SHORT_FRACTION_FLAG = 0.5


def _radius_index(cube: Cube, radius: float) -> int | None:
    hit = np.flatnonzero(np.isclose(cube.radii, radius))
    return int(hit[0]) if len(hit) else None


def scored_at(cube: Cube, keep_sta: np.ndarray, radius: float) -> np.ndarray:
    """`keep_sta` without the stations that have fewer than MIN_NEIGHBOURS
    neighbours beyond `radius`. The same for every filter. At a radius the
    cube has no neighbour lists for (the far stack at 600 km), unchanged."""
    ri = _radius_index(cube, radius)
    if ri is None:
        return keep_sta
    return keep_sta & ((cube.nbr_idx[ri] >= 0).sum(axis=1) >= MIN_NEIGHBOURS)


def neighbour_report(cube: Cube, keep_sta: np.ndarray, counts: tuple[int, ...] = NEIGHBOUR_COUNTS) -> pd.DataFrame:
    """One row per cube radius and neighbour count N, over the scored
    stations: how many are scored at the radius and how many are excluded
    (fewer than MIN_NEIGHBOURS), how many have fewer than N neighbours
    (`short`), whether the combination is flagged (more than half short),
    the median number actually read, and the median and 90th percentile of
    the distance to the farthest neighbour read."""
    rows = []
    for ri, radius in enumerate(cube.radii):
        have = (cube.nbr_idx[ri] >= 0).sum(axis=1)
        scored = scored_at(cube, keep_sta, float(radius))
        for n in counts:
            read = np.minimum(have, n)[scored]
            short = int((have[scored] < n).sum())
            far = np.array([cube.nbr_dist[ri][i, k - 1] for i, k in zip(np.flatnonzero(scored), read) if k > 0])
            total = int(scored.sum())
            rows.append({"radius_km": float(radius), "n": n, "scored": total,
                         "excluded": int((keep_sta & ~scored).sum()), "short": short,
                         "short_fraction": short / total if total else np.nan,
                         "flagged": bool(total == 0 or short / total > SHORT_FRACTION_FLAG),
                         "read_median": float(np.median(read)) if total else np.nan,
                         "reach_median_km": float(np.median(far)) if len(far) else np.nan,
                         "reach_p90_km": float(np.percentile(far, 90)) if len(far) else np.nan})
    return pd.DataFrame(rows)


def tuning_cells(cube: Cube, masks: dict[str, np.ndarray], ctx: ScoringContext,
                 cls: type) -> tuple[np.ndarray, np.ndarray]:
    """(cells hidden from the model while tuning, cells the tuning score uses).

    Models that never read the target (contract §2.1) are tuned on the
    leave-one-out fast score over every available validation day of the
    seen, scored stations: nothing needs hiding, because one prediction on
    the unhidden cube is already leave-one-out for every station. Dense days
    make the fast/slow split reliable, unlike the sparse `scatter` cells.
    Other models fall back to the validation `scatter` cells."""
    val_days = (cube.split_day == 1)[None, :]
    if getattr(cls, "never_reads_target", False):
        cells = (cube.avail & val_days & ~cube.exclude
                 & (ctx.keep_sta & (cube.split_sta == 0))[:, None])
        return np.zeros(cube.avail.shape, dtype=bool), cells
    cells = masks["scatter"] & val_days & ctx.keep_sta[:, None]
    return cells, cells


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_model(
    name: str, cube: Cube, masks: dict[str, np.ndarray], ctx: ScoringContext,
    radii: list[float] | None = None, log=print, seed: int | None = None, reuse: bool = False,
    results_dir: Path = RUNS, splits: tuple[str, ...] = ("validation",),
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit and score `name` at each exclusion radius. Returns summary rows
    and one config record (hyper-parameters, leak checks) per radius.
    `seed` (trained models only) labels the rows `name-s<seed>` unless 0.
    `reuse` loads a saved checkpoint instead of training, where one exists.
    Checkpoints and training logs go under `results_dir`. Only the mask
    scores of `splits` are computed and written (default: validation only, so
    the test years are never scored until the method is frozen; pass
    ("validation", "test") for the final run)."""
    spec = registry.get(name)
    if spec.reads_target:
        raise ValueError(f"{name} is a reference filter: it reads the target, so it is scored "
                         "only by `gnssdl bench signal`, never on the masks")
    cls = spec.cls
    label = seeded_label(name, seed)
    seed_kw = {} if seed is None else {"seed": seed}
    radii = model_radii(name, cube, radii)
    val_hide, val_cells = tuning_cells(cube, masks, ctx, cls)

    def scorer(pred: np.ndarray) -> float:
        return summarise(per_station_scores(cube, pred, val_cells, ctx.scale, ctx.skip))["nrmse_fast"]

    rows, configs = [], []
    for radius in radii:
        model: Reconstructor = cls(radius_km=radius, **seed_kw)
        _root(model, results_dir)
        if hasattr(model, "event_window"):
            model.event_window = ctx.skip
        ckpt = getattr(model, "_checkpoint", None)
        if reuse and ckpt is not None and ckpt().exists():
            model.fit(cube)                                      # loads the checkpoint
            log(f"  {label} R={radius:g}: loaded {ckpt()}")
        else:
            model.fit(cube, val_hide=val_hide, scorer=scorer)
        checks = run_checks(model, cube, masks["scatter"])
        if not checks["passed"]:
            raise RuntimeError(f"{name} R={radius:g} failed leak checks: {checks}")
        configs.append({**model.config(), "checks": checks, "git_commit": _git_commit(),
                        "scoring_qc_dropped": ctx.qc_dropped.to_dict("records")})
        for pattern in SCORED_PATTERNS:
            hide = masks[pattern]
            pred = model.predict(apply_hide(cube, hide), hide)
            fallback = getattr(model, "no_neighbour", None)
            keep = scored_at(cube, ctx.keep_sta, radius)             # neighbour rule
            for row in score_pattern(cube, pred, hide, ctx.scale, ctx.skip, keep):
                if row["split"] not in splits:
                    continue
                split = {"validation": 1, "test": 2, "prospective": 3}[row["split"]]
                cells = hide & (cube.split_day == split)[None, :] & keep[:, None]
                pooled = score_cells(cube, pred, cells, ctx.scale)
                row.update({f"pooled_{k}": v for k, v in pooled.items() if k != "cells"})
                if fallback is not None and cells.any():
                    row["fallback_fraction"] = float(fallback[cells].mean())
                rows.append({"model": label, "radius_km": np.nan if spec.radius_free else radius,
                             "pattern": pattern, **row})
        log(f"  {label} R={radius:g}: done ({_brief(configs[-1])})")
    return pd.DataFrame(rows), configs


def seeded_label(name: str, seed: int | None) -> str:
    """Results name for one training seed: seed 0 (the default) keeps the
    plain name, so earlier results stay where they were."""
    return name if not seed else f"{name}-s{seed}"


def load_model(name: str, radius: float, cube: Cube, results_dir: Path,
               ctx: "ScoringContext | None" = None, **settings) -> Reconstructor:
    """Rebuild a model with the hyper-parameters chosen in `gnssdl bench run`
    (read from its results file) and fit it, without re-tuning. `settings`
    fixes swept settings (the registry's `sweep`); `ctx` supplies the earthquake windows to
    models that leave them out of fitting."""
    spec = registry.get(name)
    cls, tuned_from = spec.cls, spec.tuning_source
    if spec.radius_free or tuned_from is None or not cls.hyperparams:
        own = spec.fixed_radius if spec.fixed_radius is not None else (0.0 if spec.radius_free else radius)
        model = cls(radius_km=own, **settings)
        _root(model, results_dir)
        if ctx is not None and hasattr(model, "event_window"):
            model.event_window = ctx.skip
        model.fit(cube)
        return model
    path = scores_path(tuned_from, results_dir, ".json")
    if not path.exists():
        raise FileNotFoundError(f"{path}: run `gnssdl bench run {tuned_from}` first")
    configs = {float(c["radius_km"]): c for c in json.loads(path.read_text())}
    if float(radius) not in configs:
        raise KeyError(f"{name} has no results at R = {radius:g} km")
    cfg = configs[float(radius)]
    kwargs = {k: float(cfg[k]) for k in cls.hyperparams if cfg.get(k) is not None}
    model = cls(radius_km=radius, **kwargs)
    _root(model, results_dir)
    model.fit(cube)
    return model


def _root(model: Reconstructor, results_dir: Path) -> None:
    """Point a trained model's checkpoints and logs at the results root."""
    if hasattr(model, "runs_root"):
        model.runs_root = Path(results_dir)


def unfitted(name: str, radius: float, results_dir: Path, **settings) -> str | None:
    """Why `load_model` would fail for this model at this radius, checked
    without fitting anything (a missing checkpoint or tuned configuration),
    or None if it would load."""
    spec = registry.get(name)
    cls, tuned_from = spec.cls, spec.tuning_source
    if spec.radius_free or tuned_from is None:
        return None
    if hasattr(cls, "_checkpoint"):
        model = cls(radius_km=radius, **settings)
        _root(model, results_dir)
        path = model._checkpoint()
        return None if path.exists() else f"no checkpoint {path}"
    if not cls.hyperparams:
        return None
    path = scores_path(tuned_from, results_dir, ".json")
    if not path.exists():
        return f"{path} missing: run `gnssdl bench run {tuned_from}` first"
    radii = {float(c["radius_km"]) for c in json.loads(path.read_text())}
    return None if float(radius) in radii else f"no tuned results at R = {radius:g} km in {path}"


def _brief(record: dict) -> str:
    extra = f"L={record['length_km']:g} km, " if record.get("length_km") is not None else ""
    if record.get("lam") is not None:
        extra = f"lambda={record['lam']:g}, "
    if record.get("best_step") is not None:
        extra = f"best step {record['best_step']}/{record['steps']}, val {record['best_val_nrmse_fast']:.4f}, "
    if record.get("best_epoch") is not None:
        extra = f"best epoch {record['best_epoch']}/{record['epochs']}, val {record['best_val_nrmse_fast']:.4f}, "
    return extra + ("leak checks passed" if record["checks"]["passed"] else "LEAK CHECKS FAILED")


def _radius_key(r) -> float:
    return -1.0 if r is None or pd.isna(r) else float(r)


@contextlib.contextmanager
def _locked(path: Path):
    """Hold an exclusive lock on `path` (read-merge-write from parallel runs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_merged(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path`, keeping rows already there for other radii: rows
    with the same (model, radius_km) are replaced, the rest are kept, so a
    run at one radius never erases results at another."""
    with _locked(path):
        _write_merged(df, path)


def _write_merged(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        old = pd.read_csv(path)
        if {"model", "radius_km"} <= set(old.columns) and {"model", "radius_km"} <= set(df.columns):
            new_keys = set(zip(df["model"].astype(str), df["radius_km"].map(_radius_key)))
            keep = [(m, r) not in new_keys
                    for m, r in zip(old["model"].astype(str), old["radius_km"].map(_radius_key))]
            df = pd.concat([old[keep], df], ignore_index=True)
            df = df.sort_values(["model", "radius_km"], kind="stable", key=lambda s: s.map(_radius_key)
                                if s.name == "radius_km" else s)
    df.to_csv(path, index=False)


def save_results(name: str, scores: pd.DataFrame, configs: list[dict], root: Path = RUNS) -> None:
    """Scores and configs in the model's run directory, merged with those of
    other radii already saved."""
    csv = scores_path(name, root)
    csv.parent.mkdir(parents=True, exist_ok=True)
    write_merged(scores, csv)
    path = scores_path(name, root, ".json")
    with _locked(path):
        new = {_radius_key(c.get("radius_km")) for c in configs}
        old = json.loads(path.read_text()) if path.exists() else []
        merged = [c for c in old if _radius_key(c.get("radius_km")) not in new] + configs
        merged.sort(key=lambda c: _radius_key(c.get("radius_km")))
        path.write_text(json.dumps(merged, indent=2, default=str) + "\n")


def scoreboard(results_dir: Path, split: str = "validation", part: str = "fast") -> pd.DataFrame:
    """Median-over-stations nrmse (mean of E/N/U) for one part, one row per
    model and radius, one column per pattern."""
    frames = [pd.read_csv(p) for p in score_files(results_dir)]
    if not frames:
        raise FileNotFoundError(f"no results in {results_dir}")
    df = pd.concat(frames)
    df = df[df.split == split].copy()
    df["R_km"] = df["radius_km"].map(lambda r: "-" if pd.isna(r) else f"{r:g}")
    df["R_sort"] = df["radius_km"].fillna(-1)
    board = df.pivot_table(index=["model", "R_sort", "R_km"], columns="pattern",
                           values=f"nrmse_{part}")
    board = board.reset_index(level="R_sort", drop=True)
    return board.reindex(columns=[p for p in SCORED_PATTERNS if p in board.columns]).round(3)
