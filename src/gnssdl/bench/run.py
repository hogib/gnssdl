"""Run a model through the benchmark and collect the scoreboard."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnssdl.bench.base import Reconstructor, apply_hide, station_scale
from gnssdl.bench.checks import run_checks
from gnssdl.bench.m0_zero import M0Zero
from gnssdl.bench.m1_stack import M1K64, M1K256, M1Stack
from gnssdl.bench.audit import C1Stack, D1FarStack, FarStackSweep
from gnssdl.bench.m2_robust import M2Robust, M2Self
from gnssdl.bench.score import (
    event_mask, per_station_scores, score_cells, score_pattern, scoring_qc, summarise,
)
from gnssdl.dataset import Cube

MODELS = {"m0": M0Zero, "m1": M1Stack, "m1k64": M1K64, "m1k256": M1K256, "m2": M2Robust,
          "d1": D1FarStack, "fs": FarStackSweep}
# name -> (class, model whose tuning it reuses, or None if it has no hyper-parameters)
REFERENCE_FILTERS = {"m2self": (M2Self, "m2"), "c1": (C1Stack, None)}
SCORED_PATTERNS = ("scatter", "block", "station")
RADIUS_FREE = {"m0", "c1"}   # models with no exclusion radius: run once
FIXED_RADIUS = {"d1": 400.0, "m2self": 0.0}   # models defined at one radius
# models that need no neighbour lists and so can run at radii outside the cube
OWN_RADII = {"fs": (0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 600.0)}


def model_radii(name: str, cube: Cube, requested: list[float] | None = None) -> list[float]:
    """Radii a model is run at: one for radius-free and fixed-radius models,
    otherwise the requested ones or every radius of the cube."""
    if name in RADIUS_FREE:
        return [0.0]
    if name in FIXED_RADIUS:
        return [FIXED_RADIUS[name]]
    if name in OWN_RADII:
        return requested or list(OWN_RADII[name])
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
    radii: list[float] | None = None, log=print,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit and score `name` at each exclusion radius. Returns summary rows
    and one config record (hyper-parameters, leak checks) per radius."""
    if name in REFERENCE_FILTERS:
        raise ValueError(f"{name} is a reference filter: it reads the target, so it is scored "
                         "only by `gnssdl bench signal`, never on the masks")
    cls = MODELS[name]
    radii = model_radii(name, cube, radii)
    val_hide, val_cells = tuning_cells(cube, masks, ctx, cls)

    def scorer(pred: np.ndarray) -> float:
        return summarise(per_station_scores(cube, pred, val_cells, ctx.scale, ctx.skip))["nrmse_fast"]

    rows, configs = [], []
    for radius in radii:
        model: Reconstructor = cls(radius_km=radius)
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
            for row in score_pattern(cube, pred, hide, ctx.scale, ctx.skip, ctx.keep_sta):
                split = {"validation": 1, "test": 2, "prospective": 3}[row["split"]]
                cells = hide & (cube.split_day == split)[None, :] & ctx.keep_sta[:, None]
                pooled = score_cells(cube, pred, cells, ctx.scale)
                row.update({f"pooled_{k}": v for k, v in pooled.items() if k != "cells"})
                if fallback is not None and cells.any():
                    row["fallback_fraction"] = float(fallback[cells].mean())
                rows.append({"model": name, "radius_km": np.nan if name in RADIUS_FREE else radius,
                             "pattern": pattern, **row})
        log(f"  {name} R={radius:g}: done ({_brief(configs[-1])})")
    return pd.DataFrame(rows), configs


def load_model(name: str, radius: float, cube: Cube, results_dir: Path) -> Reconstructor:
    """Rebuild a model with the hyper-parameters chosen in `gnssdl bench run`
    (read from its results file) and fit it, without re-tuning."""
    tuned_from = name
    if name in REFERENCE_FILTERS:
        cls, tuned_from = REFERENCE_FILTERS[name]
    else:
        cls = MODELS[name]
    if name in RADIUS_FREE or tuned_from is None or not cls.hyperparams:
        model = cls(radius_km=FIXED_RADIUS.get(name, 0.0 if name in RADIUS_FREE else radius))
        model.fit(cube)
        return model
    path = results_dir / f"{tuned_from}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run `gnssdl bench run {tuned_from}` first")
    configs = {float(c["radius_km"]): c for c in json.loads(path.read_text())}
    if float(radius) not in configs:
        raise KeyError(f"{name} has no results at R = {radius:g} km")
    cfg = configs[float(radius)]
    kwargs = {k: float(cfg[k]) for k in cls.hyperparams if cfg.get(k) is not None}
    model = cls(radius_km=radius, **kwargs)
    model.fit(cube)
    return model


def _brief(record: dict) -> str:
    extra = f"L={record['length_km']:g} km, " if record.get("length_km") is not None else ""
    return extra + ("leak checks passed" if record["checks"]["passed"] else "LEAK CHECKS FAILED")


def save_results(name: str, scores: pd.DataFrame, configs: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.json").write_text(json.dumps(configs, indent=2, default=str) + "\n")


def scoreboard(results_dir: Path, split: str = "test", part: str = "fast") -> pd.DataFrame:
    """Median-over-stations nrmse (mean of E/N/U) for one part, one row per
    model and radius, one column per pattern."""
    frames = [pd.read_csv(p) for p in sorted(results_dir.glob("*.csv"))]
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
