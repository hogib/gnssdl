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
from gnssdl.bench.m1_stack import M1Stack
from gnssdl.bench.score import (
    event_mask, per_station_scores, score_cells, score_pattern, scoring_qc, summarise,
)
from gnssdl.dataset import Cube

MODELS = {"m0": M0Zero, "m1": M1Stack}
SCORED_PATTERNS = ("scatter", "block", "station")
RADIUS_FREE = {"m0"}   # models that don't use neighbours: run once
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
    cls = MODELS[name]
    radii = [0.0] if name in RADIUS_FREE else (radii or [float(r) for r in cube.radii])
    val_cells = masks["scatter"] & (cube.split_day == 1)[None, :] & ctx.keep_sta[:, None]

    def scorer(pred: np.ndarray) -> float:
        return summarise(per_station_scores(cube, pred, val_cells, ctx.scale, ctx.skip))["nrmse_fast"]

    rows, configs = [], []
    for radius in radii:
        model: Reconstructor = cls(radius_km=radius)
        model.fit(cube, val_hide=val_cells, scorer=scorer)
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
                split = {"validation": 1, "test": 2}[row["split"]]
                cells = hide & (cube.split_day == split)[None, :] & ctx.keep_sta[:, None]
                pooled = score_cells(cube, pred, cells, ctx.scale)
                row.update({f"pooled_{k}": v for k, v in pooled.items() if k != "cells"})
                if fallback is not None and cells.any():
                    row["fallback_fraction"] = float(fallback[cells].mean())
                rows.append({"model": name, "radius_km": np.nan if name in RADIUS_FREE else radius,
                             "pattern": pattern, **row})
        log(f"  {name} R={radius:g}: done ({_brief(configs[-1])})")
    return pd.DataFrame(rows), configs


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
