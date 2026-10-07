"""Evaluation masks (contract §3): generated once, shared by every model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gnssdl.dataset import Cube

SEED = 20260101
SCATTER_FRACTION = 0.10
BLOCK_DAYS = 30
PATTERNS = ("scatter", "block", "station", "ridgecrest")


def make_masks(cube: Cube, seed: int = SEED) -> dict[str, np.ndarray]:
    """S×T bool masks. Only validation and test days are ever hidden for
    scoring; Ridgecrest-excluded cells get their own pattern and are kept out
    of the others."""
    rng = np.random.default_rng(seed)
    S, T = cube.avail.shape
    seen = cube.split_sta == 0
    eval_day = cube.split_day > 0
    base = cube.avail & eval_day[None, :] & ~cube.exclude

    scatter = base & seen[:, None] & (rng.random((S, T)) < SCATTER_FRACTION)

    block = np.zeros((S, T), dtype=bool)
    days = pd.DatetimeIndex(cube.days)
    for year in sorted(set(days[eval_day].year)):
        cols = np.flatnonzero(eval_day & (days.year == year))
        if len(cols) < BLOCK_DAYS:
            continue
        starts = rng.integers(0, len(cols) - BLOCK_DAYS + 1, size=S)
        for i in np.flatnonzero(seen):
            block[i, cols[starts[i]: starts[i] + BLOCK_DAYS]] = True
    block &= base & seen[:, None]

    station = base & ~seen[:, None]
    # All available excluded cells, including Jul-Dec 2019 (training-period
    # days that no model was allowed to train on).
    ridgecrest = cube.exclude & cube.avail
    return {"scatter": scatter, "block": block, "station": station, "ridgecrest": ridgecrest}


def save_masks(masks: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **masks)


def load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
