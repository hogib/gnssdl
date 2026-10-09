"""Leak checks every model must pass (contract §6), run by the harness.

Each check compares predictions on two inputs that differ only in data the
model is not allowed to use. If the predictions differ, the model peeked.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gnssdl.bench.base import Reconstructor, apply_hide
from gnssdl.dataset import Cube, crop_days, distance_azimuth

POISON = 1.0e6   # mm; any use of a poisoned value shows up immediately
CHECK_DAYS = 365  # the checks run on one year: a leak is local in time, so it shows there too
TOLERANCE = 1e-3  # mm


def check_hidden_not_read(model: Reconstructor, cube: Cube, hide: np.ndarray) -> float:
    """Hidden cells filled with NaN vs with POISON (still unavailable).
    Returns the largest prediction difference in mm."""
    a = model.predict(apply_hide(cube, hide), hide)
    b = model.predict(apply_hide(cube, hide, fill=POISON), hide)
    return float(np.nanmax(np.abs(a - b)))


def check_radius(
    model: Reconstructor, cube: Cube, hide: np.ndarray, n_targets: int = 5, seed: int = 0
) -> float:
    """Poison every station within the exclusion radius of a target,
    including the target itself (its own values must not be used either:
    own-history is off). The target's predictions must not change. Returns
    the largest difference in mm over the sampled targets."""
    rng = np.random.default_rng(seed)
    d, _ = distance_azimuth(cube.lat, cube.lon)
    base = apply_hide(cube, hide)
    a = model.predict(base, hide)
    worst = 0.0
    for i in rng.choice(np.flatnonzero(cube.split_sta == 0), size=n_targets, replace=False):
        near = d[i] <= model.radius_km
        near[i] = True
        r = base.r.copy()
        r[near] = np.where(base.avail[near][..., None], POISON, r[near])
        b = model.predict(replace(base, r=r), hide)
        worst = max(worst, float(np.nanmax(np.abs(a[i] - b[i]))))
    return worst


def check_slice(cube: Cube, days: int = CHECK_DAYS) -> tuple[int, int]:
    """Day range the checks run on: the first `days` days of the validation
    split (or of the cube when it has none). Running a model on one year
    instead of the whole record makes the checks cheap for slow models; any
    read of a hidden or forbidden value inside that year still shows."""
    val = np.flatnonzero(cube.split_day == 1)
    lo = int(val[0]) if len(val) else 0
    hi = min(cube.r.shape[1], lo + days)
    return max(0, hi - days), hi


def run_checks(model: Reconstructor, cube: Cube, hide: np.ndarray, days: int | None = CHECK_DAYS) -> dict:
    if days is not None and cube.r.shape[1] > days:
        lo, hi = check_slice(cube, days)
        cube, hide = crop_days(cube, lo, hi), hide[:, lo:hi]
    hidden = check_hidden_not_read(model, cube, hide)
    radius = check_radius(model, cube, hide)
    return {
        "hidden_not_read_max_diff_mm": hidden,
        "radius_max_diff_mm": radius,
        "passed": bool(hidden < TOLERANCE and radius < TOLERANCE),
    }
