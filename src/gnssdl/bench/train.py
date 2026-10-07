"""What learned models (M3 onward) train on (contract §1.5).

Scoring uses the real residuals `r`, earthquake offsets included. Training
does not: a squared-error loss would be dominated by the few stations with
offsets of tens to hundreds of centimetres, and every learned weight that
reproduces a shared offset is a weight that absorbs real signal. So learned
models train on

- quiet residuals: `r` minus the earthquake offsets estimated in the
  training-period fit;
- training cells only: training days, available, outside the Ridgecrest
  exclusion and outside ±30 days of listed M ≥ 6 earthquakes, on stations
  that are not held out;
- per-station normalisation by the robust noise scale, so every station
  counts alike.
"""

from __future__ import annotations

import numpy as np

from gnssdl.dataset import Cube


def quiet_residuals(cube: Cube) -> np.ndarray:
    """`r` with the kept earthquake offsets subtracted (S×T×3, float32).
    Offsets after the training period were never estimated and stay in."""
    q = cube.r.copy()
    for sta, day, amp in zip(cube.qstep_sta, cube.qstep_day, cube.qstep_amp):
        q[sta, day:] -= amp
    return q


def training_cells(cube: Cube, event_window: np.ndarray) -> np.ndarray:
    """S×T cells a learned model may train on. `event_window` is the
    scoring context's ±30-day M ≥ 6 earthquake mask."""
    train_days = (cube.split_day == 0)[None, :]
    seen = (cube.split_sta == 0)[:, None]
    return cube.avail & train_days & ~cube.exclude & ~event_window & seen


def normalise(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Divide each station's values by its S×3 robust noise scale."""
    s = np.where(scale > 0, scale, np.nan)[:, None, :]
    return (values / s).astype(np.float32)
