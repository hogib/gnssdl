"""What every model shares: the interface, hiding cells, station noise scales.

The harness, never the model, hides cells (contract §2): a model receives a
copy of the cube in which hidden cells have `avail = False` and NaN values,
and must fill them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gnssdl.dataset import Cube

CONTEXTS = ("same-day", "causal", "two-sided")


class Reconstructor:
    """Base class. Subclasses set `name` and `supported_contexts` and
    implement `fit` and `predict`."""

    name: str = "base"
    supported_contexts: tuple[str, ...] = ("same-day",)
    supports_own_history: bool = False
    # True when the prediction for a station never reads that station's own
    # values, so one pass on the unhidden cube is a leave-one-out cleaning of
    # every station at once (checked by the radius leak check).
    never_reads_target: bool = False
    # Reference filters (e.g. M2-self) read the target on purpose; they are
    # scored only on signal kept and noise removed, never on the masks.
    reference_filter: bool = False
    # Names of the hyper-parameters chosen by `fit`, stored in results and
    # restored by `load_model`.
    hyperparams: tuple[str, ...] = ()

    def __init__(self, radius_km: float = 0.0, context: str = "same-day", own_history: bool = False):
        if context not in self.supported_contexts:
            raise ValueError(f"{self.name} does not support context {context!r}")
        if own_history and not self.supports_own_history:
            raise ValueError(f"{self.name} does not support own-history")
        self.radius_km = float(radius_km)
        self.context = context
        self.own_history = own_history

    def fit(self, cube: Cube, val_hide: np.ndarray | None = None, scorer=None) -> None:
        """Learn from training days. `val_hide` (S×T) marks validation cells
        that may be used only to choose hyper-parameters; `scorer(pred)`
        returns the validation score to minimise (the harness builds it, so
        every model is tuned on the same criterion)."""

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        """`cube` already has `hide` applied. Return S×T×3 float32."""
        raise NotImplementedError

    def config(self) -> dict:
        return {"name": self.name, "radius_km": self.radius_km,
                "context": self.context, "own_history": self.own_history}


def apply_hide(cube: Cube, hide: np.ndarray, fill: float = np.nan) -> Cube:
    """Copy of `cube` with `hide` (S×T) cells unavailable. `fill` is written
    into the hidden values: NaN normally, a huge number in leak tests, where
    a model reading a hidden value would show up in its output."""
    r = cube.r.copy()
    sigma = cube.sigma.copy()
    r[hide] = fill
    sigma[hide] = np.nan
    return replace(cube, r=r, sigma=sigma, avail=cube.avail & ~hide)


def station_scale(cube: Cube) -> np.ndarray:
    """S×3 robust noise scale (1.4826 × MAD) used to normalise errors.

    From training cells (not excluded) when a station has at least 365 of
    them; otherwise, for stations that started late, from all its available
    cells. The scale is only a normaliser; it never feeds a prediction.
    """
    train = (cube.split_day == 0)[None, :] & cube.avail & ~cube.exclude
    out = np.full((len(cube.sta), 3), np.nan, dtype=np.float64)
    for i in range(len(cube.sta)):
        cells = train[i] if train[i].sum() >= 365 else cube.avail[i]
        x = cube.r[i, cells].astype(np.float64)
        if len(x):
            out[i] = 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)
    return out


def median_sigma(cube: Cube) -> np.ndarray:
    """S×3 median formal σ (mm) over training days; all days as a fallback."""
    train = (cube.split_day == 0)[None, :] & cube.avail
    out = np.full((len(cube.sta), 3), np.nan)
    for i in range(len(cube.sta)):
        cells = train[i] if train[i].any() else cube.avail[i]
        if cells.any():
            out[i] = np.median(cube.sigma[i, cells], axis=0)
    return out
