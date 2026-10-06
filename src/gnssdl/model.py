"""Trajectory model for daily GNSS positions.

    x(t) = a + v·t + Σ seasonal(1/yr, 2/yr) + Σ H(t − t_k)·b_k

fitted by least squares on a reference window. Extrapolating it beyond the
window and subtracting gives the *departure from the reference behaviour*:
anything the reference interval did not contain (a coseismic offset,
postseismic relaxation, a change in loading rate) is what is left.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("e", "n", "u")


def design_matrix(t: np.ndarray, steps: list[float], t0: float) -> np.ndarray:
    """`t`, `steps` and the velocity reference epoch `t0` in decimal years.
    Centring on `t0` keeps the velocity column well conditioned."""
    w = 2 * np.pi * t
    cols = [np.ones_like(t), t - t0, np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
    cols += [(t >= s).astype(float) for s in steps]
    return np.column_stack(cols)


def _to_dec(d: pd.Timestamp) -> float:
    return d.year + (d.dayofyear - 0.5) / (366 if d.is_leap_year else 365)


def trajectory_residuals(
    series: pd.DataFrame,
    fit_mask: np.ndarray,
    step_dates: list[pd.Timestamp] = (),
    keep_step_dates: list[pd.Timestamp] = (),
    equipment_dates: list[pd.Timestamp] = (),
    equipment_window_days: int = 30,
    min_epochs_each_side: int = 1,
    min_fit_epochs: int = 365,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Residuals of every epoch against a model fitted on the rows `fit_mask`
    selects.

    A step is estimated when the fit set has at least `min_epochs_each_side`
    epochs on both sides of it; others are left in the residual. Steps in
    `keep_step_dates` (earthquakes) are estimated so they don't bias the
    trend, but added back: they stay in the residual as signal. Equipment
    changes after the last fit epoch are not signal, and are removed with a
    local jump estimate (median of the residual `equipment_window_days` after
    minus before), so an antenna swap months before an event cannot pose as a
    precursor. Returns residuals in mm and the fitted velocities in mm/yr.
    """
    fit_mask = np.asarray(fit_mask, dtype=bool)
    t_all = series["decyear"].to_numpy()
    t_fit = t_all[fit_mask]
    if len(t_fit) < min_fit_epochs:
        raise ValueError(f"only {len(t_fit)} epochs in the fit window")

    keep = {_to_dec(d) for d in keep_step_dates}
    steps = sorted(
        s for s in {_to_dec(d) for d in step_dates}
        if (t_fit < s).sum() >= min_epochs_each_side and (t_fit >= s).sum() >= min_epochs_each_side
    )
    kept_cols = [6 + k for k, s in enumerate(steps) if s in keep]

    t0 = t_fit.mean()
    G_fit = design_matrix(t_fit, steps, t0)
    G_all = design_matrix(t_all, steps, t0)

    out = pd.DataFrame(index=series.index)
    vel = {}
    for c in COMPONENTS:
        m, *_ = np.linalg.lstsq(G_fit, series[c].to_numpy()[fit_mask], rcond=None)
        model = G_all @ m - G_all[:, kept_cols] @ m[kept_cols]
        out[c] = (series[c].to_numpy() - model) * 1000.0
        vel[c] = m[1] * 1000.0

    last_fit = series.index[fit_mask].max()
    w = pd.Timedelta(days=equipment_window_days)
    for d in sorted(equipment_dates):
        if d <= last_fit:
            continue
        before, after = out.loc[d - w : d - pd.Timedelta(days=1)], out.loc[d : d + w]
        if len(before) < 5 or len(after) < 5:
            continue  # not enough data to estimate it; leave it visible
        jump = after.median() - before.median()
        out.loc[d:] -= jump
    return out, vel


def departure(
    series: pd.DataFrame,
    fit_start: str,
    fit_end: str,
    step_dates: list[pd.Timestamp] = (),
    equipment_dates: list[pd.Timestamp] = (),
    equipment_window_days: int = 30,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Residuals of every epoch against a model fitted on [fit_start, fit_end].

    All steps inside the window are estimated and removed; later ones stay in
    the residual, except equipment changes (see `trajectory_residuals`).
    """
    idx = series.index
    fit_mask = (idx >= pd.Timestamp(fit_start)) & (idx <= pd.Timestamp(fit_end))
    return trajectory_residuals(
        series, fit_mask, step_dates, (), equipment_dates, equipment_window_days
    )
