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


def departure(
    series: pd.DataFrame,
    fit_start: str,
    fit_end: str,
    step_dates: list[pd.Timestamp] = (),
    equipment_dates: list[pd.Timestamp] = (),
    equipment_window_days: int = 30,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Residuals of every epoch against a model fitted on [fit_start, fit_end].

    Steps inside the fit window are estimated (with data on both sides); later
    ones are deliberately left in the residual -- except equipment changes,
    which are not signal. Those are removed after the fact with a local jump
    estimate (median of the residual `equipment_window_days` after minus
    before), so an antenna swap months before an event cannot pose as a
    precursor. Returns residuals in mm and the fitted velocities in mm/yr.
    """
    fit = series.loc[fit_start:fit_end]
    t_fit = fit["decyear"].to_numpy()
    if len(t_fit) < 365:
        raise ValueError(f"only {len(t_fit)} epochs in the fit window")
    to_dec = lambda d: d.year + (d.dayofyear - 0.5) / (366 if d.is_leap_year else 365)
    steps = [to_dec(d) for d in step_dates if t_fit.min() < to_dec(d) < t_fit.max()]

    t0 = t_fit.mean()
    G_fit = design_matrix(t_fit, steps, t0)
    G_all = design_matrix(series["decyear"].to_numpy(), steps, t0)

    out = pd.DataFrame(index=series.index)
    vel = {}
    for c in COMPONENTS:
        m, *_ = np.linalg.lstsq(G_fit, fit[c].to_numpy(), rcond=None)
        out[c] = (series[c].to_numpy() - G_all @ m) * 1000.0
        vel[c] = m[1] * 1000.0

    w = pd.Timedelta(days=equipment_window_days)
    for d in sorted(equipment_dates):
        if d <= fit.index.max():
            continue
        before, after = out.loc[d - w : d - pd.Timedelta(days=1)], out.loc[d : d + w]
        if len(before) < 5 or len(after) < 5:
            continue  # not enough data to estimate it; leave it visible
        jump = after.median() - before.median()
        out.loc[d:] -= jump
    return out, vel
