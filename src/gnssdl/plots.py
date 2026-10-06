"""Small-multiple departure plots: one row per station, one column per component."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
EVENT = "#52514e"

COMPONENT_LABEL = {"e": "East", "n": "North", "u": "Up"}


def departure_grid(
    residuals: dict[str, pd.DataFrame],
    subtitles: dict[str, str],
    events: list[tuple[pd.Timestamp, str]],
    out: Path,
    title: str,
    fit_window: tuple[str, str],
    xlim: tuple[str, str],
    corrected_steps: dict[str, list[pd.Timestamp]] | None = None,
) -> Path:
    stations = list(residuals)
    fig, axes = plt.subplots(
        len(stations), 3, figsize=(13, 1.55 * len(stations) + 1.4),
        sharex=True, squeeze=False, facecolor=SURFACE,
    )
    x0, x1 = pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1])
    for i, sta in enumerate(stations):
        r = residuals[sta].loc[x0:x1]
        for j, c in enumerate(("e", "n", "u")):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.axvspan(pd.Timestamp(fit_window[0]), pd.Timestamp(fit_window[1]),
                       color=GRID, alpha=0.45, lw=0, zorder=0)
            ax.axhline(0, color=TEXT_2, lw=0.6, zorder=1)
            for t, _ in events:
                ax.axvline(t, color=EVENT, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.plot(r.index, r[c], ".", ms=1.6, color=SERIES, zorder=2, rasterized=True)
            for d in (corrected_steps or {}).get(sta, []):
                if x0 <= d <= x1:
                    ax.plot(d, 0, "^", transform=ax.get_xaxis_transform(), ms=5,
                            color=TEXT_2, clip_on=False, zorder=3)
            if r[c].notna().any():
                # A handful of bad days would otherwise set the scale for the panel.
                lo, hi = r[c].quantile([0.005, 0.995])
                lo, hi = min(lo, 0.0), max(hi, 0.0)
                pad = 0.12 * max(hi - lo, 2.0)
                ax.set_ylim(lo - pad, hi + pad)
            ax.grid(axis="y", color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=TEXT_2, labelsize=7.5, length=2)
            if i == 0:
                ax.set_title(f"{COMPONENT_LABEL[c]} (mm)", color=TEXT, fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(f"{sta}\n{subtitles.get(sta, '')}", color=TEXT, fontsize=8.5,
                              rotation=0, ha="right", va="center", labelpad=8)
    ax = axes[-1, 0]
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=12)
    event_text = "   ".join(f"{t.date()} {label}" for t, label in events)
    fig.text(0.01, 0.006,
             f"Dashed: {event_text}.   Shaded: reference window {fit_window[0]} to {fit_window[1]} "
             "(trend + annual + semiannual + in-window steps); elsewhere, departure from that model "
             "extrapolated.\n▲ equipment change after the reference window, removed with a local ±30-day jump estimate.   Y-axes clipped to the 0.5–99.5 percentile range.   "
             "Source: Nevada Geodetic Laboratory, IGS20 daily.",
             color=TEXT_2, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
