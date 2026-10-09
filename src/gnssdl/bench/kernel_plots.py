"""Figures of a trained M3k kernel (`gnssdl bench kernel`).

Every curve is drawn only over the distances of the neighbours the model was
trained on (1st–99th percentile), because outside them the network's output
is an extrapolation. Weights are relative: the prediction is a normalised
average, so only ratios between weights matter.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from gnssdl.bench.m3k_kernel import M3Kernel, static_features
from gnssdl.dataset import Cube
from gnssdl.plots import GRID, SURFACE, TEXT, TEXT_2

COMPONENTS = {"East": "#2a78d6", "North": "#eb6834", "Up": "#1baf7a"}
RADIUS_SHADES = ["#9ec5f4", "#5b9ce6", "#2a78d6", "#123f7a"]     # one hue, light -> dark
DIVERGING = LinearSegmentedColormap.from_list("more_less", ["#2a78d6", "#e4e3df", "#eb6834"])
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
}


class KernelView:
    """A trained M3k network that can be evaluated at any geometry."""

    def __init__(self, cls: type[M3Kernel], radius_km: float, cube: Cube, seed: int = 0):
        import torch
        self.torch = torch
        model = cls(radius_km=radius_km, seed=seed)
        path = model._checkpoint()
        if not path.exists():
            raise FileNotFoundError(f"{path}: run `gnssdl bench run {cls.name}` first")
        self.net = model._net()
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        ri = cube.radius_index(radius_km)
        k = cls.n_neighbours
        idx = cube.nbr_idx[ri][:, :k]
        dist = cube.nbr_dist[ri][:, :k][idx >= 0]
        self.lo, self.median, self.hi = np.percentile(dist, [1, 50, 99])

    def __call__(self, d_km, az, quality) -> np.ndarray:
        """Weights (…×3) at distance `d_km`, bearing `az` (radians clockwise
        from north) for a neighbour with static features `quality`."""
        d_km, az = np.broadcast_arrays(np.asarray(d_km, float), np.asarray(az, float))
        g = np.stack([np.log(np.maximum(d_km, 1.0)), np.sin(az), np.cos(az)], -1)
        quality = np.asarray(quality, dtype=float)
        x = np.concatenate([g, np.broadcast_to(quality, g.shape[:-1] + (4,))], -1).astype(np.float32)
        with self.torch.no_grad():
            return self.torch.nn.functional.softplus(self.net(self.torch.from_numpy(x))).numpy()


def _plain_log_ticks(axis) -> None:
    """Log axis labelled 2, 5, 10, 20 … instead of 2×10⁰."""
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))


def _title(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11)


def distance_figure(view: KernelView, q_med, name: str, radius: float, m1_length: float | None, out: Path) -> Path:
    d = np.geomspace(view.lo, view.hi, 200)
    azs = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    w = view(d[:, None], azs[None, :], q_med)                       # D×A×3
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for c, (ax, (comp, col)) in enumerate(zip(axes, COMPONENTS.items())):
        wc = w[..., c] / w[..., c].mean(axis=1).max()
        ax.axvline(view.median, color=TEXT_2, linewidth=0.8, linestyle=":")
        ax.fill_between(d, wc.min(1), wc.max(1), color=col, alpha=0.18, linewidth=0,
                        label="range over direction")
        ax.plot(d, wc.mean(1), color=col, label=f"{name}, mean over direction")
        if m1_length is not None:
            g = np.ones_like(d) if np.isinf(m1_length) else np.exp(-d**2 / (2 * m1_length**2))
            ax.plot(d, g / g.max(), color=TEXT_2, linestyle="--", linewidth=1.5,
                    label=f"M1 Gaussian, L = {m1_length:g} km")
        if view.hi / view.lo > 4:                                    # narrow ranges read better linear
            ax.set_xscale("log")
            _plain_log_ticks(ax.xaxis)
        ax.set_title(comp, loc="left", fontsize=11)
        ax.set_xlabel("distance to neighbour (km)")
        if c == 0:
            ax.set_ylabel("relative weight")
            ax.legend(fontsize=8, loc="lower left")
    _title(fig, f"Learned kernel ({name}, R = {radius:g} km): weight against distance for a typical "
                f"neighbour (dotted: median neighbour distance, {view.median:.0f} km)")
    fig.tight_layout()
    path = out / f"{name}_distance_R{radius:g}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def direction_figure(view: KernelView, q_med, name: str, radius: float, out: Path) -> Path:
    dd = np.geomspace(view.lo, view.hi, 120)
    th = np.linspace(-np.pi, np.pi, 181)
    w = view(dd[:, None], th[None, :], q_med)                       # D×A×3
    ratio = np.log2(w / w.mean(axis=1, keepdims=True))
    lim = max(float(np.abs(ratio).max()), 0.05)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(wspace=0.45)
    T, D = np.meshgrid(th, np.log10(dd))
    for c, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        im = ax.pcolormesh(T, D, ratio[..., c], cmap=DIVERGING, vmin=-lim, vmax=lim, shading="auto")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)                                   # bearings clockwise, as on a map
        ax.set_rlim(np.log10(view.lo), np.log10(view.hi))
        ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000) if view.lo <= t <= view.hi / 1.4]
        ax.set_rticks(np.log10(ticks))
        ax.set_yticklabels([f"{t} km" for t in ticks], fontsize=7, color=TEXT_2)
        ax.set_title(comp, loc="left", fontsize=11)
        ax.grid(color=TEXT_2, linewidth=0.3, alpha=0.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.08)
    tv = np.array([1 / 2, 1 / 1.5, 1 / 1.2, 1, 1.2, 1.5, 2])
    tv = tv[np.abs(np.log2(tv)) <= lim]
    cb.set_ticks(np.log2(tv))
    cb.set_ticklabels([f"{t:.2g}×" for t in tv])
    cb.set_label("weight / mean weight at that distance")
    _title(fig, f"Learned kernel ({name}, R = {radius:g} km): which directions get more weight "
                "(bearing from the target, north up; radius = distance, log)")
    path = out / f"{name}_direction_R{radius:g}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def radii_figure(views: dict[float, KernelView], q_med, name: str, out: Path) -> Path:
    azs = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for (R, view), col in zip(views.items(), RADIUS_SHADES):
        d = np.geomspace(view.lo, view.hi, 200)
        w = view(d[:, None], azs[None, :], q_med).mean(1)
        for c, ax in enumerate(axes):
            ax.plot(d, w[:, c] / w[:, c].max(), color=col, label=f"R = {R:g} km")
    for ax, comp in zip(axes, COMPONENTS):
        ax.set_xscale("log")
        _plain_log_ticks(ax.xaxis)
        ax.set_title(comp, loc="left", fontsize=11)
        ax.set_xlabel("distance to neighbour (km)")
    axes[0].set_ylabel("relative weight (max = 1)")
    axes[0].legend(fontsize=8, loc="lower left")
    _title(fig, f"Learned kernel ({name}) at different exclusion radii, mean over direction")
    fig.tight_layout()
    path = out / f"{name}_radii.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def quality_figure(view: KernelView, q, q_med, name: str, radius: float, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for c, (ax, (comp, col)) in enumerate(zip(axes, COMPONENTS.items())):
        logs = np.linspace(np.percentile(q[:, c], 2), np.percentile(q[:, c], 98), 100)
        qq = np.tile(q_med, (len(logs), 1))
        qq[:, c] = logs
        w = view(np.full(len(logs), view.median), np.zeros(len(logs)), qq)[:, c]
        s_mm = np.exp(logs)
        mid = int(np.argmin(np.abs(logs - q_med[c])))               # the median station
        ax.axvspan(*np.exp(np.percentile(q[:, c], [25, 75])), color=GRID, zorder=0,
                   label="middle half of stations")
        ax.plot(s_mm, w / w[mid], color=col, label=name)
        ax.plot(s_mm, (s_mm[mid] / s_mm) ** 2, color=TEXT_2, linestyle="--", linewidth=1.5,
                label="1 / scale² (M1-like)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log_ticks(ax.xaxis)
        ax.set_title(comp, loc="left", fontsize=11)
        ax.set_xlabel("neighbour noise scale (mm)")
        if c == 0:
            ax.set_ylabel("relative weight")
            ax.legend(fontsize=8, loc="upper right")
    _title(fig, f"Learned kernel ({name}, R = {radius:g} km): weight against the neighbour's noise level, "
                f"at {view.median:.0f} km (1 = the median station)")
    fig.tight_layout()
    path = out / f"{name}_quality_R{radius:g}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def kernel_figures(cls: type[M3Kernel], cube: Cube, results_dir: Path, out: Path,
                   radius: float = 0.0, radii: tuple[float, ...] = (0.0, 25.0, 100.0, 400.0),
                   seed: int = 0) -> list[Path]:
    """All four figures for model class `cls`; returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    q = static_features(cube)
    q_med = np.median(q, axis=0)                                     # a typical neighbour
    m1_path = results_dir / "m1.json"
    m1_length = None
    if m1_path.exists():
        m1_length = {float(c["radius_km"]): c["length_km"] for c in json.loads(m1_path.read_text())}.get(radius)
    with plt.rc_context(STYLE):
        view = KernelView(cls, radius, cube, seed)
        paths = [distance_figure(view, q_med, cls.name, radius, m1_length, out),
                 direction_figure(view, q_med, cls.name, radius, out),
                 quality_figure(view, q, q_med, cls.name, radius, out)]
        views = {}
        for R in radii:
            try:
                views[R] = KernelView(cls, R, cube, seed)
            except FileNotFoundError:
                continue
        if len(views) > 1:
            paths.append(radii_figure(views, q_med, cls.name, out))
    return paths
