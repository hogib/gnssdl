"""Figures for the proposal and the findings report (`gnssdl figures`).

- data_processing: one station's raw daily positions, the fitted trajectory,
  the residual the benchmark uses, and one year of it with the network
  common mode removed.
- network: the stations, held-out and excluded ones, and for one target
  the exclusion radius, its nearest neighbours and the far-stack stations.
- audit: a planted afterslip event on a real fault and, at one station,
  the planted signal against what survives two filters.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnssdl import ngl
from gnssdl.bench.paths import scores_path
from gnssdl.dataset import Cube, crop_days, distance_azimuth
from gnssdl.plots import GRID, SURFACE, TEXT, TEXT_2

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 8.5, "axes.titlesize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.0, "legend.frameon": False, "legend.fontsize": 7.5,
}
COMPONENTS = ("East", "North", "Up")
RIDGECREST = pd.Timestamp("2019-07-06")
EL_MAYOR = pd.Timestamp("2010-04-04")


def _station(cube: Cube, sta: str) -> int:
    return int(np.flatnonzero(cube.sta == sta)[0])


# ---------------------------------------------------------------------------
# 1. raw data and what it becomes
# ---------------------------------------------------------------------------

def data_processing(cube: Cube, data_dir: Path, out: Path, sta: str = "BSRY",
                    zoom: tuple[str, str] = ("2021-01-01", "2021-12-31")) -> Path:
    from gnssdl.bench.audit import FarStackSweep
    i = _station(cube, sta)
    raw = ngl.read_tenv3(data_dir / "tenv3" / f"{sta}.tenv3")
    days = pd.DatetimeIndex(cube.days)
    raw = raw[~raw.index.duplicated()].reindex(days)
    xyz = raw[["e", "n", "u"]].to_numpy() * 1000.0
    first = np.flatnonzero(np.isfinite(xyz[:, 0]))[0]
    xyz = xyz - xyz[first]
    r = np.where(cube.avail[i][:, None], cube.r[i], np.nan)
    model = xyz - r                                           # trajectory + removed equipment steps
    steps = ngl.read_steps(data_dir / "meta" / "steps.txt")
    equip = sorted(d for d in set(steps[(steps.sta == sta) & (steps.kind == "equipment")].date)
                   if days[0] <= d <= days[-1])
    fs = FarStackSweep(radius_km=0.0)
    fs.fit(cube)
    z0, z1 = (int(days.searchsorted(pd.Timestamp(d))) for d in zoom)
    small = crop_days(cube, z0, z1 + 1)
    cm = fs.predict(small, np.zeros(small.avail.shape, dtype=bool))[i]
    zr = np.where(small.avail[i][:, None], small.r[i], np.nan)
    mean = np.nanmean(zr, axis=0)
    zr = zr - mean                                            # centred on the year's mean, for display
    cm = cm - np.nanmean(np.where(np.isfinite(zr), cm, np.nan), axis=0)
    zd = days[z0:z1 + 1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex="row")
        for c in range(3):
            a = ax[0, c]
            a.plot(days, xyz[:, c], ".", ms=0.8, color=GREY, rasterized=True)
            a.plot(days, model[:, c], color=BLUE, lw=0.9)
            a.set_title(COMPONENTS[c], loc="left")
            a = ax[1, c]
            a.plot(days, r[:, c], ".", ms=0.8, color=TEXT_2, rasterized=True)
            for d in equip:
                a.axvline(d, color=AQUA, lw=0.8, ls=":")
            a.axvline(RIDGECREST, color=ORANGE, lw=0.8, ls="--")
            a.axvline(EL_MAYOR, color=ORANGE, lw=0.8, ls="-.")
            a = ax[2, c]
            a.plot(zd, zr[:, c], ".", ms=2, color=GREY, label="residual")
            a.plot(zd, cm[:, c], color=BLUE, lw=1.0, label="common mode (far stack)")
            off = -5 * np.nanstd(zr[:, c])
            a.plot(zd, zr[:, c] - cm[:, c] + off, ".", ms=2, color=TEXT, label="cleaned (offset for display)")
            a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
            a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        for row, lab in enumerate(("position (mm)", "residual (mm)", f"{zoom[0][:4]}, mm")):
            ax[row, 0].set_ylabel(lab)
        for c in range(3):
            ax[0, c].xaxis.set_major_locator(mdates.YearLocator(5))
            ax[1, c].xaxis.set_major_locator(mdates.YearLocator(5))
        from matplotlib.lines import Line2D
        ax[0, 0].legend(handles=[Line2D([], [], ls="", marker="o", ms=3, color=GREY, label="daily position"),
                                 Line2D([], [], color=BLUE, label="fitted trajectory")], loc="upper right")
        h, lab = ax[2, 0].get_legend_handles_labels()
        h += [Line2D([], [], color=AQUA, ls=":"), Line2D([], [], color=ORANGE, ls="-."),
              Line2D([], [], color=ORANGE, ls="--")]
        lab += ["equipment change (removed)", "El Mayor-Cucapah 2010 (kept)", "Ridgecrest 2019 (kept)"]
        fig.legend(h, lab, loc="lower center", ncol=3, markerscale=3, bbox_to_anchor=(0.5, -0.01))
        for row, letter in enumerate("abc"):
            ax[row, 0].text(-0.32, 1.06, f"({letter})", transform=ax[row, 0].transAxes, fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        path = out / "data_processing.pdf"
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. the network and one target's neighbourhood
# ---------------------------------------------------------------------------

def network(cube: Cube, out: Path, sta: str = "BSRY", radius_km: float = 50.0, k: int = 16) -> list[Path]:
    """Two figures: the network (network_map.pdf, Section 2) and what one
    prediction may use at radius R (network_radius.pdf, Section 3)."""
    summary = json.loads(Path(str(Path("data/cube/california.json"))).read_text())
    dropped = summary["stations_dropped"]["qc"]
    i = _station(cube, sta)
    d, _ = distance_azimuth(cube.lat, cube.lon)
    ri = cube.radius_index(radius_km)
    nb = cube.nbr_idx[ri, i, :k]
    nb = nb[nb >= 0]
    held = cube.split_sta == 1
    with plt.rc_context(STYLE):
        fig_a, a = plt.subplots(figsize=(4.6, 4.6))
        fig_b, b = plt.subplots(figsize=(4.6, 4.2))
        for ax in (a, b):
            ax.set_aspect(1 / np.cos(np.deg2rad(36.5)))
            ax.grid(False)
        a.scatter(cube.lon[~held], cube.lat[~held], s=2, color=TEXT_2, lw=0, label=f"stations ({(~held).sum()})")
        a.scatter(cube.lon[held], cube.lat[held], s=9, facecolor="none", edgecolor=BLUE, lw=0.7,
                  label=f"held out ({held.sum()})")
        a.scatter([q["lon"] for q in dropped], [q["lat"] for q in dropped], s=12, marker="x", color=ORANGE,
                  lw=0.8, label=f"excluded by QC ({len(dropped)})")
        a.plot(cube.lon[i], cube.lat[i], "*", ms=9, color=TEXT, lw=0, label=f"{sta}")
        a.set_xlabel("longitude")
        a.set_ylabel("latitude")
        a.legend(loc="lower left", markerscale=1.5)
        # (b) the neighbourhood of one target
        lat0, lon0 = cube.lat[i], cube.lon[i]
        span = 2.6
        win = (np.abs(cube.lon - lon0) < span) & (np.abs(cube.lat - lat0) < span * 0.8)
        far = d[i] >= radius_km
        near = (d[i] < radius_km) & (np.arange(len(cube.sta)) != i)
        b.scatter(cube.lon[win & far], cube.lat[win & far], s=6, color="#9ec5f4", lw=0,
                  label="far stack: every station beyond R")
        b.scatter(cube.lon[win & near], cube.lat[win & near], s=6, color=GREY, lw=0,
                  label="excluded: closer than R")
        b.scatter(cube.lon[nb], cube.lat[nb], s=14, color=ORANGE, lw=0, label=f"{len(nb)} nearest beyond R (M1)")
        b.plot(lon0, lat0, "*", ms=11, color=TEXT, label=f"target ({sta})")
        th = np.linspace(0, 2 * np.pi, 200)
        b.plot(lon0 + radius_km / (111.2 * np.cos(np.deg2rad(lat0))) * np.cos(th),
               lat0 + radius_km / 111.2 * np.sin(th), color=TEXT, lw=0.8, ls="--")
        b.text(lon0 + 0.05, lat0 + radius_km / 111.2 + 0.05, f"R = {radius_km:g} km", fontsize=7.5)
        b.set_xlim(lon0 - span, lon0 + span)
        b.set_ylim(lat0 - span * 0.8, lat0 + span * 0.8)
        b.set_xlabel("longitude")
        b.set_ylabel("latitude")
        b.legend(loc="lower left", fontsize=7, markerscale=1.3)
        paths = []
        for fig, name in ((fig_a, "network_map.pdf"), (fig_b, "network_radius.pdf")):
            fig.tight_layout()
            fig.savefig(out / name, dpi=300)
            plt.close(fig)
            paths.append(out / name)
    return paths


# ---------------------------------------------------------------------------
# 3. the audit: a planted event and what two filters keep of it
# ---------------------------------------------------------------------------

def audit(cube: Cube, results_dir: Path, out: Path, seed: int = 3) -> Path:
    from gnssdl.bench import event_signal as es
    from gnssdl.bench import events
    from gnssdl.bench.audit import FarStackSweep
    from gnssdl.bench.m1_stack import M1Stack
    tp = next(t for t in es.load_templates() if t.name == "afterslip")
    cat = events.fault_catalogue()
    rng = np.random.default_rng(seed)
    while True:
        ev = events.random_event(tp, cat, rng, cube.lon, cube.lat)
        peak = np.hypot(*ev.final_mm()[:, :2].T)
        if (peak > 1).sum() >= 25 and peak.max() > 15:
            break
    # a station 15-40 km from the patch with a clear signal
    view = events.LocalFrame(*ev.centre)
    tris = events.fault_triangles(ev.meta["fault"], view, ev.tri).mean(1)[:, :2]
    x, y = view.xy(cube.lon, cube.lat)
    dist = np.sqrt(((np.c_[x, y][:, None] - tris[None]) ** 2).sum(-1)).min(1) / 1000
    val = np.flatnonzero(cube.split_day == 1)
    cover = cube.avail[:, val[0]:val[0] + 400].mean(axis=1)
    cand = np.flatnonzero((dist > 15) & (dist < 40) & (peak > 2) & (cover > 0.9))
    t0 = int(val[60])
    small = crop_days(cube, int(val[0]), int(val[-1]) + 1)
    t0 -= int(val[0])
    window = int(ev.duration_days) + 60
    n = small.r.shape[1] - t0
    s = np.zeros_like(small.r)
    m = min(n, window)
    s[:, t0:t0 + m] = ev.displacement_mm(np.arange(m))
    s[:, t0 + m:] = ev.final_mm()[:, None, :]
    planted = replace(small, r=np.where(small.avail[..., None], small.r + s, small.r))
    m1 = M1Stack(radius_km=0.0, length_km=json.loads(scores_path("m1", results_dir, ".json").read_text())[0]["length_km"])
    m1.fit(small)
    fs = FarStackSweep(radius_km=0.0)
    fs.fit(small)
    hide = np.zeros(small.avail.shape, dtype=bool)
    deltas = {name: (planted.r - f.predict(planted, hide)) - (small.r - f.predict(small, hide))
              for name, f in (("far stack", fs), ("M1 (16 nearest)", m1))}
    # the example station: the candidate whose M1 retention is the median, so
    # that it is representative of the 15-40 km band
    ok_all = small.avail[cand]
    rho_m1 = (np.where(ok_all[..., None], deltas["M1 (16 nearest)"][cand, :, :2] * s[cand, :, :2], 0).sum((1, 2))
              / np.where(ok_all[..., None], s[cand, :, :2] ** 2, 0).sum((1, 2)))
    j = int(cand[np.argsort(rho_m1)[len(rho_m1) // 2]])
    u = ev.final_mm()[j, :2] / np.linalg.norm(ev.final_mm()[j, :2])   # direction of the planted motion
    curves = {name: dl[j, :, :2] @ u for name, dl in deltas.items()}
    truth = s[j, :, :2] @ u
    self_note = f"median of {len(cand)} stations 15-40 km away"
    dd = pd.DatetimeIndex(small.days)
    ok = small.avail[j]
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.2]})
        lon0, lat0 = ev.centre
        win = (np.abs(cube.lon - lon0) < 1.0) & (np.abs(cube.lat - lat0) < 0.8)
        fault = ev.meta["fault"]
        for t in fault.tri[ev.tri]:
            a.fill(fault.lon[t], fault.lat[t], color=ORANGE, alpha=0.35, lw=0)
        a.scatter(cube.lon[win], cube.lat[win], s=4, color=TEXT_2, lw=0)
        big = win & (peak > 0.5)
        U = ev.final_mm()
        q = a.quiver(cube.lon[big], cube.lat[big], U[big, 0], U[big, 1], color=BLUE, scale=150, width=0.005)
        a.quiverkey(q, 0.15, 0.06, 20, "20 mm", labelpos="E", color=BLUE)
        a.plot(cube.lon[j], cube.lat[j], "o", ms=7, mfc="none", mec=TEXT, mew=1.2)
        a.set_xlim(lon0 - 1.0, lon0 + 1.0)
        a.set_ylim(lat0 - 0.8, lat0 + 0.8)
        a.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        a.grid(False)
        a.set_xlabel("longitude")
        a.set_ylabel("latitude")
        a.set_title("(a) planted afterslip, final motion", loc="left")
        b.plot(dd, np.where(ok, truth, np.nan), color=TEXT, lw=1.6, ls="--", label="planted signal $s$")
        for (name, cv), col in zip(curves.items(), (BLUE, ORANGE)):
            rho = float(np.sum(cv[ok] * truth[ok]) / np.sum(truth[ok] ** 2))
            b.plot(dd, np.where(ok, cv, np.nan), color=col, lw=1.0, label=fr"kept by {name}, $\rho$ = {rho:.2f}")
        b.set_ylabel("displacement along the planted motion (mm)")
        b.set_title(f"(b) station {cube.sta[j]}, {dist[j]:.0f} km from the slip", loc="left")
        print(f"audit figure: station {cube.sta[j]}, {self_note}")
        b.legend(loc="upper left")
        b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.tight_layout()
        path = out / "audit.pdf"
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return path


def all_figures(cube_path: Path, data_dir: Path, results_dir: Path, out: Path,
                period: str = "validation") -> list[Path]:
    """The data figures, and the results figures from `period`'s results
    (validation for now; test for the paper, after the method freeze)."""
    out.mkdir(parents=True, exist_ok=True)
    cube = Cube.load(cube_path)
    return [data_processing(cube, data_dir, out), *network(cube, out), audit(cube, results_dir, out),
            tradeoff(results_dir, out, period), count_vs_distance(cube, data_dir, results_dir, out, period),
            distance_from_fault(results_dir, out, period), planted_vs_real(results_dir, out, period)]


# ---------------------------------------------------------------------------
# 4-7. results figures, read from the results files of one period
# ---------------------------------------------------------------------------

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FAMILY = {  # filter -> (family colour, display name)
    "c1": (GREY, "C1"), "m2self": (GREY, "M2-self"), "fs": (BLUE, "far stack"), "d1": (BLUE, "D1"),
    "m1": (ORANGE, "M1"), "m1k64": (ORANGE, "M1, 64"), "m1k256": (ORANGE, "M1, 256"), "m2": (ORANGE, "M2"),
    "m3a": (ORANGE, "M3a"), "m3b": (ORANGE, "M3b"), "m4a": (ORANGE, "M4a"), "m4b": (ORANGE, "M4b"),
    "m3k": (AQUA, "M3k"), "m3k256": (AQUA, "M3k, 256"), "m5": (AQUA, "M5"), "m5aug": (AQUA, "M5-aug"),
}


def _signal_tables(results_dir: Path, period: str) -> dict[str, pd.DataFrame]:
    from gnssdl.bench.paths import SIGNAL_KINDS, signal_files
    out = {}
    for kind in SIGNAL_KINDS:
        files = signal_files(period, kind, results_dir)
        if kind == "ridgecrest" and not files:                 # computed once, the same for every period
            files = signal_files("validation", kind, results_dir)
        out[kind] = pd.concat([pd.read_csv(p) for p in files], ignore_index=True) if files else pd.DataFrame()
        if not out[kind].empty:
            out[kind]["R"] = out[kind]["radius_km"].fillna(0.0)
    return out


def _point(t: dict, model: str, R: float = 0.0) -> dict:
    """Summary numbers of one filter at one radius (NaN where not run)."""
    def med(df, col, **eq):
        if df.empty:
            return np.nan
        m = (df.model == model) & np.isclose(df.R, R)
        for k, v in eq.items():
            m &= df[k] == v
        return float(df.loc[m, col].median()) if m.any() else np.nan
    st = t["event_stations"]
    far = np.nan
    if not st.empty:
        m = (st.model == model) & np.isclose(st.R, R) & (st.template == "afterslip") & st.dist_km.between(15, 40)
        far = float(st.loc[m, "rho"].median()) if m.any() else np.nan
    return {"noise": med(t["noise"], "noise_removed"), "k25": med(t["injections"], "rho", footprint_km=25.0),
            "k100": med(t["injections"], "rho", footprint_km=100.0),
            "fake100": med(t["injections"], "spurious_max", footprint_km=100.0), "afterslip_far": far,
            "offsets": med(t["offsets"], "kept"), "ridgecrest": med(t["ridgecrest"], "retained")}


def _labels(ax, pts, dx=0.05, dy=0.05, fontsize=6.5) -> None:
    """Label points; points closer than (dx, dy) in axis-range fractions share
    one label listing their names, so clusters stay readable."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    todo = [p for p in pts if np.isfinite(p[1]) and np.isfinite(p[2])]
    while todo:
        name, x, y = todo.pop(0)
        group = [name]
        rest = []
        for q in todo:
            if abs(q[1] - x) / (x1 - x0) < dx and abs(q[2] - y) / (y1 - y0) < dy:
                group.append(q[0])
            else:
                rest.append(q)
        todo = rest
        ax.annotate(", ".join(group), (x, y), textcoords="offset points", xytext=(5, 3), fontsize=fontsize)


def tradeoff(results_dir: Path, out: Path, period: str) -> Path:
    t = _signal_tables(results_dir, period)
    sweeps = {"fs": (0, 25, 50, 100, 200, 400, 600), "m1": (0, 10, 25, 50, 100, 200, 400),
              "m1k256": (0, 100, 400), "m3k256": (0, 100, 400),
              "m5": (0, 10, 25, 50, 100), "m5aug": (0, 25)}
    singles = ["c1", "m2self", "m2", "m3a", "m3b", "m4a", "m4b", "m3k"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True)
        for ax, key, width in ((axes[0], "k25", 25), (axes[1], "k100", 100)):
            labs = []
            for m, radii in sweeps.items():
                col, name = FAMILY[m]
                pts = [(_point(t, m, R)["noise"] * 100, _point(t, m, R)[key]) for R in radii]
                pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                hollow = m in ("m5", "m5aug")
                ax.plot(xs, ys, "--" if m == "m5aug" else "-", color=col, lw=0.9, alpha=0.7)
                ax.plot(xs[0], ys[0], "o", color=col, ms=5, mfc="white" if hollow else col)
                labs.append((name, xs[0], ys[0]))
            for m in singles:
                col, name = FAMILY[m]
                p = _point(t, m)
                if np.isfinite(p["noise"]) and np.isfinite(p[key]):
                    ax.plot(p["noise"] * 100, p[key], "s" if col == GREY else "o", color=col, ms=5,
                            mfc="white" if col == AQUA and m in ("m5", "m5aug") else col)
                    labs.append((name, p["noise"] * 100, p[key]))
            c3 = t["injections"][t["injections"].model.str.startswith("c3")] if not t["injections"].empty else None
            if c3 is not None and len(c3):
                nz = t["noise"].set_index("model").noise_removed
                for lab, g in c3[c3.footprint_km == width].groupby("model"):
                    ax.plot(nz.get(lab, np.nan) * 100, g.rho.median(), "s", color=GREY, ms=3, alpha=0.6)
            ax.set_title(f"({'ab'[key == 'k100']}) {width}-km transients", loc="left")
            ax.set_xlabel("noise removed (%)")
            ax.set_ylim(-0.05, 1.08)
            _labels(ax, labs, dx=0.08, dy=0.06)
        axes[0].set_ylabel("signal kept (median $\\rho$)")
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], ls="", marker="s", color=GREY, label="target included (C1, C3, M2-self)"),
                            Line2D([], [], marker="o", color=BLUE, label="far stack (line: radius 0 to 600 km)"),
                            Line2D([], [], marker="o", color=ORANGE, label="nearest stations (line: radius)"),
                            Line2D([], [], marker="o", color=AQUA, label="learned (line: radius)")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        path = out / "tradeoff.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def loo_error(model, cube: Cube, ctx, split: int) -> float:
    """Median-over-stations fast leave-one-out error on every available day
    of `split` at seen, scored stations (the harness's tuning measure)."""
    from gnssdl.bench.score import per_station_scores, summarise
    days = (cube.split_day == split)[None, :]
    cells = cube.avail & days & ~cube.exclude & (ctx.keep_sta & (cube.split_sta == 0))[:, None]
    pred = model.predict(cube, np.zeros(cube.avail.shape, dtype=bool))
    return float(summarise(per_station_scores(cube, pred, cells, ctx.scale, ctx.skip))["nrmse_fast"])


def count_vs_distance(cube: Cube, data_dir: Path, results_dir: Path, out: Path, period: str) -> Path:
    from gnssdl.bench.run import ScoringContext, load_model
    from gnssdl.bench.signal import PERIODS
    ctx = ScoringContext.build(cube, ngl.read_steps(data_dir / "meta" / "steps.txt"))
    t = _signal_tables(results_dir, period)
    lines = {"m1": ("16 nearest", (0, 10, 25, 50, 100, 200, 400)),
             "m1k64": ("64 nearest", (0, 100, 400)),
             "m1k256": ("256 nearest", (0, 100, 400)),
             "fs": ("all beyond R (far stack)", (0, 25, 50, 100, 200, 400))}
    shades = {"m1": "#f2a988", "m1k64": "#eb6834", "m1k256": "#a63d12", "fs": BLUE}
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3))
        for m, (lab, radii) in lines.items():
            err = [loo_error(load_model(m, R, cube, results_dir, ctx=ctx), cube, ctx, PERIODS[period]) for R in radii]
            a.plot(radii, err, "o-", color=shades[m], ms=3.5, label=lab)
            kept = [(R, _point(t, m, R)["k25"]) for R in radii]
            kept = [(R, k) for R, k in kept if np.isfinite(k)]
            if kept:
                b.plot(*zip(*kept), "o-", color=shades[m], ms=3.5, label=lab)
        a.set_xlabel("exclusion radius R (km)")
        a.set_ylabel("fast reconstruction error")
        a.set_title("(a) error (lower is better)", loc="left")
        b.set_xlabel("exclusion radius R (km)")
        b.set_ylabel("signal kept, 25-km transients")
        b.set_title("(b) signal kept", loc="left")
        b.set_ylim(-0.05, 1.05)
        a.legend(loc="upper left")
        fig.tight_layout()
        path = out / "count_vs_distance.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def distance_from_fault(results_dir: Path, out: Path, period: str) -> Path:
    t = _signal_tables(results_dir, period)
    st = t["event_stations"]
    st = st[(st.template == "afterslip") & np.isclose(st.R, 0.0)]
    bins = [0, 5, 10, 20, 40, 80, 400]
    mids = np.sqrt(np.array(bins[:-1]).clip(1) * np.array(bins[1:]))
    show = {"fs": BLUE, "m5aug": AQUA, "m5": "#7fd3b3", "m3k256": "#0f7553", "m3k": "#0b4f38",
            "m1k256": "#f2a988", "m4a": ORANGE, "m1": "#a63d12"}
    styles = {"m5": "--", "m3k": ":", "m1": ":"}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for m, col in show.items():
            g = st[st.model == m]
            if g.empty:
                continue
            med = g.groupby(pd.cut(g.dist_km, bins), observed=False).rho.median().to_numpy()
            ax.plot(mids, med, marker="o", ls=styles.get(m, "-"), color=col, ms=3.5, lw=1.3, label=FAMILY[m][1])
        ax.axhline(1, color=TEXT_2, lw=0.6, ls=":")
        ax.axhline(0, color=TEXT_2, lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("distance from the slipping patch (km)")
        ax.set_ylabel("afterslip kept (median $\\rho$ per station)")
        ax.legend(loc="lower left", fontsize=7, ncol=2)
        fig.tight_layout()
        path = out / "distance_from_fault.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def spearman(x, y) -> float:
    """Spearman rank correlation (no SciPy dependency)."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def planted_vs_real(results_dir: Path, out: Path, period: str) -> Path:
    t = _signal_tables(results_dir, period)
    models = ["c1", "m2self", "fs", "d1", "m1", "m1k256", "m2", "m3a", "m3b", "m4a", "m4b", "m3k", "m3k256", "m5", "m5aug"]
    pts = []
    for m in models:
        R = 400.0 if m == "d1" else 0.0
        p = _point(t, m, R)
        pts.append((m, p["k25"], p["offsets"], p["ridgecrest"]))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True)
        for ax, idx, lab in ((axes[0], 2, "real earthquake offsets kept"), (axes[1], 3, "Ridgecrest postseismic kept")):
            xs, ys, labs = [], [], []
            for m, x, *ys_ in pts:
                y = ys_[idx - 2]
                if np.isfinite(x) and np.isfinite(y):
                    col, name = FAMILY[m]
                    ax.plot(x, y, "s" if col == GREY else "o", color=col, ms=5)
                    labs.append((name, x, y))
                    xs.append(x)
                    ys.append(y)
            ax.plot([0, 1], [0, 1], color=TEXT_2, lw=0.6, ls=":")
            ax.set_xlabel("planted signal kept, 25-km transients")
            ax.set_ylabel(lab)
            ax.set_title(f"({'ab'[idx - 2]}) Spearman rank correlation {spearman(xs, ys):.2f}", loc="left", fontsize=8)
            ax.set_xlim(-0.05, 1.12)
            ax.set_ylim(0.2, 1.05)
            _labels(ax, labs, dx=0.06, dy=0.05, fontsize=6)
        fig.tight_layout()
        path = out / "planted_vs_real.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path
