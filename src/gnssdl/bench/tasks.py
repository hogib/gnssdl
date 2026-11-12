"""What the benchmark does, one function per task, callable from Python.

The command line (`gnssdl bench ...`) parses arguments, calls these and prints;
nothing is computed there. Each function returns tables or the paths it wrote,
reports progress through `log` (a function of one string; silent by default)
and raises `TaskError` with a message that says what to do when it cannot run.

    from gnssdl.bench import tasks
    s = tasks.Network.load("data/cube/pnw/pnw.npz")
    tasks.run_signal(s, "fs", measure="seasonal", radii=[0.0])
    table, path = tasks.verdict(s.results)

Every function that takes a period refuses the test and prospective periods
before the freeze (contract §1.4), like the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gnssdl import ngl
from gnssdl.bench import signal as bsig
from gnssdl.bench import stats
from gnssdl.bench.freeze import require
from gnssdl.bench.paths import RUNS, scores_path, signal_files, signal_path
from gnssdl.bench import registry
from gnssdl.bench.run import (ScoringContext, load_model, model_radii, neighbour_report, run_model, save_results,
                              scored_at, seeded_label, unfitted, variant_label, write_merged)
from gnssdl.dataset import Cube, crop_days
from gnssdl.regions import Region

Log = Callable[[str], None]

# `run_signal(measure=...)`: what each one writes under <results>/<model>/signal/<period>/
MEASURES = {
    "transients": ("noise", "noise_blocks", "injections"),   # noise removed, then planted Gaussian transients
    "noise": ("noise", "noise_blocks"),
    "seasonal": ("seasonal",),
    "offsets": ("offsets",),
    "episodes": ("episodes",),
    "velocity": ("velocity",),
    "events": ("events", "event_stations"),                  # planted fault-slip events from the templates
}


class TaskError(Exception):
    """A task could not run; the message says why and what to do."""


def _quiet(_: str) -> None:
    pass


@dataclass
class Network:
    """One network and one results root: the cube, and, loaded when first
    used, NGL's step catalogue and the model-independent scoring context.

    `steps` normally comes from NGL's steps file under `data_dir` (downloaded
    if absent); assign `network.steps = table` before use to supply your own.
    `region` holds what is specific to the study area (gnssdl.regions), or is
    None for a network with no region file."""
    cube_path: Path
    data_dir: Path
    results: Path
    cube: Cube
    region: Region | None = None

    @classmethod
    def load(cls, cube: Path | str, data_dir: Path | str = "data", results: Path | str = RUNS,
             region: "Region | str | Path | None" = "auto") -> "Network":
        """`cube`: a cube file (.npz), or a network name, read from
        <data_dir>/cube/<name>/<name>.npz as `gnssdl build` writes it.
        `region`: "auto" (the region recorded with the cube, or the built-in
        one of the same name, or none), a region name, a region file, or None."""
        from gnssdl import regions
        path = Path(cube)
        if path.suffix != ".npz":
            path = Path(data_dir) / "cube" / str(cube) / f"{cube}.npz"
        if not path.exists():
            raise TaskError(f"no cube at {path}; build it with `gnssdl build` first")
        try:
            found = regions.for_cube(path) if isinstance(region, str) and region == "auto" else \
                None if region is None else regions.get(region)
        except regions.RegionError as err:
            raise TaskError(str(err)) from None
        return cls(path, Path(data_dir), Path(results), Cube.load(path), found)

    @cached_property
    def steps(self) -> pd.DataFrame:
        _, steps_path = ngl.fetch_metadata(self.data_dir)
        return ngl.read_steps(steps_path)

    @cached_property
    def ctx(self) -> ScoringContext:
        return ScoringContext.build(self.cube, self.steps)

    @property
    def masks_path(self) -> Path:
        return self.cube_path.with_name("masks.npz")


def no_episodes(s: Network) -> str:
    """Why the slow-slip episode measure cannot run on this network."""
    where = "this network has no region file" if s.region is None else \
        f"the region {s.region.name!r} has no [episodes] table"
    return f"no slow-slip episode list: {where}, so the episode measure is skipped (see gnssdl/regions)"


def check_seed(model: str, seed: int | None) -> None:
    """Only trained models have a training seed."""
    spec = registry.find(model)
    if seed is not None and (spec is None or not spec.has_seed):
        raise TaskError(f"{model} has no training seed")


def load_signal(results: Path, period: str, kind: str) -> pd.DataFrame | None:
    """Every model's saved `<kind>.csv` of a period in one table; None when there is none."""
    frames = [f for f in (pd.read_csv(p) for p in signal_files(period, kind, results)) if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else None


# ---------------------------------------------------------------- masks, fit and score

def make_masks(s: Network) -> tuple[dict[str, np.ndarray], Path]:
    """Generate and save the fixed evaluation masks beside the cube."""
    from gnssdl.bench import masks as bmasks
    m = bmasks.make_masks(s.cube)
    bmasks.save_masks(m, s.masks_path)
    return m, s.masks_path


def fit_and_score(s: Network, model: str, radii: list[float] | None = None, seed: int | None = None,
                  reuse: bool = False, period: str = "validation", log: Log = _quiet) -> Path:
    """Fit (or train) a model at each radius and score it on the masks of
    `period` ("validation", or "all" for every split after the freeze).
    Returns the scores file."""
    from gnssdl.bench import masks as bmasks
    require("test" if period == "all" else period)
    if not s.masks_path.exists():
        raise TaskError(f"no masks at {s.masks_path}; run `gnssdl bench masks` first")
    check_seed(model, seed)
    splits = ("validation",) if period == "validation" else ("validation", "test", "prospective")
    scores, configs = run_model(model, s.cube, bmasks.load_masks(s.masks_path), s.ctx, radii, log=log, seed=seed,
                                reuse=reuse, results_dir=s.results, splits=splits)
    label = seeded_label(model, seed)
    save_results(label, scores, configs, s.results)
    return scores_path(label, s.results)


# ---------------------------------------------------------------- signal tests

def signal_points(s: Network, model: str, radii: list[float] | None = None, seed: int | None = None,
                  log: Log = _quiet) -> list[tuple[str, float, dict]]:
    """(label, radius, settings) of every point a signal test runs at. Every
    radius is checked before any is run, so a missing checkpoint or tuning
    cannot stop a run partway through: asked-for radii that are missing raise,
    default radii that are missing are skipped."""
    if registry.get(model).sweep:
        # one labelled point per swept setting
        points = [(variant_label(model, kw), 0.0, kw) for kw in registry.get(model).sweep]
    else:
        # a non-default training seed loads that seed's checkpoint and labels its rows <model>-s<seed>
        seed_kw = {} if seed is None else {"seed": seed}
        points = [(seeded_label(model, seed), r, seed_kw) for r in model_radii(model, s.cube, radii)]
    missing = [(r, why) for _, r, kw in points if (why := unfitted(model, r, s.results, **kw))]
    if missing and radii:
        raise TaskError("\n".join([f"R={r:g}: {why}" for r, why in missing]
                                  + ["nothing run: drop these radii or train them first"]))
    if missing:
        log(f"skipping radii with no trained model: {', '.join(f'{r:g}' for r, _ in missing)} km")
        points = [p for p in points if p[1] not in {r for r, _ in missing}]
        if not points:
            raise TaskError(f"{model}: no radius has a trained model")
    return points


def run_signal(s: Network, model: str, period: str = "validation", measure: str = "transients",
               radii: list[float] | None = None, seed: int | None = None, centres: int | None = None,
               per_template: int | None = None, log: Log = _quiet) -> dict[str, Path]:
    """Run one signal test (a key of MEASURES) of a fitted model at each
    radius and write its rows. Rows are written as each radius finishes
    (`write_merged` replaces only that radius's rows), so an interrupted run
    keeps its results. Returns {kind: file} of what it writes."""
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {sorted(MEASURES)}, not {measure!r}")
    require(period)
    check_seed(model, seed)
    cube, steps, ctx = s.cube, s.steps, s.ctx
    points = signal_points(s, model, radii, seed, log)
    log(f"{period} period, {', '.join(f'{lab} R={r:g}' for lab, r, _ in points)}")
    free = registry.get(model).radius_free

    def out(kind: str) -> Path:
        path = signal_path(model, period, kind, s.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def labelled(df: pd.DataFrame, label: str, radius: float) -> pd.DataFrame:
        df.insert(0, "radius_km", np.nan if free else radius)
        df.insert(0, "model", label)
        return df

    def tag(label: str, radius: float) -> str:
        return label if free else f"R={radius:g}"

    def each(kind: str, compute, describe) -> dict[str, Path]:
        """compute(model, scored stations) -> rows; describe(rows) -> progress text"""
        for label, radius, settings in points:
            fitted = load_model(model, radius, cube, s.results, ctx=ctx, **settings)
            keep = scored_at(cube, ctx.keep_sta, radius)             # neighbour rule (contract §2.2)
            df = labelled(compute(fitted, keep), label, radius)
            write_merged(df, out(kind))
            log(f"  {tag(label, radius)}: {describe(df)}")
        return {kind: out(kind)}

    if measure == "events":
        return _run_events(s, model, period, points, per_template, free, out, log)
    if measure == "velocity":
        if period != "test":
            raise TaskError("--velocity compares 2022-23 with 2024-25, so it runs on the test period only")
        return each("velocity", lambda m, keep: bsig.velocity_repeatability(m, cube, keep, steps, ctx.skip),
                    lambda df: f"{len(df)} stations, median |dv| E/N/U "
                               f"{df[['dv_e', 'dv_n', 'dv_u']].abs().median().round(2).tolist()} mm/yr")
    if measure == "seasonal":
        return each("seasonal", lambda m, keep: bsig.seasonal_kept(m, cube, keep, steps, period),
                    lambda df: f"{len(df)} stations, median seasonal kept E/N/U "
                               f"{df[['rho_e', 'rho_n', 'rho_u']].median().round(2).tolist()}")
    if measure == "episodes":
        if s.region is None or not s.region.has_episodes:
            raise TaskError(no_episodes(s))
        band = s.region.load_band()
        episodes = bsig.episodes_in_period(cube, s.region.load_episodes(), period)
        log(f"{len(episodes)} episodes on {period} days ({episodes.timing.value_counts().to_dict()})")
        return each("episodes", lambda m, keep: bsig.episode_retention(m, cube, keep, episodes, band),
                    lambda df: f"{len(df)} station-episodes; "
                               + "; ".join(f"{r.timing}: {r.episodes} episodes counted ({r.weak} below the floor "
                                           f"rule), median kept {r.retained:.2f}"
                                           for r in bsig.episode_summary(df).itertuples()))
    if measure == "offsets":
        return each("offsets", lambda m, keep: bsig.quake_offsets(m, cube, keep, steps, period),
                    lambda df: f"{len(df)} offsets, median kept {df.kept.median():.2f}" if len(df)
                    else "no earthquake offsets at the scored stations in this period")

    # noise removed, then (measure "transients") planted Gaussian transients
    for label, radius, settings in points:
        fitted = load_model(model, radius, cube, s.results, ctx=ctx, **settings)
        keep = scored_at(cube, ctx.keep_sta, radius)                 # neighbour rule (contract §2.2)
        noise, blocks = bsig.noise_parts(fitted, cube, keep, period, scale=ctx.scale, skip=ctx.skip)
        write_merged(pd.DataFrame([{"model": label, "radius_km": np.nan if free else radius, **noise}]),
                     out("noise"))
        write_merged(labelled(blocks, label, radius), out("noise_blocks"))   # what the noise bootstrap reads
        log(f"  {tag(label, radius)}: noise removed {noise['noise_removed']:.3f}")
        if measure == "noise":
            continue
        runs = bsig.plan_injections(cube, keep, centres or bsig.CENTRES_PER_CELL, period=period,
                                    radius_km=radius, model=fitted)
        log(f"  {tag(label, radius)}: {sum(len(r) for r in runs)} injections in {len(runs)} runs")

        def progress(k: int, n: int, where: str = tag(label, radius)) -> None:
            if k % 20 == 0 or k == n:
                log(f"  {where}: run {k}/{n}")

        inj = bsig.injection_scores(fitted, cube, keep, runs, log=progress)
        write_merged(labelled(inj, label, radius), out("injections"))
    return {kind: out(kind) for kind in MEASURES[measure]}


def _run_events(s: Network, model: str, period: str, points, per_template: int | None, free: bool, out,
                log: Log) -> dict[str, Path]:
    from gnssdl.bench import event_signal as es
    from gnssdl.bench import events

    cube, ctx = s.cube, s.ctx
    templates = es.load_templates()
    if not templates:
        raise TaskError(f"no event templates in {es.EVENT_DIR}/*.toml; write them from the source papers first "
                        "(contract §5.5)")
    catalogue = events.fault_catalogue()
    log(f"{len(templates)} templates ({', '.join(t.name for t in templates)}), {len(catalogue)} CFM faults")
    for label, radius, settings in points:
        fitted = load_model(model, radius, cube, s.results, ctx=ctx, **settings)
        where = label if free else f"R={radius:g}"
        runs = es.plan_events(cube, ctx.keep_sta, templates, catalogue,
                              per_template=per_template or es.EVENTS_PER_TEMPLATE,
                              period=period, radius_km=radius, model=fitted)
        log(f"  {where}: {sum(len(r) for r in runs)} events in {len(runs)} runs")

        def progress(k: int, n: int, where: str = where) -> None:
            if k % 5 == 0 or k == n:
                log(f"  {where}: run {k}/{n}")

        ev_rows, sta_rows = es.event_scores(fitted, cube, ctx.keep_sta, runs, log=progress)
        for df in (ev_rows, sta_rows):
            df.insert(0, "radius_km", np.nan if free else radius)
            df.insert(0, "model", label)
        write_merged(ev_rows, out("events"))
        write_merged(sta_rows, out("event_stations"))
        log(f"  {where}: median rho by template {ev_rows.groupby('template').rho.median().round(2).to_dict()}")
    return {"events": out("events"), "event_stations": out("event_stations")}


# ---------------------------------------------------------------- tables from saved results

def signal_tables(results: Path, period: str = "validation") -> dict[str, pd.DataFrame]:
    """Summary tables of the saved signal tests, one row per model and radius
    (index: model, R_km; "-" for a model with no radius). Keys, each present
    only when its results exist: kept and spurious (columns: transient
    footprint in km), kept_by_amplitude (columns: amplitude in mm), seasonal,
    offsets, episodes, noise. Values are not rounded."""
    require(period)

    def load(kind: str) -> pd.DataFrame | None:
        df = load_signal(results, period, kind)
        if df is None:
            return None
        df["R_km"] = df["radius_km"].map(lambda r: "-" if pd.isna(r) else f"{r:g}")
        df["R_sort"] = df["radius_km"].fillna(-1)
        return df

    key = ["model", "R_sort", "R_km"]
    drop = lambda t: t.reset_index(level="R_sort", drop=True)
    tables: dict[str, pd.DataFrame] = {}
    df = load("injections")
    if df is not None:
        tables["kept"] = drop(df.pivot_table(index=key, columns="footprint_km", values="rho", aggfunc="median"))
        if "spurious_max" in df:
            tables["spurious"] = drop(df.pivot_table(index=key, columns="footprint_km", values="spurious_max",
                                                     aggfunc="median"))
        tables["kept_by_amplitude"] = drop(df.pivot_table(index=key, columns="amplitude_mm", values="rho",
                                                          aggfunc="median"))
    sea = load("seasonal")
    if sea is not None:
        t = sea.groupby(key)[["rho_e", "rho_n", "rho_u", "rho_h", "rho_all"]].median()
        t["stations"] = sea.groupby(key).size()
        tables["seasonal"] = drop(t)
    off = load("offsets")
    if off is not None:
        tables["offsets"] = drop(off.groupby(key).agg(kept=("kept", "median"), offsets=("kept", "size"),
                                                      events=("event", "nunique")))
    epi = load("episodes")
    if epi is not None:
        rows = []
        for k, g in epi.groupby(key):
            for r in bsig.episode_summary(g).itertuples(index=False):
                rows.append(dict(zip(key, k), timing=r.timing, episodes=r.episodes, stations=r.stations,
                                 kept=r.retained, weak=r.weak))
        t = pd.DataFrame(rows).pivot_table(index=key, columns="timing", values=["kept", "episodes", "weak"])
        for col in ("episodes", "weak"):
            t[col] = t[col].round().astype("Int64")
        tables["episodes"] = drop(t)
    noise = load("noise")
    if noise is not None:
        tables["noise"] = noise.sort_values(["model", "R_sort"]).set_index(["model", "R_km"])[["noise_removed"]]
    return tables


def verdict(results: Path, period: str = "validation", write: bool = True) -> tuple[pd.DataFrame, Path]:
    """The success criterion (contract §1.6) applied to every filter with saved
    noise blocks and planted-transient results. No filter is run. Returns the
    table and the file it is written to (<results>/verdict_<period>.csv)."""
    require(period)
    noise, inj = load_signal(results, period, "noise_blocks"), load_signal(results, period, "injections")
    if noise is None or inj is None:
        missing = "noise_blocks.csv (run `gnssdl bench signal MODEL --noise-only`)" if noise is None else \
                  "injections.csv (run `gnssdl bench signal MODEL`)"
        raise TaskError(f"no {period} {missing} under {results}")
    try:
        t = stats.criterion_table(noise, inj)
    except ValueError as err:
        raise TaskError(f"{err}; run `gnssdl bench signal fs --radius 0` first") from None
    path = Path(results) / f"verdict_{period}.csv"
    if write:
        t.to_csv(path, index=False)
    return t, path


def intervals(results: Path, period: str = "validation", write: bool = True,
              log: Log = _quiet) -> tuple[pd.DataFrame, Path]:
    """95% bootstrap interval of every measure of every filter, from saved
    results (noise_blocks.csv, injections.csv, seasonal.csv). Returns the
    table and the file it is written to (<results>/intervals_<period>.csv)."""
    require(period)
    kinds = ("noise_blocks", "injections", "seasonal")
    noise, inj, sea = (load_signal(results, period, kind) for kind in kinds)
    if noise is None and inj is None and sea is None:
        raise TaskError(f"no {period} signal results under {results}")
    for kind, df in zip(kinds, (noise, inj, sea)):
        if df is None:
            log(f"(no {kind}.csv: its measures are left out)")
    t = stats.interval_table(noise, inj, sea)
    path = Path(results) / f"intervals_{period}.csv"
    if write:
        t.to_csv(path, index=False)
    return t, path


def neighbours(s: Network, write: bool = True) -> tuple[pd.DataFrame, Path]:
    """Neighbours actually read per exclusion radius and neighbour count
    (contract §2.2). Returns the table and <results>/neighbours.csv."""
    t = neighbour_report(s.cube, s.ctx.keep_sta)
    path = Path(s.results) / "neighbours.csv"
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(path, index=False)
    return t, path


# ---------------------------------------------------------------- paired comparison

@dataclass
class Comparison:
    """`compare`'s result. `noise` is `stats.paired_bootstrap`'s record of the
    leave-one-out fast nrmse; `kept` the paired planted-transient table, or
    None with the reason in `kept_note`."""
    period: str
    stations: str
    label_a: str
    label_b: str
    noise: dict
    kept: pd.DataFrame | None
    kept_note: str | None


def compare(s: Network, model_a: str, model_b: str, radius: float = 0.0, radius_b: float | None = None,
            period: str = "validation", stations: str = "all") -> Comparison:
    """Paired block bootstrap of two fitted models over stations and 30-day
    blocks, and, where both have saved planted-transient results, the paired
    comparison of signal kept."""
    require(period)
    cube, ctx = s.cube, s.ctx
    code = bsig.PERIODS[period]
    days = np.flatnonzero(cube.split_day == code)
    lo = max(0, days[0] - bsig.CROP_MARGIN_DAYS)
    hi = min(cube.r.shape[1], days[-1] + bsig.CROP_MARGIN_DAYS + 1)
    small = crop_days(cube, lo, hi)
    radius_b = radius if radius_b is None else radius_b
    use_sta = scored_at(cube, ctx.keep_sta, radius) & scored_at(cube, ctx.keep_sta, radius_b)   # neighbour rule
    if stations != "all":
        use_sta &= small.split_sta == (0 if stations == "seen" else 1)
    cells = (small.avail & (small.split_day == code)[None, :] & ~small.exclude
             & ~ctx.skip[:, lo:hi] & use_sta[:, None])

    spec_a, spec_b = registry.get(model_a), registry.get(model_b)
    radius = radius if spec_a.fixed_radius is None else spec_a.fixed_radius
    radius_b = radius_b if spec_b.fixed_radius is None else spec_b.fixed_radius
    block = {}
    for name, r in ((model_a, radius), (model_b, radius_b)):
        fitted = load_model(name, r, cube, s.results)
        resid = small.r - bsig.clean(fitted, small)
        block[name, r] = stats.block_stats(small, resid, cells, ctx.scale)
    noise = stats.paired_bootstrap(block[model_a, radius], block[model_b, radius_b])
    label_a = f"{model_a}" + ("" if spec_a.radius_free else f" R={radius:g}")
    label_b = f"{model_b}" + ("" if spec_b.radius_free else f" R={radius_b:g}")

    kept, note = None, None
    fa = signal_path(model_a, period, "injections", s.results)
    fb = signal_path(model_b, period, "injections", s.results)
    if fa.exists() and fb.exists():
        da, db = pd.read_csv(fa), pd.read_csv(fb)
        pick = lambda d, name, r: d if registry.get(name).radius_free else d[d.radius_km == r]
        try:
            kept = stats.paired_rho_bootstrap(pick(da, model_a, radius), pick(db, model_b, radius_b))
        except ValueError as err:
            note = f"not compared ({err})"
    else:
        note = "no saved planted-transient results for both models"
    return Comparison(period, stations, label_a, label_b, noise, kept, note)
