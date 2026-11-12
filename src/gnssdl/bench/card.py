"""`audit`: one call that runs a filter through the benchmark and returns its report card.

    import gnssdl

    net = gnssdl.Network.load("pnw")
    card = gnssdl.audit(MyFilter, net)        # a registered name or a Reconstructor subclass
    print(card)                               # every measure with its 95% interval, and the verdict
    card.table(); card.verdict(); card.measures

It does, in order, what the `gnssdl bench` commands do by hand: the masks if
the network has none, fit and leak checks (`bench run`), the signal tests
(`bench signal`), the same for the reference filter if it has no results yet,
then the intervals and the success criterion (contract §1.6) for this filter
against the reference. Results are written under the network's results root
in the usual files, so a second call reuses them: only what is missing runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnssdl.bench import registry, stats
from gnssdl.bench import signal as bsig
from gnssdl.bench import tasks
from gnssdl.bench.base import Reconstructor
from gnssdl.bench.freeze import require
from gnssdl.bench.paths import scores_path, signal_path
from gnssdl.bench.run import model_radii, seeded_label, unfitted
from gnssdl.bench.tasks import Log, Network, TaskError, _quiet

# the signal tests `audit` runs by default, in this order (keys of tasks.MEASURES)
AUDIT_MEASURES = ("transients", "seasonal", "offsets", "episodes")


@dataclass
class ReportCard:
    """What `audit` returns.

    measures   one row per model, radius, measure and (planted transients)
               footprint: value, ci_low, ci_high (95% bootstrap), n. Measures:
               noise_removed, error, kept, spurious, seasonal.
    verdicts   the success criterion for the filter against the reference
               (`stats.criterion_table`); None when no planted-transient test ran
    offsets    real earthquake offsets kept: median, number of offsets and events
    episodes   slow-slip episodes kept, per timing group; both are None when
               the network has none in the period
    """
    filter: str
    period: str
    results: Path
    reference: tuple[str, float]
    measures: pd.DataFrame
    verdicts: pd.DataFrame | None
    offsets: pd.DataFrame | None
    episodes: pd.DataFrame | None

    def table(self) -> pd.DataFrame:
        """One row per model and radius: every measure's value (planted
        transients at the criterion's 100-km footprint), offsets and episodes
        kept where the network has them, and the verdict."""
        m = self.measures
        m = m[m.footprint_km.isna() | np.isclose(m.footprint_km, stats.CRITERION_FOOTPRINT_KM)]
        key = ["model", "radius_km"]
        wide = (m.assign(radius_km=m.radius_km.fillna(-1.0))
                .pivot_table(index=key, columns="measure", values="value", aggfunc="first"))
        wide = wide.reindex(columns=[c for c in ("noise_removed", "error", "kept", "spurious", "seasonal")
                                     if c in wide.columns])
        wide = wide.rename(columns={"kept": "kept_100", "spurious": "spurious_100"})
        for name, extra, col in (("offsets", self.offsets, "kept"), ("episodes", self.episodes, "kept")):
            if extra is not None and len(extra):
                e = extra.assign(radius_km=extra.radius_km.fillna(-1.0))
                if name == "episodes":
                    e = e.pivot_table(index=key, columns="timing", values=col, aggfunc="first")
                    e.columns = [f"episodes_{t}" for t in e.columns]
                    wide = wide.join(e)
                else:
                    wide = wide.join(e.set_index(key)[col].rename(name))
        if self.verdicts is not None:
            v = self.verdicts.assign(radius_km=self.verdicts.radius_km.fillna(-1.0)).set_index(key)
            wide = wide.join(v["verdict"])
        wide = wide.reset_index()
        wide["radius_km"] = wide.radius_km.where(wide.radius_km >= 0)
        return wide

    def verdict(self) -> str | dict[tuple[str, float | None], str]:
        """The filter's verdict against the reference: one word for a filter
        run at one point, otherwise {(label, radius): verdict}."""
        if self.verdicts is None:
            return "no signal test"
        mine = self.verdicts[self.verdicts.verdict != "reference"]
        if not len(mine):                       # the filter audited is the reference itself
            return "reference"
        if len(mine) == 1:
            return str(mine.verdict.iloc[0])
        return {(m, None if pd.isna(r) else float(r)): v for m, r, v in zip(mine.model, mine.radius_km, mine.verdict)}

    def __str__(self) -> str:
        ref = f"{self.reference[0]} at R = {self.reference[1]:g} km"
        m = self.measures
        m = m[m.footprint_km.isna() | np.isclose(m.footprint_km, stats.CRITERION_FOOTPRINT_KM)]
        text = m.assign(radius_km=m.radius_km.map(lambda r: "" if pd.isna(r) else f"{r:g}"),
                        cell=[f"{v:.3f} [{a:.3f}, {b:.3f}]" for v, a, b in zip(m.value, m.ci_low, m.ci_high)])
        show = text.pivot_table(index=["model", "radius_km"], columns="measure", values="cell", aggfunc="first")
        show = show[[c for c in ("noise_removed", "error", "kept", "spurious", "seasonal") if c in show.columns]]
        lines = [f"{self.filter}, {self.period} period; reference: {ref}",
                 f"value [95% interval]; kept and spurious for {stats.CRITERION_FOOTPRINT_KM:g}-km planted transients",
                 show.fillna("").to_string()]
        with pd.option_context("display.width", 200):
            if self.offsets is not None and len(self.offsets):
                lines += ["", "real earthquake offsets kept (median over station-offsets)",
                          self.offsets.round(2).to_string(index=False)]
            if self.episodes is not None and len(self.episodes):
                lines += ["", "slow-slip episodes kept (median over counted episodes)",
                          self.episodes.round(2).to_string(index=False)]
            if self.verdicts is not None:
                cols = ["model", "radius_km", "noise_removed", "diff", "ci_low", "ci_high", "kept_100",
                        "spurious_100", "verdict"]
                lines += ["", f"success criterion against {ref} (diff: noise removed minus the reference's)",
                          self.verdicts[cols].round(3).to_string(index=False)]
        return "\n".join(lines)


def _resolve(filter) -> registry.FilterSpec:
    """A registered name, or a Reconstructor subclass (registered on the way)."""
    if isinstance(filter, str):
        return registry.get(filter)
    if isinstance(filter, type) and issubclass(filter, Reconstructor):
        known = registry.find(filter.name)
        if known is None:
            return registry.register(filter)
        if known.cls is not filter:
            raise TaskError(f"another filter is already registered as {filter.name!r}; give this one its own "
                            "`name` (results are stored under it)")
        return known
    raise TypeError("filter: a registered name or a subclass of gnssdl.bench.base.Reconstructor")


def _has_rows(path: Path, label: str, radius: float, free: bool) -> bool:
    """Whether a results file already has rows of this label at this radius."""
    if not path.exists():
        return False
    df = pd.read_csv(path, usecols=["model", "radius_km"])
    mine = df[df.model == label]
    return bool(len(mine) if free else np.isclose(mine.radius_km, radius).any())


def _missing(net: Network, name: str, period: str, measure: str, points, free: bool) -> list[float]:
    kinds = tasks.MEASURES[measure]
    return sorted({r for label, r, _ in points
                   if not all(_has_rows(signal_path(name, period, k, net.results), label, r, free) for k in kinds)})


def audit(filter, network: Network | str | Path, period: str = "validation",
          radii: list[float] | tuple[float, ...] | str = (0.0,), measures: tuple[str, ...] = AUDIT_MEASURES,
          seed: int | None = None, centres: int | None = None, rerun: bool = False, train: bool = False,
          log: Log = _quiet) -> ReportCard:
    """Run `filter` through the benchmark on `network` and return its report card.

    filter    a registered name ("fs", "m1", ...) or a Reconstructor subclass.
              It must either never read the target station (`never_reads_target
              = True`, verified by the leak checks) or declare that it does
              (`reference_filter = True`); the noise and signal measures mean
              different things for the two, and the card keeps them apart.
    network   a `Network`, a network name or a cube path
    radii     exclusion radii in km (default: 0 only, where the criterion's
              reference sits), or "all" for every radius the filter runs at
    measures  signal tests to run, from tasks.MEASURES
    seed      training seed of a learned filter
    centres   planted transients per grid cell (default: the benchmark's 50;
              a small number makes a quick, noisier card). The reference is
              planted with the same number when it has to be run.
    rerun     recompute results that are already saved
    train     allow training a learned filter that has no checkpoint (hours on
              a GPU); by default that is refused with a message
    """
    require(period)
    net = network if isinstance(network, Network) else Network.load(network)
    spec = _resolve(filter)
    name, cls = spec.name, spec.cls
    if not (cls.never_reads_target or cls.reference_filter):
        raise TaskError(f"{name}: set `never_reads_target = True` if its prediction for a station never uses that "
                        "station's own values (the leak checks verify it), or `reference_filter = True` if it "
                        "reads the target on purpose")
    unknown = [m for m in measures if m not in tasks.MEASURES]
    if unknown:
        raise ValueError(f"unknown measures {unknown}; choose from {sorted(tasks.MEASURES)}")
    tasks.check_seed(name, seed)
    asked = None if isinstance(radii, str) and radii == "all" else [float(r) for r in radii]
    run_radii = model_radii(name, net.cube, asked)
    free = spec.radius_free

    # 1. fit, tune and leak-check on the masks (not for filters that read the target)
    if not spec.reads_target:
        if not net.masks_path.exists():
            log(f"no masks at {net.masks_path}: generating them")
            tasks.make_masks(net)
        label = seeded_label(name, seed)
        todo = [r for r in run_radii if rerun or not _has_rows(scores_path(label, net.results), label, r, free)]
        if todo:
            seed_kw = {} if seed is None else {"seed": seed}
            untrained = [r for r in todo if hasattr(cls, "_checkpoint") and unfitted(name, r, net.results, **seed_kw)]
            if untrained and not train:
                raise TaskError(f"{label} is a trained filter with no checkpoint at R = "
                                f"{', '.join(f'{r:g}' for r in untrained)} km. Training is a long job: run "
                                f"`gnssdl bench run {name}` yourself, or pass train=True")
            log(f"{label}: fit, leak checks and mask scores at R = {', '.join(f'{r:g}' for r in todo)} km")
            tasks.fit_and_score(net, name, todo, seed, reuse=True, log=log)

    # 2. the signal tests, for the filter and then for the reference where it has none
    ref_name, ref_radius = stats.REFERENCE
    points = tasks.signal_points(net, name, asked and run_radii, seed, log)
    for measure in measures:
        todo = run_radii if rerun else _missing(net, name, period, measure, points, free)
        if todo:
            tasks.run_signal(net, name, period, measure, todo, seed, centres=centres, log=log)
    if (name, run_radii) != (ref_name, [ref_radius]):
        ref_points = tasks.signal_points(net, ref_name, [ref_radius], None, log)
        todo = [m for m in measures if _missing(net, ref_name, period, m, ref_points, False)]
        if todo:
            log(f"the reference ({ref_name} at R = {ref_radius:g} km) has no {', '.join(todo)} results here: "
                "running it")
        for measure in todo:
            tasks.run_signal(net, ref_name, period, measure, [ref_radius], centres=centres, log=log)

    # 3. the card, from the saved results of this filter and the reference
    labels = {label for label, _, _ in points}

    def mine(kind: str, with_reference: bool = True) -> pd.DataFrame | None:
        df = tasks.load_signal(net.results, period, kind)
        if df is None:
            return None
        own = df.model.isin(labels) & (df.radius_km.isna() if free else
                                       df.radius_km.round(6).isin([round(r, 6) for r in run_radii]))
        ref = (df.model == ref_name) & np.isclose(df.radius_km, ref_radius) if with_reference else False
        df = df[own | ref]
        return df if len(df) else None

    noise, inj, sea = mine("noise_blocks"), mine("injections"), mine("seasonal")
    if noise is None and inj is None and sea is None:
        raise TaskError(f"{name}: no measure produced results; measures asked: {list(measures)}")
    table = stats.interval_table(noise, inj, sea)
    verdicts = None
    if noise is not None and inj is not None:
        try:
            verdicts = stats.criterion_table(noise, inj)
        except ValueError as err:                # no reference results (transients not among the measures)
            log(f"no verdict: {err}")
    return ReportCard(name, period, Path(net.results), stats.REFERENCE, table, verdicts,
                      _offsets(mine("offsets")), _episodes(mine("episodes")))


def _offsets(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    g = df.assign(radius_km=df.radius_km.fillna(-1.0)).groupby(["model", "radius_km"])
    t = g.agg(kept=("kept", "median"), offsets=("kept", "size"), events=("event", "nunique")).reset_index()
    t["radius_km"] = t.radius_km.where(t.radius_km >= 0)
    return t


def _episodes(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    rows = []
    for (model, radius), g in df.assign(radius_km=df.radius_km.fillna(-1.0)).groupby(["model", "radius_km"]):
        for r in bsig.episode_summary(g).itertuples(index=False):
            rows.append({"model": model, "radius_km": radius if radius >= 0 else np.nan, "timing": r.timing,
                         "kept": r.retained, "episodes": r.episodes, "stations": r.stations, "weak": r.weak})
    return pd.DataFrame(rows) if rows else None
