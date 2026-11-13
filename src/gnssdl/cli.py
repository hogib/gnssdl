"""gnssdl command line.

    gnssdl meta                                   # cache station table + steps
    gnssdl near --lat 37.23 --lon 37.01 -r 300 --around 2023-02-06
    gnssdl fetch ANKR ISTA                        # explicit stations
    gnssdl fetch --lat 37.23 --lon 37.01 -r 300 --around 2023-02-06 --before 365 --after 180
    gnssdl departure ELAZ ERGN --fit 2009-01-01 2015-12-31 \
        --event 2020-01-24:"Elazığ Mw6.8" --event 2023-02-06:"Kahramanmaraş Mw7.8"
    gnssdl build --manifest data/manifests/pnw.csv          # benchmark data cube
    gnssdl bench masks                            # fixed evaluation masks
    gnssdl bench run m1                           # fit + score a model at every radius
    gnssdl bench report                           # scoreboard
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gnssdl import ngl


def _has_region(args) -> bool:
    if (args.lat is None) != (args.lon is None):
        raise SystemExit("gnssdl: --lat and --lon go together")
    if args.bbox and args.lat is not None:
        raise SystemExit("gnssdl: use either --bbox or --lat/--lon")
    return bool(args.bbox) or args.lat is not None


def _selection(args, holdings: pd.DataFrame) -> pd.DataFrame:
    start = end = None
    if args.around:
        t0 = pd.Timestamp(args.around)
        start = t0 - pd.Timedelta(days=args.before)
        end = t0 + pd.Timedelta(days=args.after)
    if args.bbox:
        sel = ngl.stations_in_bbox(holdings, *args.bbox)
        if start is not None:
            sel = sel[(sel["start"] <= start) & (sel["end"] >= end)]
    else:
        sel = ngl.stations_near(holdings, args.lat, args.lon, args.radius, start=start, end=end)
    return sel[sel["n_sol"] >= args.min_days].reset_index(drop=True)


def _add_region_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("-r", "--radius", type=float, default=200.0, help="km (default 200)")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("S", "N", "W", "E"),
                   help="lat/lon box instead of --lat/--lon")
    p.add_argument("--min-days", type=int, default=0, help="minimum number of daily solutions")
    p.add_argument("--around", help="event date YYYY-MM-DD; keep stations covering the window")
    p.add_argument("--before", type=int, default=365, help="days before --around (default 365)")
    p.add_argument("--after", type=int, default=90, help="days after --around (default 90)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gnssdl", description="Fetch NGL daily GNSS time series")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("meta", help="download station table and steps file")

    p_near = sub.add_parser("near", help="list stations near a point")
    _add_region_args(p_near)

    p_fetch = sub.add_parser("fetch", help="download daily .tenv3 series")
    p_fetch.add_argument("stations", nargs="*", help="4-char station ids")
    _add_region_args(p_fetch)
    p_fetch.add_argument("-j", "--workers", type=int, default=4)
    p_fetch.add_argument("--manifest", type=Path, help="write the selection + fetch status as CSV")

    p_dep = sub.add_parser("departure", help="plot departure from a reference-window trajectory")
    p_dep.add_argument("stations", nargs="+")
    p_dep.add_argument("--fit", nargs=2, required=True, metavar=("START", "END"))
    p_dep.add_argument("--event", action="append", default=[], metavar="DATE:LABEL")
    p_dep.add_argument("--xlim", nargs=2, metavar=("START", "END"))
    p_dep.add_argument("--title", default="Departure from reference trajectory")
    p_dep.add_argument("-o", "--out", type=Path, default=Path("figures/departure.png"))

    p_build = sub.add_parser("build", help="build the benchmark data cube from fetched series")
    p_build.add_argument("--manifest", type=Path, default=Path("data/manifests/pnw.csv"))
    p_build.add_argument("-o", "--out", type=Path, default=Path("data/cube/pnw/pnw.npz"),
                         help="cube path; the masks file is written beside it, so give each network its own directory")
    p_build.add_argument("--region", help="region of the cube: a built-in name, a region .toml file, or `none` "
                                          "(default: the built-in region named like the cube file, if any)")
    p_build.add_argument("--no-episode-exclusion", action="store_true",
                         help="do not bar the region's catalogued slow-slip episodes from training (contract 1.6)")
    p_build.add_argument("--end", help="last day of the cube (default: latest epoch)")
    p_build.add_argument("--freeze-date",
                         help="method freeze date; later days become the prospective test period")

    p_fig = sub.add_parser("figures", help="figures for the proposal and the findings report")
    p_fig.add_argument("--cube", type=Path, default=Path("data/cube/pnw/pnw.npz"))
    p_fig.add_argument("--results", type=Path, default=Path("data/runs"), help="runs root (default data/runs)")
    p_fig.add_argument("-o", "--out", type=Path, default=Path("paper/figures"))
    p_fig.add_argument("--only", choices=["all", "tradeoff"], default="all",
                       help="tradeoff: only the noise-against-signal curves per transient width, from "
                            "saved results of any network (the other figures are still California-specific)")
    p_fig.add_argument("--period", choices=["validation", "test", "prospective"], default="validation",
                       help="results period for the results figures (test for the paper)")
    p_queue = sub.add_parser("queue", help="a restartable queue of bench jobs described in a manifest file")
    p_queue.add_argument("action", choices=["plan", "run", "status"],
                         help="plan: list the jobs; run: run what is not done; status: where it stands")
    p_queue.add_argument("manifest", type=Path)
    p_bench = sub.add_parser("bench", help="benchmark: masks, run models, report")
    bsub = p_bench.add_subparsers(dest="bench_cmd", required=True)
    for bp in (bsub.add_parser("masks", help="generate the fixed evaluation masks"),
               bsub.add_parser("run", help="fit and score a model"),
               bsub.add_parser("report", help="print the scoreboard"),
               bsub.add_parser("signal", help="planted-transient, seasonal, offset and slow-slip episode tests"),
               bsub.add_parser("compare", help="paired bootstrap comparison of two models"),
               bsub.add_parser("verdict", help="apply the success criterion (contract §1.6) to every filter "
                                               "with saved noise blocks and planted-transient results"),
               bsub.add_parser("intervals", help="95% bootstrap interval of every measure of every filter, "
                                                 "from saved results"),
               bsub.add_parser("neighbours", help="neighbours actually read per exclusion radius and neighbour "
                                                  "count; flags combinations where most stations fall short"),
               bsub.add_parser("status", help="what exists under the results root: fitted filters and "
                                              "signal tests, per filter, seed and radius (reads files only)"),
               bsub.add_parser("kernel", help="plot a trained M3k kernel"),
               bsub.add_parser("episodes", help="write the fixed Cascadia slow-slip episode list "
                                                "from the GNSS catalogue file and a PNSN tremor export")):
        bp.add_argument("--cube", type=Path, default=Path("data/cube/pnw/pnw.npz"))
        bp.add_argument("--results", type=Path, default=Path("data/runs"),
                        help="runs root: one directory per model (default data/runs)")
    from gnssdl.bench import registry
    bsub.choices["run"].add_argument("model", choices=registry.names(masks=True))
    bsub.choices["run"].add_argument("--seed", type=int, default=None,
                                     help="training seed of a learned model (default 0)")
    bsub.choices["run"].add_argument("--period", choices=["validation", "all"], default="validation",
                                     help="mask scores to compute: validation only (default, until the freeze) or all splits")
    bsub.choices["run"].add_argument("--reuse", action="store_true",
                                     help="load a saved checkpoint instead of training, where one exists")
    bsub.choices["run"].add_argument("--radius", type=float, action="append",
                                     help="exclusion radius in km (repeatable; default: all)")
    bsub.choices["signal"].add_argument("model", choices=registry.names())
    bsub.choices["signal"].add_argument("--radius", type=float, action="append",
                                        help="exclusion radius in km (repeatable; default: all)")
    bsub.choices["signal"].add_argument("--seed", type=int, default=None,
                                        help="training seed of a learned model (default 0); results are "
                                             "labelled <model>-s<seed> in the same signal files")
    bsub.choices["signal"].add_argument("--centres", type=int, default=None,
                                        help="injections per grid cell (default 50)")
    bsub.choices["signal"].add_argument("--period", choices=["validation", "test", "prospective"],
                                        default="validation",
                                        help="period to plant signals in (design decisions: validation)")
    bsub.choices["signal"].add_argument("--noise-only", action="store_true",
                                        help="only recompute the noise-removed measure")
    bsub.choices["signal"].add_argument("--velocity", action="store_true",
                                        help="velocity repeatability (contract §5.2): MIDAS velocity of the "
                                             "cleaned series, 2022-23 vs 2024-25 (test period only)")
    bsub.choices["signal"].add_argument("--seasonal", action="store_true",
                                        help="seasonal kept: fit each station's annual and semi-annual harmonic "
                                             "to r and to r - r^ over the period and report the fraction kept")
    bsub.choices["signal"].add_argument("--offsets", action="store_true",
                                        help="only score real earthquake offsets from NGL's step catalogue")
    bsub.choices["signal"].add_argument("--episodes", action="store_true",
                                        help="only score the region's fixed slow-slip episodes (contract §1.6; "
                                             "refused for a network whose region has no episode list)")
    bsub.choices["episodes"].add_argument("--tremor", type=Path, required=True,
                                          help="PNSN tremor export csv (lat, lon, depth, starttime, ...)")
    bsub.choices["signal"].add_argument("--events", action="store_true",
                                        help="plant realistic fault-slip events from the event templates "
                                             "instead of Gaussian transients")
    bsub.choices["signal"].add_argument("--per-template", type=int, default=None,
                                        help="events per template (default 20)")
    bsub.choices["compare"].add_argument("model_a")
    bsub.choices["compare"].add_argument("model_b")
    bsub.choices["compare"].add_argument("--radius", type=float, default=0.0)
    bsub.choices["compare"].add_argument("--stations", choices=["all", "seen", "heldout"], default="all",
                                         help="score all stations (default), only those seen in training, "
                                              "or only the held-out stations")
    bsub.choices["compare"].add_argument("--radius-b", type=float, help="radius for model_b (default: --radius)")
    bsub.choices["compare"].add_argument("--period", choices=["validation", "test", "prospective"],
                                         default="validation")
    bsub.choices["verdict"].add_argument("--period", choices=["validation", "test", "prospective"],
                                         default="validation")
    bsub.choices["intervals"].add_argument("--period", choices=["validation", "test", "prospective"],
                                           default="validation")
    bsub.choices["status"].add_argument("--period", choices=["validation", "test", "prospective"],
                                        default="validation")
    bsub.choices["status"].add_argument("--missing", action="store_true",
                                        help="only the rows with something missing")
    bsub.choices["kernel"].add_argument("model", choices=["m3k", "m3k256"])
    bsub.choices["kernel"].add_argument("--radius", type=float, default=0.0,
                                        help="exclusion radius of the distance, direction and quality "
                                             "figures (default 0)")
    bsub.choices["kernel"].add_argument("--seed", type=int, default=0)
    bsub.choices["kernel"].add_argument("-o", "--out", type=Path, default=Path("figures/m3k"))
    bsub.choices["report"].add_argument("--split", choices=["validation", "test", "prospective"],
                                        default="validation")
    bsub.choices["report"].add_argument("--part", choices=["fast", "slow", "total", "all"], default="all")

    args = ap.parse_args(argv)
    # nothing touches the test or prospective period before the freeze (contract §1.4)
    from gnssdl.bench.freeze import NotFrozen, require
    asked = getattr(args, "split", None) or getattr(args, "period", None)
    if asked == "all":                       # `bench run --period all` scores the test masks too
        asked = "test"
    if asked in ("test", "prospective"):
        try:
            require(asked)
        except NotFrozen as err:
            print(err, file=sys.stderr)
            return 1
    if args.cmd == "queue":
        return _queue(args)
    if args.cmd == "bench":
        if args.bench_cmd in ("run", "signal"):
            with _command_log(args):
                return _bench(args)
        return _bench(args)
    if args.cmd == "figures" and args.only == "tradeoff":
        from gnssdl.paper_figures import tradeoff_widths
        try:
            print(f"wrote {tradeoff_widths(args.results, args.out, args.period)}")
        except ValueError as err:
            print(err, file=sys.stderr)
            return 1
        return 0
    if args.cmd == "figures":
        from gnssdl.paper_figures import all_figures
        for path in all_figures(args.cube, args.data_dir, args.results, args.out, args.period):
            print(f"wrote {path}")
        return 0
    holdings_path, steps_path = ngl.fetch_metadata(
        args.data_dir, refresh=args.refresh and args.cmd == "meta"
    )

    if args.cmd == "meta":
        h = ngl.read_holdings(holdings_path)
        s = ngl.read_steps(steps_path)
        print(f"{len(h)} stations, {len(s)} steps "
              f"({(s.kind == 'equipment').sum()} equipment, {(s.kind == 'quake').sum()} quake)")
        return 0

    holdings = ngl.read_holdings(holdings_path)

    if args.cmd == "near":
        if not _has_region(args):
            ap.error("give --lat/--lon or --bbox")
        sel = _selection(args, holdings)
        cols = [c for c in ("sta", "lat", "lon", "dist_km", "start", "end", "n_sol") if c in sel]
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(sel[cols].round({"lat": 3, "lon": 3, "dist_km": 1}).to_string())
        print(f"\n{len(sel)} stations")
        return 0

    if args.cmd == "departure":
        return _departure(args, holdings, ngl.read_steps(steps_path))

    if args.cmd == "build":
        return _build(args, ngl.read_steps(steps_path))

    # fetch
    stations = [s.upper() for s in args.stations]
    if _has_region(args):
        stations += _selection(args, holdings)["sta"].tolist()
    stations = list(dict.fromkeys(stations))
    if not stations:
        ap.error("give station ids or --lat/--lon")
    bad = [s for s in stations if not ngl.is_station_id(s)]
    if bad:
        ap.error(f"not 4-character station ids: {bad}")

    results = ngl.fetch_tenv3(stations, args.data_dir, refresh=args.refresh, workers=args.workers)
    failed = [r for r in results if r.error]
    for r in failed:
        print(f"FAIL {r.station}: {r.error}", file=sys.stderr)
    if args.manifest:
        status = pd.DataFrame({"sta": [r.station for r in results],
                               "ok": [r.error is None for r in results],
                               "error": [r.error or "" for r in results]})
        man = status.merge(holdings, on="sta", how="left")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        man.to_csv(args.manifest, index=False)
        print(f"manifest: {args.manifest}")
    print(f"{len(results) - len(failed)}/{len(results)} series in {args.data_dir / 'tenv3'}")
    return 1 if failed else 0


def _departure(args, holdings: pd.DataFrame, steps: pd.DataFrame) -> int:
    from gnssdl import model, plots

    events = []
    for ev in args.event:
        date, _, label = ev.partition(":")
        events.append((pd.Timestamp(date), label or date))
    stations = [s.upper() for s in args.stations]
    bad = [s for s in stations if not ngl.is_station_id(s)]
    if bad:
        print(f"not 4-character station ids: {bad}", file=sys.stderr)
        return 2
    ngl.fetch_tenv3(stations, args.data_dir)

    residuals, subtitles, corrected = {}, {}, {}
    for sta in stations:
        series = ngl.read_tenv3(args.data_dir / "tenv3" / f"{sta}.tenv3")
        mine = steps[steps.sta == sta]
        step_dates = sorted(set(mine["date"]))
        equipment = sorted(set(mine.loc[mine.kind == "equipment", "date"]))
        try:
            res, vel = model.departure(series, *args.fit, step_dates, equipment)
        except ValueError as exc:
            print(f"skip {sta}: {exc}", file=sys.stderr)
            continue
        residuals[sta] = res
        corrected[sta] = [d for d in equipment if d > pd.Timestamp(args.fit[1])]
        subtitles[sta] = f"vE {vel['e']:+.1f}\nvN {vel['n']:+.1f} mm/yr"
        print(f"{sta}: vE {vel['e']:+.1f}  vN {vel['n']:+.1f}  vU {vel['u']:+.1f} mm/yr")
    if not residuals:
        return 1

    xlim = args.xlim or (str(min(r.index.min() for r in residuals.values()).date()),
                         str(max(r.index.max() for r in residuals.values()).date()))
    out = plots.departure_grid(residuals, subtitles, events, args.out, args.title,
                               tuple(args.fit), tuple(xlim), corrected)
    print(f"wrote {out}")
    return 0


def _build_region(args):
    """The region a cube is built for: --region (a name, a region file, or
    "none"), else the built-in region named like the cube file, else none."""
    from gnssdl import regions
    if args.region is None:
        return regions.get(args.out.stem) if args.out.stem in regions.names() else None
    return None if args.region == "none" else regions.get(args.region)


def _build(args, steps: pd.DataFrame) -> int:
    from gnssdl import dataset

    man = pd.read_csv(args.manifest)
    if "ok" in man:
        man = man[man["ok"]]
    tenv3 = args.data_dir / "tenv3"
    missing = [s for s in man.sta if not (tenv3 / f"{s}.tenv3").exists()]
    if missing:
        print(f"{len(missing)} stations in the manifest have no file, e.g. {missing[:5]}; "
              "run `gnssdl fetch` first", file=sys.stderr)
        return 1

    from gnssdl import regions
    try:
        region = _build_region(args)
    except regions.RegionError as err:
        print(err, file=sys.stderr)
        return 1
    # training exclusion of the region's catalogued slow-slip episodes (contract §1.6)
    exclusion = None if region is None or args.no_episode_exclusion else region.training_exclusion()

    def progress(i: int, n: int) -> None:
        if i % 100 == 0 or i == n:
            print(f"  {i}/{n}", file=sys.stderr, flush=True)

    cube, summary = dataset.build_cube(
        man[["sta", "lat", "lon"]],
        lambda sta: ngl.read_tenv3(tenv3 / f"{sta}.tenv3"),
        steps,
        end=pd.Timestamp(args.end) if args.end else None,
        progress=progress,
        freeze_date=pd.Timestamp(args.freeze_date) if args.freeze_date else None,
        extra_exclude=exclusion,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cube.save(args.out)
    summary["manifest"] = str(args.manifest)
    summary["region"] = None if region is None else (region.name if region.name in regions.names()
                                                     else str(region.source.resolve()))
    dataset.write_summary(summary, args.out.with_suffix(".json"))

    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    print("  region: " + ("none" if region is None else f"{region.name} ({region.title})")
          + ("" if exclusion is None else ", catalogued slow-slip episodes barred from training"))
    dr = summary["stations_dropped"]
    print(f"  stations {summary['stations_kept']}/{summary['stations_in']} "
          f"(dropped: {len(dr['no_baseline'])} no baseline, {len(dr['qc'])} QC, "
          f"{len(dr['twin'])} twins), days {summary['days']} "
          f"({summary['first_day']} .. {summary['last_day']}), coverage {summary['coverage']:.1%}")
    print(f"  held-out stations {summary['heldout_stations']} "
          f"({summary['heldout_from_fallback_fit']} forced by short training record), "
          f"screened days {summary['screened_days']}, "
          f"Ridgecrest-excluded stations {summary['ridgecrest_stations']}, "
          f"episode-excluded cells {summary['extra_excluded_cells']} "
          f"({summary['excluded_stations']} stations with any exclusion)")
    for rad, v in summary["neighbours_per_radius"].items():
        print(f"  R = {rad:>3} km: median neighbours {v['median']:.0f}, "
              f"stations with < 4: {v['stations_with_fewer_than_4']}")
    return 0


class _Tee:
    """Write to a stream and a log file at once."""
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def _command_log(args):
    """Copy a bench command's console output to the model's logs directory:
    data/runs/<model>/logs/<time>_<command>.log, starting with the command line."""
    from datetime import datetime
    from gnssdl.bench.paths import log_dir
    d = log_dir(args.model, args.results)
    d.mkdir(parents=True, exist_ok=True)
    kind = args.bench_cmd
    if args.bench_cmd == "signal":
        kind += ("-events" if args.events else "-offsets" if args.offsets else "-episodes" if args.episodes
                 else "-velocity" if args.velocity else "-seasonal" if args.seasonal else "-blobs")
        kind += f"-{args.period}"
    if getattr(args, "seed", None):
        kind += f"-s{args.seed}"
    path = d / f"{datetime.now():%Y%m%d-%H%M%S}_{kind}.log"
    with path.open("a") as log:
        log.write("$ gnssdl " + " ".join(sys.argv[1:]) + "\n")
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = out, err


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _signal_measure(args) -> str:
    """The `bench signal` flags as a `tasks.run_signal` measure."""
    return ("events" if args.events else "velocity" if args.velocity else "seasonal" if args.seasonal
            else "episodes" if args.episodes else "offsets" if args.offsets
            else "noise" if args.noise_only else "transients")


def _bench(args) -> int:
    """Parse, call `gnssdl.bench.tasks`, print. Nothing is computed here."""
    from gnssdl.bench import tasks

    try:
        if args.bench_cmd == "report":
            return _bench_report(args)
        if args.bench_cmd == "signal":
            tasks.check_seed(args.model, getattr(args, "seed", None))       # before the cube is read
            s = tasks.Network.load(args.cube, args.data_dir, args.results)
            measure = _signal_measure(args)
            wrote = tasks.run_signal(s, args.model, args.period, measure, args.radius, args.seed,
                                     centres=args.centres, per_template=args.per_template, log=_err)
            if measure == "events":
                print(f"wrote {wrote['events']} and event_stations.csv")
            else:
                print(f"wrote {wrote[tasks.MEASURES[measure][-1] if measure != 'noise' else 'noise']}")
            return 0
        if args.bench_cmd == "episodes":
            from gnssdl.bench import episodes as eps
            episodes, band = eps.build(args.tremor)
            eps.EPISODE_DIR.mkdir(exist_ok=True)
            episodes.to_csv(eps.EPISODES, index=False)
            band.to_csv(eps.BAND, index=False, float_format="%.3f")
            counts = episodes.timing.value_counts().to_dict()
            print(f"wrote {eps.EPISODES} ({len(episodes)} episodes: {counts}) and {eps.BAND} ({len(band)} bins)")
            return 0
        if args.bench_cmd == "kernel":
            from gnssdl.bench.kernel_plots import kernel_figures
            from gnssdl.bench import registry
            from gnssdl.dataset import Cube
            for path in kernel_figures(registry.get(args.model).cls, Cube.load(args.cube), args.results, args.out,
                                       radius=args.radius, seed=args.seed):
                print(f"wrote {path}")
            return 0
        if args.bench_cmd == "compare":
            return _bench_compare(args)
        if args.bench_cmd == "verdict":
            return _bench_verdict(args)
        if args.bench_cmd == "intervals":
            return _bench_intervals(args)

        s = tasks.Network.load(args.cube, args.data_dir, args.results)
        if args.bench_cmd == "neighbours":
            from gnssdl.bench.run import MIN_NEIGHBOURS, SHORT_FRACTION_FLAG
            t, path = tasks.neighbours(s)
            print(f"{int(s.ctx.keep_sta.sum())} scored stations. A filter over N neighbours reads up to N stations "
                  "beyond R, within the cube's search range.")
            print(f"short: stations with fewer than N; flagged: more than {SHORT_FRACTION_FLAG:.0%} short; excluded: "
                  f"stations with fewer than {MIN_NEIGHBOURS} neighbours, not scored at that radius; reach: distance to "
                  "the farthest neighbour read (km)")
            with pd.option_context("display.width", 200):
                print(t.round({"short_fraction": 2, "read_median": 0, "reach_median_km": 0, "reach_p90_km": 0})
                      .to_string(index=False))
            print(f"wrote {path}")
            return 0
        if args.bench_cmd == "status":
            t = tasks.status(s, args.period)
            tests = [c for c in tasks.STATUS_KINDS if t[c].notna().any()]
            complete = t.fitted & t[tests].all(axis=1)
            show = (t[~complete] if args.missing else t).assign(
                radius_km=lambda d: d.radius_km.map(lambda r: "-" if pd.isna(r) else f"{r:g}"))
            mark = lambda v: "-" if v is None or pd.isna(v) else "yes" if v else "NO"
            for c in ("fitted", *tasks.STATUS_KINDS):
                show[c] = show[c].map(mark)
            print(f"{args.period} period, results under {s.results}: {int(complete.sum())} of {len(t)} rows "
                  f"(filter, seed, radius) are fitted with every signal test; {int((~t.fitted).sum())} not fitted")
            if len(show):
                with pd.option_context("display.width", 200, "display.max_rows", None):
                    print(show.to_string(index=False))
            return 0
        if args.bench_cmd == "masks":
            m, path = tasks.make_masks(s)
            for k, v in m.items():
                print(f"  {k:<10} {int(v.sum()):>9} cells, {int(v.any(axis=1).sum())} stations")
            print(f"wrote {path}")
            return 0

        # run: fit and score on the masks
        if not s.masks_path.exists():
            raise tasks.TaskError(f"no masks at {s.masks_path}; run `gnssdl bench masks` first")
        dropped = s.ctx.qc_dropped
        _err(f"scoring QC: {len(dropped)} stations left out of scoring "
             f"({', '.join(dropped.sta) if len(dropped) else 'none'})")
        path = tasks.fit_and_score(s, args.model, args.radius, args.seed, args.reuse, args.period, log=_err)
        print(f"wrote {path} and .json")
        return 0
    except tasks.TaskError as err:
        _err(str(err))
        return 1


def _bench_report(args) -> int:
    from gnssdl.bench.run import PARTS, scoreboard

    parts = PARTS if args.part == "all" else (args.part,)
    if args.split != "validation":
        print(f"NOTE: {args.split} results are for final reporting; make design and tuning "
              "decisions on the validation split only (contract §1.4).")
    print(f"Median over stations of nrmse, mean of E/N/U, {args.split} split.")
    print("Lower is better. Compare against the m0 row (predict zero), which is the floor;")
    print("for the fast part it sits below 1 because splitting off the slow part also removes")
    print("some of each station's own noise.")
    labels = {"fast": "fast part (periods under ~2 months; primary)",
              "slow": "slow part (61-day running median of the error)", "total": "total"}
    with pd.option_context("display.width", 120):
        for part in parts:
            print(f"\n{labels[part]}")
            print(scoreboard(args.results, args.split, part).to_string())
        _signal_report(args.results, args.split)
    return 0


def _signal_report(root: Path, period: str = "validation") -> None:
    from gnssdl.bench import signal as bsig
    from gnssdl.bench import tasks

    t = tasks.signal_tables(root, period)
    if "kept" in t:
        print(f"\nsignal kept, {period} period (median rho over injections; 1 = kept, 0 = absorbed)")
        print("columns: transient footprint L in km")
        print(t["kept"].round(2).to_string())
        if "spurious" in t:
            print("\nspurious signal (median over injections of the largest false movement at")
            print("stations outside the footprint, as a fraction of the planted amplitude)")
            print(t["spurious"].round(2).to_string())
        print("\nsignal kept by planted amplitude (median rho; columns: amplitude in mm)")
        print(t["kept_by_amplitude"].round(2).to_string())
    if "seasonal" in t:
        print(f"\nseasonal kept, {period} period (median over stations of rho on the annual and")
        print("semi-annual coefficients; 1 = kept, 0 = removed)")
        print(t["seasonal"].round(2).to_string())
    if "offsets" in t:
        print(f"\nreal earthquake offsets kept, {period} period (median over station-offsets)")
        print(t["offsets"].round(2).to_string())
    if "episodes" in t:
        print(f"\nslow-slip episodes kept, {period} period (slip estimate = displacement minus the median of")
        print(f"stations beyond {bsig.EPISODE_FAR_KM:g} km of the band; kept pooled over stations per episode, then the")
        print(f"median over episodes with at least {bsig.EPISODE_MIN_STATIONS} stations and a median estimate of at least")
        print(f"{bsig.EPISODE_FLOOR_FACTOR:g} noise floors; weak = left out by the floor rule; one column group per timing)")
        print(t["episodes"].round(2).to_string())
    if "noise" in t:
        print(f"\nnoise removed ({period} period, fast part, median over stations)")
        print(t["noise"].round(3).to_string())


def _bench_intervals(args) -> int:
    from gnssdl.bench import stats, tasks

    t, path = tasks.intervals(args.results, args.period, log=_err)
    # one line per model and radius; the planted-transient columns are the 100-km footprint
    wide = t[t.footprint_km.isna() | np.isclose(t.footprint_km, stats.CRITERION_FOOTPRINT_KM)]
    text = wide.assign(radius_km=wide.radius_km.map(lambda r: "" if pd.isna(r) else f"{r:g}"),
                       cell=[f"{v:.3f} [{a:.3f}, {b:.3f}]" for v, a, b in zip(wide.value, wide.ci_low, wide.ci_high)])
    show = text.pivot_table(index=["model", "radius_km"], columns="measure", values="cell", aggfunc="first")
    order = [c for c in ("noise_removed", "error", "kept", "spurious", "seasonal") if c in show.columns]
    print(f"{args.period} period: value [95% interval]; kept and spurious for "
          f"{stats.CRITERION_FOOTPRINT_KM:g}-km planted transients (all footprints are in the file)")
    with pd.option_context("display.width", 250):
        print(show[order].fillna("").to_string())
    print(f"wrote {path}")
    return 0


def _bench_verdict(args) -> int:
    from gnssdl.bench import stats, tasks

    t, path = tasks.verdict(args.results, args.period)
    ref = t[t.verdict == "reference"].iloc[0]
    print(f"{args.period} period. Reference: far stack at R = 0 (noise removed {ref.noise_removed:.3f}, "
          f"100-km kept {ref.kept_100:.3f}, spurious {ref.spurious_100:.3f}).")
    print(f"better = noise removed higher by at least {stats.CRITERION_MIN_GAIN:g} with a 95% interval above zero, "
          f"100-km kept within "
          f"{stats.CRITERION_TOLERANCE:g} below and spurious within {stats.CRITERION_TOLERANCE:g} above the reference")
    show = t.assign(radius_km=t.radius_km.map(lambda r: "" if pd.isna(r) else f"{r:g}"))
    cols = ["model", "radius_km", "noise_removed", "diff", "ci_low", "ci_high", "kept_100", "spurious_100", "verdict"]
    with pd.option_context("display.width", 200):
        print(show.sort_values("noise_removed", ascending=False)[cols].round(3).to_string(index=False))
    best = t[t.best]
    if len(best):
        b = best.iloc[0]
        same = t[t.same_as_best]
        print(f"\nbest: {b.model}" + ("" if pd.isna(b.radius_km) else f" at R = {b.radius_km:g} km")
              + f" (noise removed {b.noise_removed:.3f})"
              + (f"; not distinguishable from it: "
                 + ", ".join(m if pd.isna(r) else f"{m} R={r:g}" for m, r in zip(same.model, same.radius_km))
                 if len(same) else ""))
    else:
        print("\nno filter is better than the reference under this criterion")
    print(f"wrote {path}")
    return 0


def _bench_compare(args) -> int:
    from gnssdl.bench import tasks

    s = tasks.Network.load(args.cube, args.data_dir, args.results)
    c = tasks.compare(s, args.model_a, args.model_b, args.radius, args.radius_b, args.period, args.stations)
    res = c.noise
    print(f"{c.period} period, {c.stations} stations, leave-one-out fast nrmse (median over {res['stations']} stations, "
          f"{res['blocks']} blocks of 30 days; lower is better)")
    print(f"  {c.label_a:<14} {res['score_a']:.4f}")
    print(f"  {c.label_b:<14} {res['score_b']:.4f}")
    print(f"  difference (b - a) {res['diff']:+.4f}   95% interval [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
          + ("   (interval excludes 0)" if res["ci_low"] > 0 or res["ci_high"] < 0 else "   (not distinguishable)"))
    if c.kept is None and c.kept_note.startswith("not compared"):
        print(f"\nsignal kept: {c.kept_note}")
    elif c.kept is None:
        print(f"\n(no {c.period} signal results for both models; run `gnssdl bench signal` for rho intervals)")
    elif len(c.kept):
        if (c.kept.pairing == "unverified").any():
            print("\nNOTE: these planted-transient results were written before 2026-11-12, when ids differed "
                  "between filters; the pairs below are not the same transients, so the interval is not paired.")
        print("\nsignal kept (median rho), paired over injections"
              + ("" if c.stations == "all" else "; all injections, not restricted by --stations"))
        print(c.kept.round(3).to_string(index=False))
    return 0


def _queue(args) -> int:
    from gnssdl import queue
    from gnssdl.bench.freeze import NotFrozen
    try:
        q = queue.load(args.manifest)
    except (queue.QueueError, NotFrozen) as err:
        print(err, file=sys.stderr)
        return 1
    if args.action == "plan":
        print(queue.plan(q))
        return 0
    if args.action == "status":
        st = queue.status(q)
        print(pd.DataFrame(st["phases"]).to_string(index=False))
        for key, title in (("failed", "failed"), ("started_not_ended", "started and not ended (running, or stopped)")):
            if st[key]:
                print(f"{title}: {', '.join(st[key])}")
        print(f"log: {st['log']}" + ("" if st["last_run_finished"] else "  (no finished run yet)"))
        return 0
    print(f"{len(q.jobs)} jobs; follow with: tail -f {q.main_log}", flush=True)
    failing = queue.run(q)
    print(f"done; still failing: {failing}")
    return 1 if failing else 0
