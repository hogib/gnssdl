"""gnssdl command line.

    gnssdl meta                                   # cache station table + steps
    gnssdl near --lat 37.23 --lon 37.01 -r 300 --around 2023-02-06
    gnssdl fetch ANKR ISTA                        # explicit stations
    gnssdl fetch --lat 37.23 --lon 37.01 -r 300 --around 2023-02-06 --before 365 --after 180
    gnssdl departure ELAZ ERGN --fit 2009-01-01 2015-12-31 \
        --event 2020-01-24:"Elazığ Mw6.8" --event 2023-02-06:"Kahramanmaraş Mw7.8"
    gnssdl build --manifest data/manifests/california.csv   # benchmark data cube
    gnssdl bench masks                            # fixed evaluation masks
    gnssdl bench run m1                           # fit + score a model at every radius
    gnssdl bench report                           # scoreboard
"""

from __future__ import annotations

import argparse
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
    p_build.add_argument("--manifest", type=Path, default=Path("data/manifests/california.csv"))
    p_build.add_argument("-o", "--out", type=Path, default=Path("data/cube/california.npz"))
    p_build.add_argument("--end", help="last day of the cube (default: latest epoch)")
    p_build.add_argument("--freeze-date",
                         help="method freeze date; later days become the prospective test period")

    p_bench = sub.add_parser("bench", help="benchmark: masks, run models, report")
    bsub = p_bench.add_subparsers(dest="bench_cmd", required=True)
    for bp in (bsub.add_parser("masks", help="generate the fixed evaluation masks"),
               bsub.add_parser("run", help="fit and score a model"),
               bsub.add_parser("report", help="print the scoreboard"),
               bsub.add_parser("signal", help="injected-transient and Ridgecrest signal tests"),
               bsub.add_parser("compare", help="paired bootstrap comparison of two models"),
               bsub.add_parser("kernel", help="plot a trained M3k kernel")):
        bp.add_argument("--cube", type=Path, default=Path("data/cube/california.npz"))
        bp.add_argument("--results", type=Path, default=Path("data/results"))
    bsub.choices["run"].add_argument("model", choices=["m0", "m1", "m1k64", "m1k256", "m2", "d1", "fs", "m3k", "m3k256", "m5", "m5aug"])
    bsub.choices["run"].add_argument("--seed", type=int, default=None,
                                     help="training seed of a learned model (default 0)")
    bsub.choices["run"].add_argument("--radius", type=float, action="append",
                                     help="exclusion radius in km (repeatable; default: all)")
    bsub.choices["signal"].add_argument("model", choices=["m0", "m1", "m1k64", "m1k256", "m2", "m2self",
                                                           "c1", "c3", "d1", "fs", "m3k", "m3k256", "m5", "m5aug"])
    bsub.choices["signal"].add_argument("--radius", type=float, action="append",
                                        help="exclusion radius in km (repeatable; default: all)")
    bsub.choices["signal"].add_argument("--centres", type=int, default=None,
                                        help="injections per grid cell (default 50)")
    bsub.choices["signal"].add_argument("--period", choices=["validation", "test", "prospective"],
                                        default="validation",
                                        help="period to plant signals in (design decisions: validation)")
    bsub.choices["signal"].add_argument("--noise-only", action="store_true",
                                        help="only recompute the noise-removed measure")
    bsub.choices["signal"].add_argument("--events", action="store_true",
                                        help="plant realistic fault-slip events from the event templates "
                                             "instead of Gaussian transients")
    bsub.choices["signal"].add_argument("--per-template", type=int, default=None,
                                        help="events per template (default 20)")
    bsub.choices["compare"].add_argument("model_a")
    bsub.choices["compare"].add_argument("model_b")
    bsub.choices["compare"].add_argument("--radius", type=float, default=0.0)
    bsub.choices["compare"].add_argument("--radius-b", type=float, help="radius for model_b (default: --radius)")
    bsub.choices["compare"].add_argument("--period", choices=["validation", "test", "prospective"],
                                         default="validation")
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
    if args.cmd == "bench":
        return _bench(args)
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
    )
    cube.save(args.out)
    summary["manifest"] = str(args.manifest)
    dataset.write_summary(summary, args.out.with_suffix(".json"))

    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    dr = summary["stations_dropped"]
    print(f"  stations {summary['stations_kept']}/{summary['stations_in']} "
          f"(dropped: {len(dr['no_baseline'])} no baseline, {len(dr['qc'])} QC, "
          f"{len(dr['twin'])} twins), days {summary['days']} "
          f"({summary['first_day']} .. {summary['last_day']}), coverage {summary['coverage']:.1%}")
    print(f"  held-out stations {summary['heldout_stations']} "
          f"({summary['heldout_from_fallback_fit']} forced by short training record), "
          f"screened days {summary['screened_days']}, "
          f"Ridgecrest-excluded stations {summary['ridgecrest_stations']}")
    for rad, v in summary["neighbours_per_radius"].items():
        print(f"  R = {rad:>3} km: median neighbours {v['median']:.0f}, "
              f"stations with < 4: {v['stations_with_fewer_than_4']}")
    return 0


def _bench(args) -> int:
    from gnssdl.bench import masks as bmasks
    from gnssdl.bench.run import PARTS, ScoringContext, run_model, save_results, scoreboard
    from gnssdl.dataset import Cube

    masks_path = args.cube.with_name("masks.npz")
    if args.bench_cmd == "report":
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
            _signal_report(args.results / "signal" / args.split, args.split)
        return 0

    if args.bench_cmd == "signal":
        return _bench_signal(args)
    if args.bench_cmd == "kernel":
        from gnssdl.bench.kernel_plots import kernel_figures
        from gnssdl.bench.run import MODELS
        from gnssdl.dataset import Cube
        for path in kernel_figures(MODELS[args.model], Cube.load(args.cube), args.results, args.out,
                                   radius=args.radius, seed=args.seed):
            print(f"wrote {path}")
        return 0
    if args.bench_cmd == "compare":
        return _bench_compare(args)

    cube = Cube.load(args.cube)
    if args.bench_cmd == "masks":
        m = bmasks.make_masks(cube)
        bmasks.save_masks(m, masks_path)
        for k, v in m.items():
            print(f"  {k:<10} {int(v.sum()):>9} cells, {int(v.any(axis=1).sum())} stations")
        print(f"wrote {masks_path}")
        return 0

    if not masks_path.exists():
        print(f"no masks at {masks_path}; run `gnssdl bench masks` first", file=sys.stderr)
        return 1
    _, steps_path = ngl.fetch_metadata(args.data_dir)
    ctx = ScoringContext.build(cube, ngl.read_steps(steps_path))
    print(f"scoring QC: {len(ctx.qc_dropped)} stations left out of scoring "
          f"({', '.join(ctx.qc_dropped.sta) if len(ctx.qc_dropped) else 'none'})", file=sys.stderr)
    from gnssdl.bench.run import MODELS, seeded_label
    if args.seed is not None and "seed" not in MODELS[args.model].__init__.__code__.co_varnames:
        print(f"{args.model} has no training seed", file=sys.stderr)
        return 1
    scores, configs = run_model(args.model, cube, bmasks.load_masks(masks_path), ctx, args.radius,
                                log=lambda msg: print(msg, file=sys.stderr, flush=True), seed=args.seed)
    label = seeded_label(args.model, args.seed)
    save_results(label, scores, configs, args.results)
    print(f"wrote {args.results / (label + '.csv')} and .json")
    return 0


def _bench_signal(args) -> int:
    from gnssdl.bench import signal as bsig
    from gnssdl.bench.run import RADIUS_FREE, SWEEPS, ScoringContext, load_model, model_radii, variant_label
    from gnssdl.dataset import Cube

    cube = Cube.load(args.cube)
    _, steps_path = ngl.fetch_metadata(args.data_dir)
    ctx = ScoringContext.build(cube, ngl.read_steps(steps_path))
    if args.model in SWEEPS:
        # (label, radius, settings): one labelled point per swept setting
        points = [(variant_label(args.model, kw), 0.0, kw) for kw in SWEEPS[args.model]]
    else:
        points = [(args.model, r, {}) for r in model_radii(args.model, cube, args.radius)]
    print(f"{args.period} period, {', '.join(f'{lab} R={r:g}' for lab, r, _ in points)}", file=sys.stderr)

    free = args.model in RADIUS_FREE
    inj_frames, rc_frames, noise_rows = [], [], []
    out = args.results / "signal" / args.period
    out.mkdir(parents=True, exist_ok=True)
    if args.events:
        return _bench_events(args, cube, ctx, points, free, out)
    for label, radius, settings in points:
        model = load_model(args.model, radius, cube, args.results, ctx=ctx, **settings)
        tag = label if free else f"R={radius:g}"
        noise = bsig.noise_removed(model, cube, ctx.keep_sta, args.period)
        noise_rows.append({"model": label, "radius_km": np.nan if free else radius, **noise})
        print(f"  {tag}: noise removed {noise['noise_removed']:.3f}", file=sys.stderr, flush=True)
        if args.noise_only:
            continue
        runs = bsig.plan_injections(
            cube, ctx.keep_sta, args.centres or bsig.CENTRES_PER_CELL, period=args.period,
            radius_km=radius, model=model)
        print(f"  {tag}: {sum(len(r) for r in runs)} injections in {len(runs)} runs", file=sys.stderr)
        log = lambda k, n: print(f"  {tag}: run {k}/{n}", file=sys.stderr, flush=True) if k % 20 == 0 or k == n else None
        inj = bsig.injection_scores(model, cube, ctx.keep_sta, runs, log=log)
        rc = bsig.ridgecrest_retention(model, cube, ctx.keep_sta)
        for df in (inj, rc):
            df.insert(0, "radius_km", np.nan if free else radius)
            df.insert(0, "model", label)
        inj_frames.append(inj)
        rc_frames.append(rc)
    pd.DataFrame(noise_rows).to_csv(out / f"{args.model}_noise.csv", index=False)
    if args.noise_only:
        print(f"wrote {out}/{args.model}_noise.csv")
        return 0
    pd.concat(inj_frames).to_csv(out / f"{args.model}_injections.csv", index=False)
    pd.concat(rc_frames).to_csv(out / f"{args.model}_ridgecrest.csv", index=False)
    print(f"wrote {out}/{args.model}_injections.csv and _ridgecrest.csv")
    return 0


def _bench_events(args, cube, ctx, points, free, out) -> int:
    from gnssdl.bench import event_signal as es
    from gnssdl.bench import events
    from gnssdl.bench.run import load_model

    templates = es.load_templates()
    if not templates:
        print(f"no event templates in {es.EVENT_DIR}/*.toml; write them from the source papers first "
              "(contract §5.5)", file=sys.stderr)
        return 1
    catalogue = events.fault_catalogue()
    print(f"{len(templates)} templates ({', '.join(t.name for t in templates)}), "
          f"{len(catalogue)} CFM faults", file=sys.stderr)
    ev_frames, sta_frames = [], []
    for label, radius, settings in points:
        model = load_model(args.model, radius, cube, args.results, ctx=ctx, **settings)
        tag = label if free else f"R={radius:g}"
        runs = es.plan_events(cube, ctx.keep_sta, templates, catalogue,
                              per_template=args.per_template or es.EVENTS_PER_TEMPLATE,
                              period=args.period, radius_km=radius, model=model)
        print(f"  {tag}: {sum(len(r) for r in runs)} events in {len(runs)} runs", file=sys.stderr)
        log = lambda k, n: print(f"  {tag}: run {k}/{n}", file=sys.stderr, flush=True) if k % 5 == 0 or k == n else None
        ev_rows, sta_rows = es.event_scores(model, cube, ctx.keep_sta, runs, log=log)
        for df in (ev_rows, sta_rows):
            df.insert(0, "radius_km", np.nan if free else radius)
            df.insert(0, "model", label)
        ev_frames.append(ev_rows)
        sta_frames.append(sta_rows)
        summary = ev_rows.groupby("template").rho.median().round(2).to_dict()
        print(f"  {tag}: median rho by template {summary}", file=sys.stderr)
    pd.concat(ev_frames).to_csv(out / f"{args.model}_events.csv", index=False)
    pd.concat(sta_frames).to_csv(out / f"{args.model}_event_stations.csv", index=False)
    print(f"wrote {out}/{args.model}_events.csv and _event_stations.csv")
    return 0


def _signal_report(sig_dir: Path, period: str = "validation") -> None:
    inj = sorted(sig_dir.glob("*_injections.csv")) if sig_dir.exists() else []
    if not inj:
        return
    df = pd.concat(pd.read_csv(p) for p in inj)
    df["R_km"] = df["radius_km"].map(lambda r: "-" if pd.isna(r) else f"{r:g}")
    df["R_sort"] = df["radius_km"].fillna(-1)
    print(f"\nsignal kept, {period} period (median rho over injections; 1 = kept, 0 = absorbed)")
    print("columns: transient footprint L in km")
    board = df.pivot_table(index=["model", "R_sort", "R_km"], columns="footprint_km", values="rho",
                           aggfunc="median").reset_index(level="R_sort", drop=True)
    print(board.round(2).to_string())
    if "spurious_max" in df:
        print("\nspurious signal (median over injections of the largest false movement at")
        print("stations outside the footprint, as a fraction of the planted amplitude)")
        sp = df.pivot_table(index=["model", "R_sort", "R_km"], columns="footprint_km",
                            values="spurious_max", aggfunc="median").reset_index(level="R_sort", drop=True)
        print(sp.round(2).to_string())
    print("\nsignal kept by planted amplitude (median rho; columns: amplitude in mm)")
    am = df.pivot_table(index=["model", "R_sort", "R_km"], columns="amplitude_mm", values="rho",
                        aggfunc="median").reset_index(level="R_sort", drop=True)
    print(am.round(2).to_string())
    rc = sorted(sig_dir.glob("*_ridgecrest.csv"))
    if rc:
        r = pd.concat(pd.read_csv(p) for p in rc)
        r["R_km"] = r["radius_km"].map(lambda x: "-" if pd.isna(x) else f"{x:g}")
        r["R_sort"] = r["radius_km"].fillna(-1)
        print("\nRidgecrest postseismic kept (median over stations within 80 km)")
        t = r.groupby(["model", "R_sort", "R_km"])["retained"].agg(["median", "count"])
        print(t.reset_index(level="R_sort", drop=True).round(2).to_string())
    nz = sorted(sig_dir.glob("*_noise.csv"))
    if nz:
        n = pd.concat(pd.read_csv(p) for p in nz)
        n["R_km"] = n["radius_km"].map(lambda x: "-" if pd.isna(x) else f"{x:g}")
        n["R_sort"] = n["radius_km"].fillna(-1)
        print(f"\nnoise removed ({period} period, fast part, median over stations; same for every model)")
        print(n.sort_values(["model", "R_sort"]).set_index(["model", "R_km"])[["noise_removed"]].round(3).to_string())


def _bench_compare(args) -> int:
    from gnssdl.bench import signal as bsig
    from gnssdl.bench import stats
    from gnssdl.bench.run import RADIUS_FREE, ScoringContext, load_model
    from gnssdl.dataset import Cube, crop_days

    cube = Cube.load(args.cube)
    _, steps_path = ngl.fetch_metadata(args.data_dir)
    ctx = ScoringContext.build(cube, ngl.read_steps(steps_path))
    code = bsig.PERIODS[args.period]
    period = np.flatnonzero(cube.split_day == code)
    lo = max(0, period[0] - bsig.CROP_MARGIN_DAYS)
    hi = min(cube.r.shape[1], period[-1] + bsig.CROP_MARGIN_DAYS + 1)
    small = crop_days(cube, lo, hi)
    cells = (small.avail & (small.split_day == code)[None, :] & ~small.exclude
             & ~ctx.skip[:, lo:hi] & ctx.keep_sta[:, None])

    from gnssdl.bench.run import FIXED_RADIUS
    radius_b = args.radius if args.radius_b is None else args.radius_b
    args.radius = FIXED_RADIUS.get(args.model_a, args.radius)
    radius_b = FIXED_RADIUS.get(args.model_b, radius_b)
    block = {}
    for name, radius in ((args.model_a, args.radius), (args.model_b, radius_b)):
        model = load_model(name, radius, cube, args.results)
        resid = small.r - bsig.clean(model, small)
        block[name, radius] = stats.block_stats(small, resid, cells, ctx.scale)
    a, b = block[args.model_a, args.radius], block[args.model_b, radius_b]
    res = stats.paired_bootstrap(a, b)
    label_a = f"{args.model_a}" + ("" if args.model_a in RADIUS_FREE else f" R={args.radius:g}")
    label_b = f"{args.model_b}" + ("" if args.model_b in RADIUS_FREE else f" R={radius_b:g}")
    print(f"{args.period} period, leave-one-out fast nrmse (median over {res['stations']} stations, "
          f"{res['blocks']} blocks of 30 days; lower is better)")
    print(f"  {label_a:<14} {res['score_a']:.4f}")
    print(f"  {label_b:<14} {res['score_b']:.4f}")
    print(f"  difference (b - a) {res['diff']:+.4f}   95% interval [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
          + ("   (interval excludes 0)" if res["ci_low"] > 0 or res["ci_high"] < 0 else "   (not distinguishable)"))

    sig = args.results / "signal" / args.period
    fa, fb = sig / f"{args.model_a}_injections.csv", sig / f"{args.model_b}_injections.csv"
    if fa.exists() and fb.exists():
        da, db = pd.read_csv(fa), pd.read_csv(fb)
        pick = lambda d, name, r: d if name in RADIUS_FREE else d[d.radius_km == r]
        t = stats.paired_rho_bootstrap(pick(da, args.model_a, args.radius), pick(db, args.model_b, radius_b))
        if len(t):
            print("\nsignal kept (median rho), paired over injections")
            print(t.round(3).to_string(index=False))
    else:
        print(f"\n(no signal results for both models in {sig}; run `gnssdl bench signal` for rho intervals)")
    return 0
