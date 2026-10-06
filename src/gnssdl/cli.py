"""gnssdl command line.

    gnssdl meta                                   # cache station table + steps
    gnssdl near --lat 37.23 --lon 37.01 -r 300 --around 2023-02-06
    gnssdl fetch ANKR ISTA                        # explicit stations
    gnssdl fetch --lat 37.23 --lon 37.01 -r 300 --around 2023-02-06 --before 365 --after 180
    gnssdl departure ELAZ ERGN --fit 2009-01-01 2015-12-31 \
        --event 2020-01-24:"Elazığ Mw6.8" --event 2023-02-06:"Kahramanmaraş Mw7.8"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    args = ap.parse_args(argv)
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
