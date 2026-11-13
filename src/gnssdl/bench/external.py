"""File mode: audit a filter that lives outside Python (Matlab, GipsyX, anything).

The benchmark never needs the filter itself, only what it returns for a
fixed set of inputs: the network's series as they are, and copies of them
with transients planted. So the audit is split in three steps:

    gnssdl external export mine --dir work/mine --centres 5
        writes every dataset the audit will ask about to work/mine/inputs/
    ... you run your filter on each dataset and write the result, in the same
        layout, to work/mine/outputs/ ...
    gnssdl external check --dir work/mine        # which outputs are missing or malformed
    gnssdl external score --dir work/mine        # the report card

`export` runs the audit's own code with a stand-in filter that records each
series it is asked to clean; `score` runs the same code again with a stand-in
that answers from your files. The measures are therefore exactly those of
`gnssdl.audit`, with nothing reimplemented.

An external filter is treated as one that reads the target station (it is
given whole networks), so it has no leave-one-out error and is scored by the
signal tests, like C1 or a PCA filter. Because nothing is known about what it
reads, every planted transient is assumed to be able to change every station:
transients that overlap in time are never planted together, which makes more
datasets than a filter with a known reach needs.

Layout of a work directory (csv format; see `README.txt` written by export):

    manifest.json                 what was exported, and from what
    stations.csv                  sta, lat, lon
    inputs/<key>/e.csv n.csv u.csv    days × stations, mm, empty where there is no solution
    outputs/<key>/e.csv n.csv u.csv   yours: the filtered series (or the common mode), same layout
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gnssdl.bench import registry, tasks
from gnssdl.bench.base import Reconstructor
from gnssdl.bench.card import AUDIT_MEASURES, ReportCard, audit
from gnssdl.bench.freeze import require
from gnssdl.bench.tasks import Log, Network, TaskError, _quiet
from gnssdl.dataset import Cube

FORMATS = ("csv", "npz")
OUTPUT_KINDS = ("filtered", "common_mode")
COMPONENTS = ("e", "n", "u")
MANIFEST = "manifest.json"

README = """This directory holds one audit of an external filter with gnssdl (file mode).

What to do
  For every dataset under inputs/, run your filter on it and write the result
  under outputs/ with the same name and the same layout.

A dataset (csv format): a directory inputs/<key>/ with e.csv, n.csv and u.csv,
the east, north and up components in millimetres. Each file has one row per
day (first column `date`, YYYY-MM-DD) and one column per station (the header
row has the station names). An empty cell means there is no solution that day.
The series are residuals: each station's rate and equipment steps are removed;
the seasonal cycle and earthquake offsets are still in them.
stations.csv gives each station's latitude and longitude.

A dataset (npz format): inputs/<key>.npz with `r` (stations x days x 3, mm,
NaN where there is no solution), `sta`, `days`, `lat`, `lon`. Write
outputs/<key>.npz with one array `r` of the same shape.

What to write: {what}
  Every cell that has a value in the input must have a value in the output.
  Cells that are empty in the input are ignored.

The datasets are all the same network. Most of them are short windows with
synthetic transients added; the others are the series as observed. Treat every
dataset the same way, with the same settings, and do not look for the
transients: the audit measures what your filter does to them.

Then
  gnssdl external check --dir {dir}     # what is missing or malformed
  gnssdl external score --dir {dir}     # the report card
"""


def _key(cube: Cube) -> str:
    """A short name for one input: the same series always get the same key."""
    h = hashlib.sha1()
    h.update(str(cube.r.shape).encode())
    h.update(np.ascontiguousarray(np.nan_to_num(cube.r, nan=0.0)).tobytes())
    h.update(np.ascontiguousarray(cube.avail).tobytes())
    h.update(str(cube.days[0]).encode())
    return h.hexdigest()[:16]


def _values(cube: Cube) -> np.ndarray:
    return np.where(cube.avail[..., None], cube.r, np.nan).astype(np.float32)


def _write_dataset(cube: Cube, root: Path, key: str, fmt: str) -> None:
    values = _values(cube)
    days = pd.DatetimeIndex(cube.days).strftime("%Y-%m-%d")
    if fmt == "npz":
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"{key}.tmp.npz"
        np.savez_compressed(tmp, r=values, sta=cube.sta.astype(str), days=np.array(days, dtype=str), lat=cube.lat,
                            lon=cube.lon)
        tmp.replace(root / f"{key}.npz")
        return
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(COMPONENTS):
        frame = pd.DataFrame(values[:, :, k].T, index=pd.Index(days, name="date"), columns=cube.sta.astype(str))
        frame.to_csv(d / f"{c}.csv", float_format="%.4f")


def _read_dataset(root: Path, key: str, fmt: str, cube: Cube) -> np.ndarray:
    """A dataset written by the user, as stations × days × 3 in the cube's
    station and day order; raises TaskError naming what is wrong."""
    S, T, _ = cube.r.shape
    if fmt == "npz":
        path = root / f"{key}.npz"
        if not path.exists():
            raise TaskError(f"missing output {path}")
        with np.load(path, allow_pickle=False) as z:
            if "r" not in z.files:
                raise TaskError(f"{path}: no array named `r`")
            out = np.asarray(z["r"], dtype=np.float32)
        if out.shape != (S, T, 3):
            raise TaskError(f"{path}: `r` has shape {out.shape}, the input has {(S, T, 3)}")
        return out
    days = pd.DatetimeIndex(cube.days).strftime("%Y-%m-%d")
    sta = [str(s) for s in cube.sta]
    out = np.full((S, T, 3), np.nan, dtype=np.float32)
    for k, c in enumerate(COMPONENTS):
        path = root / key / f"{c}.csv"
        if not path.exists():
            raise TaskError(f"missing output {path}")
        frame = pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        lost = [s for s in sta if s not in frame.columns]
        if lost:
            raise TaskError(f"{path}: {len(lost)} stations of the input are not columns here, e.g. {lost[:5]}")
        if len(frame) != T or not frame.index.isin(days).all() or frame.index.has_duplicates:
            raise TaskError(f"{path}: the rows must be the input's {T} days ({days[0]} to {days[-1]}), once each")
        try:
            out[:, :, k] = frame.loc[days, sta].to_numpy(dtype=np.float32).T
        except ValueError:
            raise TaskError(f"{path}: every cell must be a number or empty") from None
    return out


class _External(Reconstructor):
    """What both stand-ins share: the filter reads the target, and a planted
    transient is assumed to be able to change every station."""
    reference_filter = True
    never_reads_target = False

    def affected_by(self, cube: Cube, inside: np.ndarray) -> np.ndarray:
        return np.ones_like(inside)


def _recorder(name: str, root: Path, fmt: str, seen: dict) -> type[_External]:
    """A filter that writes every series it is asked to clean and removes nothing."""
    class Recorder(_External):
        purpose = ""

        def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
            key = _key(cube)
            if key not in seen:
                _write_dataset(cube, root, key, fmt)
                days = pd.DatetimeIndex(cube.days)
                # "for": the first measure that asked for it; later measures may use it too
                seen[key] = {"key": key, "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
                             "days": int(len(days)), "stations": int(len(cube.sta)), "for": Recorder.purpose}
            return np.zeros(cube.r.shape, dtype=np.float32)

    Recorder.name = name
    return Recorder


def _replayer(name: str, root: Path, fmt: str, kind: str) -> type[_External]:
    """A filter that answers from the user's files."""
    class Replayer(_External):
        def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
            key = _key(cube)
            given = _read_dataset(root, key, fmt, cube)
            bad = int((cube.avail[..., None] & ~np.isfinite(given)).sum())
            if bad:
                raise TaskError(f"output {key}: {bad} cells that have a value in the input have none in the "
                                "output; a filter must return a value wherever it was given one")
            given = np.where(cube.avail[..., None], given, 0.0)
            series = np.where(cube.avail[..., None], cube.r, 0.0)
            return (series - given if kind == "filtered" else given).astype(np.float32)

    Replayer.name = name
    return Replayer


def _manifest(work: Path) -> dict:
    path = Path(work) / MANIFEST
    if not path.exists():
        raise TaskError(f"no {MANIFEST} in {work}: run `gnssdl external export` first")
    return json.loads(path.read_text())


def _free_name(name: str) -> None:
    known = registry.find(name)
    if known is not None and not issubclass(known.cls, _External):
        raise TaskError(f"{name!r} is the name of a built-in or registered filter; give the external filter its own")


def export(net: Network, name: str, work: Path | str, period: str = "validation",
           measures: tuple[str, ...] = AUDIT_MEASURES, centres: int | None = None, fmt: str = "csv",
           output: str = "filtered", log: Log = _quiet) -> dict:
    """Write every dataset the audit of an external filter needs under
    `work`/inputs and return the manifest (also written to `work`).

    name      the filter's name in the results (lower-case letters, digits, underscores)
    measures  the signal tests to prepare for, as in `gnssdl.audit`
    centres   planted transients per grid cell (default: the benchmark's 50;
              each transient pair needs its own dataset, so a small number,
              e.g. 5, keeps the number of datasets in the low hundreds)
    fmt       "csv" (one directory of e/n/u tables per dataset) or "npz"
    output    what the filter will write: "filtered" (the cleaned series) or
              "common_mode" (what it removes)
    """
    require(period)
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
    if output not in OUTPUT_KINDS:
        raise ValueError(f"output must be one of {OUTPUT_KINDS}")
    _free_name(name)
    work = Path(work)
    if (work / MANIFEST).exists():
        raise TaskError(f"{work} already holds an export ({MANIFEST}); use a new directory, or delete this one")
    if "episodes" in measures and (net.region is None or not net.region.has_episodes):
        log(tasks.no_episodes(net))
        measures = tuple(m for m in measures if m != "episodes")
    unknown = [m for m in measures if m not in tasks.MEASURES]
    if unknown:
        raise ValueError(f"unknown measures {unknown}; choose from {sorted(tasks.MEASURES)}")
    work.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    recorder = _recorder(name, work / "inputs", fmt, seen)
    registry.register(recorder, family="reads-target", fixed_radius=0.0, replace=True)
    try:
        with tempfile.TemporaryDirectory(prefix="gnssdl-export-") as tmp:      # the stand-in's results are discarded
            scratch = replace(net, results=Path(tmp))
            scratch.steps, scratch.ctx = net.steps, net.ctx
            for measure in measures:
                recorder.purpose = measure
                before = len(seen)
                # the stand-in removes nothing, so the progress lines of the measure would be
                # meaningless numbers: only the count of datasets is reported
                tasks.run_signal(scratch, name, period, measure, centres=centres)
                log(f"{measure}: {len(seen) - before} new datasets")
    finally:
        registry.unregister(name)
    pd.DataFrame({"sta": net.cube.sta.astype(str), "lat": net.cube.lat, "lon": net.cube.lon}).to_csv(
        work / "stations.csv", index=False, float_format="%.5f")
    what = ("the filtered series: the input with the common mode removed" if output == "filtered"
            else "the common mode: what your filter would subtract from each station")
    (work / "README.txt").write_text(README.format(what=what, dir=work))
    manifest = {"name": name, "cube": str(net.cube_path), "data_dir": str(net.data_dir),
                "region": None if net.region is None else net.region.name, "period": period,
                "measures": list(measures), "centres": centres, "format": fmt, "output": output,
                "exported": datetime.now().isoformat(timespec="seconds"), "datasets": list(seen.values())}
    (work / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"{len(seen)} datasets written to {work / 'inputs'}; instructions in {work / 'README.txt'}")
    return manifest


def check(work: Path | str) -> pd.DataFrame:
    """One row per dataset: whether its output is there, by file only (shape
    and values are checked when it is read by `score`)."""
    work = Path(work)
    m = _manifest(work)
    rows = []
    for d in m["datasets"]:
        if m["format"] == "npz":
            present = (work / "outputs" / f"{d['key']}.npz").exists()
        else:
            present = all((work / "outputs" / d["key"] / f"{c}.csv").exists() for c in COMPONENTS)
        rows.append({**d, "output": present})
    return pd.DataFrame(rows)


def score(work: Path | str, net: Network | None = None, results: Path | str | None = None,
          log: Log = _quiet) -> ReportCard:
    """Score the outputs in `work` and return the report card. The network is
    the one the export was made from, unless `net` is given; results are
    written under its results root (or `results`) like any other filter's."""
    work = Path(work)
    m = _manifest(work)
    require(m["period"])
    status = check(work)
    missing = status[~status.output]
    if len(missing):
        raise TaskError(f"{len(missing)} of {len(status)} outputs are missing under {work / 'outputs'}, e.g. "
                        f"{', '.join(missing.key.head(3))}; run `gnssdl external check --dir {work}`")
    if net is None:
        net = Network.load(m["cube"], data_dir=m["data_dir"], **({} if results is None else {"results": results}))
    elif results is not None:
        steps, ctx = net.steps, net.ctx
        net = replace(net, results=Path(results))
        net.steps, net.ctx = steps, ctx
    _free_name(m["name"])
    replayer = _replayer(m["name"], work / "outputs", m["format"], m["output"])
    registry.register(replayer, family="reads-target", fixed_radius=0.0, replace=True)
    try:
        return audit(m["name"], net, period=m["period"], measures=tuple(m["measures"]), centres=m["centres"],
                     rerun=True, log=log)
    finally:
        registry.unregister(m["name"])
