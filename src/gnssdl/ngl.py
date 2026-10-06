"""Nevada Geodetic Laboratory (NGL) daily GNSS time series: fetch and parse.

NGL processes ~20k stations worldwide with one consistent strategy (GipsyX PPP,
IGS20 frame) and publishes them as plain text. Three files matter:

- DataHoldings.txt: one row per station -- position and the span of solutions.
- steps.txt:       known offsets. Type 1 = equipment change (antenna/receiver),
                   type 2 = a catalogued earthquake close enough to possibly
                   offset the station. Both must be handled before any analysis.
- {STA}.tenv3:     the daily east/north/up position series for one station.
"""

from __future__ import annotations

import io
import re
import os
import time
import urllib.error
import urllib.request
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE = "https://geodesy.unr.edu"
HOLDINGS_URL = f"{BASE}/NGLStationPages/DataHoldings.txt"
STEPS_URL = f"{BASE}/NGLStationPages/steps.txt"
TENV3_URL = f"{BASE}/gps_timeseries/IGS20/tenv3/IGS20/{{sta}}.tenv3"

USER_AGENT = "gnssdl/0.1 (research; daily tenv3 fetcher)"
EARTH_RADIUS_KM = 6371.0
MJD_EPOCH = pd.Timestamp("1858-11-17")


_STATION_ID = re.compile(r"[A-Z0-9]{4}")


def is_station_id(s: str) -> bool:
    """NGL ids are four upper-case alphanumerics; anything else would be pasted
    into a URL path."""
    return bool(_STATION_ID.fullmatch(s))


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #


def download(url: str, dest: Path, *, refresh: bool = False, retries: int = 3) -> Path:
    """Fetch `url` to `dest`, skipping if cached. Writes atomically via a .part file
    so an interrupted run never leaves a truncated file that looks complete."""
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as fh:
                while chunk := resp.read(1 << 16):
                    fh.write(chunk)
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_metadata(data_dir: Path, *, refresh: bool = False) -> tuple[Path, Path]:
    meta = data_dir / "meta"
    return (
        download(HOLDINGS_URL, meta / "DataHoldings.txt", refresh=refresh),
        download(STEPS_URL, meta / "steps.txt", refresh=refresh),
    )


@dataclass
class FetchResult:
    station: str
    path: Path | None
    error: str | None = None


def fetch_tenv3(
    stations: list[str], data_dir: Path, *, refresh: bool = False, workers: int = 4
) -> list[FetchResult]:
    """Download daily series for `stations`. A small worker pool: NGL is a
    university server, not a CDN."""
    out_dir = data_dir / "tenv3"

    def one(sta: str) -> FetchResult:
        try:
            path = download(TENV3_URL.format(sta=sta), out_dir / f"{sta}.tenv3", refresh=refresh)
            return FetchResult(sta, path)
        except Exception as exc:  # report per station, never abort the batch
            return FetchResult(sta, None, f"{type(exc).__name__}: {exc}")

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, s) for s in stations]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results[r.station] = r
            if len(stations) >= 50 and (i % 50 == 0 or i == len(stations)):
                print(f"  {i}/{len(stations)}", file=sys.stderr, flush=True)
    return [results[s] for s in stations]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def read_holdings(path: Path) -> pd.DataFrame:
    """Station table: sta, lat, lon (-180..180), height, start, end, n_sol."""
    rows = []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            p = line.split()
            if len(p) < 11:
                continue
            rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), p[7], p[8], int(p[10])))
    df = pd.DataFrame(rows, columns=["sta", "lat", "lon", "height", "start", "end", "n_sol"])
    df["lon"] = (df["lon"] + 180.0) % 360.0 - 180.0
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def _yymmmdd(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, format="%y%b%d")


def read_steps(path: Path) -> pd.DataFrame:
    """Known offsets. kind='equipment' rows carry a description; kind='quake'
    rows carry the event's influence radius, distance, magnitude and id."""
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            if p[2] == "1":
                rows.append((p[0], _yymmmdd(p[1]), "equipment", " ".join(p[3:]), None, None, None))
            elif p[2] == "2" and len(p) >= 7:
                rows.append((p[0], _yymmmdd(p[1]), "quake", p[6], float(p[3]), float(p[4]), float(p[5])))
    return pd.DataFrame(
        rows, columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"]
    )


TENV3_COLS = [
    "sta", "yymmmdd", "decyear", "mjd", "gps_week", "gps_dow", "reflon",
    "e0", "east", "n0", "north", "u0", "up", "ant",
    "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu", "lat", "lon", "height",
]


def read_tenv3(path: Path) -> pd.DataFrame:
    """Daily positions indexed by date. e/n/u are full metres (integer part +
    fraction) in a local frame; subtract the first value or a fit before use."""
    text = Path(path).read_text()
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", skiprows=1, header=None, names=TENV3_COLS)
    df["date"] = MJD_EPOCH + pd.to_timedelta(df["mjd"], unit="D")
    df["e"] = df["e0"] + df["east"]
    df["n"] = df["n0"] + df["north"]
    df["u"] = df["u0"] + df["up"]
    return df.set_index("date")[
        ["decyear", "e", "n", "u", "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu", "ant"]
    ]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance; works elementwise on pandas/numpy arrays."""
    import numpy as np

    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def stations_in_bbox(
    holdings: pd.DataFrame, south: float, north: float, west: float, east: float
) -> pd.DataFrame:
    """Stations inside a lat/lon box (no antimeridian crossing)."""
    m = holdings["lat"].between(south, north) & holdings["lon"].between(west, east)
    return holdings[m].sort_values("sta").reset_index(drop=True)


def stations_near(
    holdings: pd.DataFrame,
    lat: float,
    lon: float,
    radius_km: float,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Stations within `radius_km`, optionally only those whose data span covers
    the whole [start, end] window. Sorted by distance."""
    df = holdings.copy()
    df["dist_km"] = haversine_km(lat, lon, df["lat"], df["lon"])
    df = df[df["dist_km"] <= radius_km]
    if start is not None:
        df = df[df["start"] <= start]
    if end is not None:
        df = df[df["end"] >= end]
    return df.sort_values("dist_km").reset_index(drop=True)

