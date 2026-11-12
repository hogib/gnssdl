"""Cascadia slow-slip episodes for the PNW held-out network (contract §1.6).

The episode list is fixed before any PNW run and tracked in the repository
(`regions/pnw/cascadia_sse.csv`, named by the region file `regions/pnw.toml`).
It has two kinds of rows:

- `timing = gnss`: episodes from published GNSS catalogues, transcribed by
  hand into `regions/pnw/cascadia_gnss_catalogue.csv` with their source
  (Michel, Gualandi and Avouac 2019 for 2008–2017; Costantino et al. 2023
  for 2017–2022), cited as reported.
- `timing = tremor`: episodes from 2022 onward, for which no per-event GNSS
  catalogue exists, clustered here from the PNSN tremor catalogue (Wech
  2021) with the parameters below. These are reported as a separate group.

Station selection for the retention measure uses the tremor band: the
median tremor longitude per latitude bin over the whole export
(`regions/pnw/cascadia_tremor_band.csv`). Both files are written once by
`gnssdl bench episodes --tremor <export.csv>` and committed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The Cascadia files belong to the PNW region (regions/pnw.toml); these are that region's
# paths, for the builder below and as defaults. Tracked: every value is reviewable.
EPISODE_DIR = Path(__file__).parent.parent / "regions" / "pnw"
EPISODES = EPISODE_DIR / "cascadia_sse.csv"
BAND = EPISODE_DIR / "cascadia_tremor_band.csv"
CATALOGUE = EPISODE_DIR / "cascadia_gnss_catalogue.csv"

# Tremor clustering, fixed 2026-11-05 before any PNW run. Epicentres are
# 5-minute detections; a (0.5° latitude bin, day) cell is active with at least
# 20 of them; cells in adjacent bins within 3 days join one episode; an episode
# keeps at least 7 days and 1,500 epicentres. Its latitude extent is the 5th to
# 95th percentile of its epicentres.
LAT_BIN_DEG = 0.5
MIN_PER_CELL = 20
GAP_DAYS = 3
MIN_DAYS = 7
MIN_EPICENTRES = 1500
EXTENT_PERCENTILES = (5, 95)
TREMOR_FROM = pd.Timestamp("2022-01-01")       # GNSS catalogues cover the years before
LAT_RANGE = (40.0, 50.5)                       # whole margin, so extents are not clipped at the box
BAND_BIN_DEG = 0.25
BAND_MIN_EPICENTRES = 200

EPISODE_COLUMNS = ["id", "source", "timing", "start", "end", "lat_min", "lat_max", "note"]


def read_tremor(path: Path) -> pd.DataFrame:
    """PNSN tremor export (lat, lon, depth, starttime, energy, duration) as
    lat, lon, day; malformed rows are dropped."""
    df = pd.read_csv(path, skipinitialspace=True, usecols=["lat", "lon", "starttime"],
                     on_bad_lines="skip", dtype={"starttime": str})
    df["day"] = pd.to_datetime(df.starttime.str.strip(), errors="coerce").dt.normalize()
    df = df.dropna(subset=["lat", "lon", "day"])
    df = df[(df.lat >= LAT_RANGE[0]) & (df.lat <= LAT_RANGE[1])]
    return df[["lat", "lon", "day"]].reset_index(drop=True)


def tremor_band(tremor: pd.DataFrame) -> pd.DataFrame:
    """Median tremor longitude per latitude bin: the trench-parallel axis the
    retention measure takes its 150 km station band around."""
    b = np.floor(tremor.lat / BAND_BIN_DEG).astype(int)
    g = tremor.groupby(b)
    out = pd.DataFrame({"lat": (g.lat.count().index + 0.5) * BAND_BIN_DEG,
                        "lon": g.lon.median().values, "n": g.lon.count().values})
    return out[out.n >= BAND_MIN_EPICENTRES].drop(columns="n").reset_index(drop=True)


def band_longitude(band: pd.DataFrame, lat: np.ndarray) -> np.ndarray:
    """Band longitude at each latitude, interpolated between bin centres and
    held constant beyond the ends."""
    b = band.sort_values("lat")
    return np.interp(lat, b.lat.values, b.lon.values)


def cluster_tremor(tremor: pd.DataFrame, start: pd.Timestamp | None = TREMOR_FROM) -> pd.DataFrame:
    """Episodes as connected components of active (latitude bin, day) cells."""
    t = tremor if start is None else tremor[tremor.day >= start]
    cells = (t.assign(b=np.floor(t.lat / LAT_BIN_DEG).astype(int))
              .groupby(["b", "day"]).size().rename("n").reset_index())
    cells = cells[cells.n >= MIN_PER_CELL].sort_values(["day", "b"]).reset_index(drop=True)
    days = cells.day.values.astype("datetime64[D]").astype(np.int64)
    bins = cells.b.values
    parent = np.arange(len(cells))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cells)):
        j = i + 1
        while j < len(cells) and days[j] - days[i] <= GAP_DAYS:
            if abs(int(bins[j]) - int(bins[i])) <= 1:
                parent[find(i)] = find(j)
            j += 1
    cells["comp"] = [find(i) for i in range(len(cells))]
    key = pd.MultiIndex.from_arrays([cells.b, cells.day])
    comp = pd.Series(cells.comp.values, index=key)
    tb = np.floor(t.lat / LAT_BIN_DEG).astype(int)
    member = comp.reindex(pd.MultiIndex.from_arrays([tb, t.day])).values
    epi = t.assign(comp=member).dropna(subset=["comp"])
    rows = []
    for _, g in epi.groupby("comp"):
        d0, d1 = g.day.min(), g.day.max()
        if (d1 - d0).days + 1 < MIN_DAYS or len(g) < MIN_EPICENTRES:
            continue
        lo, hi = np.percentile(g.lat, EXTENT_PERCENTILES)
        rows.append({"start": d0.date(), "end": d1.date(), "lat_min": round(float(lo), 2),
                     "lat_max": round(float(hi), 2), "n_epicentres": len(g)})
    return pd.DataFrame(rows, columns=["start", "end", "lat_min", "lat_max", "n_epicentres"]) \
        .sort_values(["start", "lat_min"]).reset_index(drop=True)


def build(tremor_path: Path, catalogue_path: Path = CATALOGUE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fixed episode list (GNSS-catalogued rows, then tremor-derived rows
    from TREMOR_FROM) and the tremor band."""
    tremor = read_tremor(tremor_path)
    cat = pd.read_csv(catalogue_path, dtype={"note": str}).fillna({"note": ""})
    cat["timing"] = "gnss"
    cat = cat[cat.start < str(TREMOR_FROM.date())]
    tr = cluster_tremor(tremor)
    tr = pd.DataFrame({"id": [f"T{d:%Y%m%d}" for d in pd.to_datetime(tr.start)],
                       "source": "PNSN tremor (Wech 2021), clustered", "timing": "tremor",
                       "start": tr.start.astype(str), "end": tr.end.astype(str),
                       "lat_min": tr.lat_min, "lat_max": tr.lat_max,
                       "note": [f"{n} epicentres" for n in tr.n_epicentres]})
    episodes = pd.concat([cat[EPISODE_COLUMNS], tr[EPISODE_COLUMNS]], ignore_index=True)
    return episodes, tremor_band(tremor)


def load_episodes(path: Path = EPISODES) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"note": str}).fillna({"note": ""})
    df["start"] = pd.to_datetime(df.start)
    df["end"] = pd.to_datetime(df.end)
    return df


def load_band(path: Path = BAND) -> pd.DataFrame:
    return pd.read_csv(path)


EXCLUSION_MARGIN_DAYS = 10     # days before the start and after the end, the measure's windows
EXCLUSION_BAND_KM = 150.0      # east-west distance from the tremor band axis, as the measure


def training_exclusion(lat: np.ndarray, lon: np.ndarray, days: np.ndarray, episodes: pd.DataFrame,
                       band: pd.DataFrame, timing: tuple[str, ...] = ("gnss",)) -> np.ndarray:
    """S × T mask of cells barred from training (contract §1.6): stations in
    an episode's latitude range and within EXCLUSION_BAND_KM of the tremor
    band, on the days from EXCLUSION_MARGIN_DAYS before its start to the same
    after its end. Only episodes with `timing` in the given set count, so the
    tremor-only episodes (2022 onward, outside the training years) are
    left alone unless asked for."""
    from gnssdl.ngl import EARTH_RADIUS_KM
    days = pd.DatetimeIndex(days)
    lon_axis = band_longitude(band, lat)
    off_km = np.abs(np.radians(lon - lon_axis)) * np.cos(np.radians(lat)) * EARTH_RADIUS_KM
    out = np.zeros((len(lat), len(days)), dtype=bool)
    for ep in episodes[episodes.timing.isin(timing)].itertuples(index=False):
        sta = (lat >= ep.lat_min) & (lat <= ep.lat_max) & (off_km <= EXCLUSION_BAND_KM)
        w0 = pd.Timestamp(ep.start) - pd.Timedelta(days=EXCLUSION_MARGIN_DAYS)
        w1 = pd.Timestamp(ep.end) + pd.Timedelta(days=EXCLUSION_MARGIN_DAYS)
        day = (days >= w0) & (days <= w1)
        if sta.any() and day.any():
            out[np.ix_(sta, day)] = True
    return out
