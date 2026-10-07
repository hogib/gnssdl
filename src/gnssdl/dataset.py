"""The benchmark data cube (docs/00-benchmark-contract.md §1).

Turns one tenv3 file per station into a stations × days × components array
of trajectory residuals, plus everything a model needs to be scored fairly:
uncertainties, availability, day/station splits, the Ridgecrest exclusion
and neighbour graphs for each exclusion radius.

Every choice that could leak validation or test information is made here,
so the constants below are the contract, not tuning knobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gnssdl import model
from gnssdl.ngl import EARTH_RADIUS_KM

START = pd.Timestamp("2008-01-01")
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_END = pd.Timestamp("2021-12-31")

# 400 km reproduces the fixed distance of Bachelot et al. (2025).
RADII_KM = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0)
K_NEIGHBOURS = 16
# Neighbours are searched up to this far beyond the exclusion radius, so the
# search limit is R + NEIGHBOUR_SEARCH_KM (300 km at R = 0, 700 km at 400).
NEIGHBOUR_SEARCH_KM = 300.0


def neighbour_limit_km(radius_km: float) -> float:
    return radius_km + NEIGHBOUR_SEARCH_KM

HELDOUT_FRACTION = 0.10
HELDOUT_SEED = 0

MIN_TRAIN_EPOCHS = 365
FALLBACK_FIT_DAYS = 730
MIN_STEP_SIDE_EPOCHS = 30

# Station quality control. The QC fit removes earthquake steps as well, so a
# genuine coseismic offset does not count against a station; what is left
# above this horizontal robust scale is non-tectonic site motion (subsidence,
# landslides, unstable monuments) or a broken record.
QC_MAX_HORIZONTAL_SCALE_MM = 10.0
# Vertical noise is ~3x horizontal. Needed because groundwater subsidence
# (e.g. CRCN, Corcoran: 1.8 m in 2011-2019) is nearly invisible horizontally.
QC_MAX_VERTICAL_SCALE_MM = 30.0
# Stations closer than this are treated as one site; only the one with the
# most station-days is kept.
TWIN_DISTANCE_KM = 1.0

SCREEN_WINDOW_DAYS = 61
SCREEN_K = 5.0

RIDGECREST_LATLON = (35.77, -117.60)
RIDGECREST_RADIUS_KM = 150.0
RIDGECREST_WINDOW = (pd.Timestamp("2019-07-01"), pd.Timestamp("2020-06-30"))

COMPONENTS = ("e", "n", "u")
SIGMA_COLS = ("sig_e", "sig_n", "sig_u")


@dataclass
class Cube:
    r: np.ndarray            # S×T×3 float32, mm, NaN where missing
    sigma: np.ndarray        # S×T×3 float32, mm, NaN where missing
    avail: np.ndarray        # S×T bool
    sta: np.ndarray          # S str
    lat: np.ndarray          # S float64
    lon: np.ndarray          # S float64
    days: np.ndarray         # T datetime64[D]
    radii: np.ndarray        # NR float32, km
    nbr_idx: np.ndarray      # NR×S×K int32, -1 pads
    nbr_dist: np.ndarray     # NR×S×K float32, km
    nbr_az: np.ndarray       # NR×S×K float32, radians
    split_day: np.ndarray    # T int8: 0 train, 1 validation, 2 test
    split_sta: np.ndarray    # S int8: 0 seen, 1 held-out
    exclude: np.ndarray      # S×T bool: barred from training
    # Earthquake offsets estimated in the training-period fit and kept in r
    # (one row per offset). quiet_residuals() subtracts them for training.
    qstep_sta: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    qstep_day: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    qstep_amp: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{f.name: getattr(self, f.name) for f in fields(self)})
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Cube":
        with np.load(path, allow_pickle=False) as z:
            return cls(**{f.name: z[f.name] for f in fields(cls) if f.name in z.files})

    def radius_index(self, radius_km: float) -> int:
        hits = np.flatnonzero(np.isclose(self.radii, radius_km))
        if not len(hits):
            raise KeyError(f"no neighbour graph for R = {radius_km} km; have {self.radii.tolist()}")
        return int(hits[0])


def crop_days(cube: Cube, start: int, stop: int) -> Cube:
    """The cube restricted to day indices [start, stop). Station metadata and
    neighbour graphs are unchanged; everything indexed by day is sliced."""
    from dataclasses import replace

    sl = slice(start, stop)
    return replace(cube, r=cube.r[:, sl], sigma=cube.sigma[:, sl], avail=cube.avail[:, sl],
                   days=cube.days[sl], split_day=cube.split_day[sl], exclude=cube.exclude[:, sl])


# --------------------------------------------------------------------------- #
# Per-station processing
# --------------------------------------------------------------------------- #


@dataclass
class StationResult:
    r: pd.DataFrame          # mm, indexed by date, columns e n u
    sigma: pd.DataFrame      # mm, indexed by date, columns e n u
    fallback_fit: bool       # baseline fitted outside the training period
    n_screened: int          # days removed by outlier screening
    qc_scale_mm: float       # horizontal robust scale of the QC fit on fit epochs
    qc_scale_u_mm: float     # vertical robust scale of the QC fit on fit epochs
    quake_steps: list = field(default_factory=list)   # (date, [E,N,U] mm) kept in r


def screen_outliers(r: pd.DataFrame) -> pd.Series:
    """True for days where any component is more than SCREEN_K robust σ from
    its running median. The scale is a running median of absolute deviations
    from the running median, an inexpensive stand-in for a windowed MAD."""
    daily = r.asfreq("D")
    bad = pd.Series(False, index=daily.index)
    for c in COMPONENTS:
        x = daily[c]
        med = x.rolling(SCREEN_WINDOW_DAYS, center=True, min_periods=15).median()
        dev = (x - med).abs()
        mad = dev.rolling(SCREEN_WINDOW_DAYS, center=True, min_periods=15).median()
        bad |= (dev > SCREEN_K * 1.4826 * mad).fillna(False)
    return bad.reindex(r.index, fill_value=False)


def _robust_scale(x: np.ndarray) -> float:
    return float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))


def process_station(
    series: pd.DataFrame,
    steps: pd.DataFrame,
    exclude_window: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> StationResult | None:
    """Residuals for one station on its own dates (from START on).

    `steps` holds this station's rows of NGL's steps file. Returns None when
    the station has too little data to fit a baseline at all. Quality control
    is reported in `qc_scale_mm`; the caller decides whether to drop.
    """
    series = series.loc[START:]
    if len(series) < MIN_TRAIN_EPOCHS:
        return None
    idx = series.index
    fit_mask = idx <= TRAIN_END
    if exclude_window is not None:
        fit_mask &= ~((idx >= exclude_window[0]) & (idx <= exclude_window[1]))

    fallback = fit_mask.sum() < MIN_TRAIN_EPOCHS
    if fallback:
        # Too little training data: fit on the start of the record instead,
        # the first FALLBACK_FIT_DAYS or the first MIN_TRAIN_EPOCHS epochs,
        # whichever reaches further (sparse early records need the latter).
        # The caller forces such stations to be held out, so this baseline
        # never touches data they are scored on as "seen".
        cutoff = max(idx[0] + pd.Timedelta(days=FALLBACK_FIT_DAYS - 1), idx[MIN_TRAIN_EPOCHS - 1])
        fit_mask = idx <= cutoff

    quake = sorted(set(steps.loc[steps.kind == "quake", "date"]))
    equipment = sorted(set(steps.loc[steps.kind == "equipment", "date"]) - set(quake))
    kept: list = []
    try:
        r, _ = model.trajectory_residuals(
            series, fit_mask,
            step_dates=quake + equipment,
            keep_step_dates=quake,
            equipment_dates=equipment,
            min_epochs_each_side=MIN_STEP_SIDE_EPOCHS,
            min_fit_epochs=MIN_TRAIN_EPOCHS,
            kept_steps_out=kept,
        )
        qc, _ = model.trajectory_residuals(
            series, fit_mask,
            step_dates=quake + equipment,
            min_epochs_each_side=MIN_STEP_SIDE_EPOCHS,
            min_fit_epochs=MIN_TRAIN_EPOCHS,
        )
    except ValueError:
        return None
    qc_fit = qc[fit_mask]
    qc_scale = max(_robust_scale(qc_fit["e"].to_numpy()), _robust_scale(qc_fit["n"].to_numpy()))
    qc_scale_u = _robust_scale(qc_fit["u"].to_numpy())

    sigma = pd.DataFrame(
        {c: series[s].to_numpy() * 1000.0 for c, s in zip(COMPONENTS, SIGMA_COLS)}, index=idx
    )
    bad = screen_outliers(r)
    keep = ~bad.to_numpy()
    to_date = {model._to_dec(d): d for d in quake}
    quake_steps = [(to_date[t], amp) for t, amp in kept if t in to_date]
    return StationResult(r[keep], sigma[keep], bool(fallback), int(bad.sum()), qc_scale, qc_scale_u,
                         quake_steps)


# --------------------------------------------------------------------------- #
# Network-level pieces
# --------------------------------------------------------------------------- #


def distance_azimuth(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S×S great-circle distance (km) and initial bearing (rad) from row to
    column station."""
    p, l = np.radians(lat), np.radians(lon)
    dp = p[None, :] - p[:, None]
    dl = l[None, :] - l[:, None]
    a = np.sin(dp / 2) ** 2 + np.cos(p[:, None]) * np.cos(p[None, :]) * np.sin(dl / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    az = np.arctan2(
        np.sin(dl) * np.cos(p[None, :]),
        np.cos(p[:, None]) * np.sin(p[None, :]) - np.sin(p[:, None]) * np.cos(p[None, :]) * np.cos(dl),
    )
    return d, az


def neighbour_graphs(
    d: np.ndarray, az: np.ndarray, radii: tuple[float, ...] = RADII_KM, k: int = K_NEIGHBOURS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each radius R, the k nearest stations with R < d ≤ R + NEIGHBOUR_SEARCH_KM.
    Strict inequality also drops a station's own row and exactly co-located
    duplicates at R = 0."""
    s = d.shape[0]
    idx = np.full((len(radii), s, k), -1, dtype=np.int32)
    dist = np.full((len(radii), s, k), np.nan, dtype=np.float32)
    azi = np.full((len(radii), s, k), np.nan, dtype=np.float32)
    for ri, rad in enumerate(radii):
        dd = np.where((d > rad) & (d <= neighbour_limit_km(rad)), d, np.inf)
        m = min(k, s)
        order = np.argsort(dd, axis=1, kind="stable")[:, :m]
        valid = np.isfinite(np.take_along_axis(dd, order, axis=1))
        idx[ri, :, :m] = np.where(valid, order, -1)
        dist[ri, :, :m] = np.where(valid, np.take_along_axis(d, order, axis=1), np.nan)
        azi[ri, :, :m] = np.where(valid, np.take_along_axis(az, order, axis=1), np.nan)
    return idx, dist, azi


def twin_drops(d: np.ndarray, sta: np.ndarray, n_days: np.ndarray) -> dict[int, int]:
    """Clusters of stations closer than TWIN_DISTANCE_KM (connected
    components). Returns {dropped index: kept index}; the kept station has the
    most station-days, ties broken by id."""
    s = len(sta)
    parent = list(range(s))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(*np.nonzero(np.triu(d < TWIN_DISTANCE_KM, 1))):
        parent[root(a)] = root(b)
    clusters: dict[int, list[int]] = {}
    for i in range(s):
        clusters.setdefault(root(i), []).append(i)
    drops = {}
    for members in clusters.values():
        if len(members) > 1:
            keep = min(members, key=lambda i: (-n_days[i], sta[i]))
            drops.update({i: keep for i in members if i != keep})
    return drops


def farthest_point_sample(d: np.ndarray, n: int, seed: int = HELDOUT_SEED) -> np.ndarray:
    """n station indices spread over the network: start at a random station,
    repeatedly add the one farthest from those already chosen."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(d.shape[0]))]
    nearest = d[chosen[0]].copy()
    for _ in range(n - 1):
        j = int(np.argmax(nearest))
        chosen.append(j)
        nearest = np.minimum(nearest, d[j])
    return np.array(sorted(chosen))


SPLIT_NAMES = {0: "train", 1: "validation", 2: "test", 3: "prospective"}


def day_splits(days: pd.DatetimeIndex, freeze_date: pd.Timestamp | None = None) -> np.ndarray:
    """0 train, 1 validation, 2 test, 3 prospective (days after the freeze
    date: data recorded after the method was frozen, scored once at the end)."""
    out = np.full(len(days), 2, dtype=np.int8)
    out[days <= VAL_END] = 1
    out[days <= TRAIN_END] = 0
    if freeze_date is not None:
        if freeze_date <= VAL_END:
            raise ValueError("the freeze date must fall after the validation period")
        out[days > freeze_date] = 3
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_cube(
    stations: pd.DataFrame,
    load_series: Callable[[str], pd.DataFrame],
    steps: pd.DataFrame,
    end: pd.Timestamp | None = None,
    progress: Callable[[int, int], None] | None = None,
    freeze_date: pd.Timestamp | None = None,
) -> tuple[Cube, dict]:
    """`stations` needs columns sta, lat, lon. Returns the cube and a summary."""
    stations = stations.sort_values("sta").reset_index(drop=True)
    rc_lat, rc_lon = RIDGECREST_LATLON
    d_rc, _ = distance_azimuth(
        np.r_[rc_lat, stations.lat.to_numpy()], np.r_[rc_lon, stations.lon.to_numpy()]
    )
    near_rc = d_rc[0, 1:] <= RIDGECREST_RADIUS_KM

    results: dict[str, StationResult] = {}
    dropped: dict[str, list] = {"no_baseline": [], "qc": [], "twin": []}
    steps_by_sta = dict(tuple(steps.groupby("sta")))
    empty_steps = steps.iloc[:0]
    for i, row in stations.iterrows():
        res = process_station(
            load_series(row.sta),
            steps_by_sta.get(row.sta, empty_steps),
            RIDGECREST_WINDOW if near_rc[i] else None,
        )
        if res is None:
            dropped["no_baseline"].append(row.sta)
        elif res.qc_scale_mm > QC_MAX_HORIZONTAL_SCALE_MM or res.qc_scale_u_mm > QC_MAX_VERTICAL_SCALE_MM:
            dropped["qc"].append({
                "sta": row.sta, "lat": round(float(row.lat), 3), "lon": round(float(row.lon), 3),
                "horizontal_scale_mm": round(res.qc_scale_mm, 1),
                "vertical_scale_mm": round(res.qc_scale_u_mm, 1),
            })
        else:
            results[row.sta] = res
        if progress:
            progress(i + 1, len(stations))

    cand = stations[stations.sta.isin(results)].reset_index(drop=True)
    d_c, _ = distance_azimuth(cand.lat.to_numpy(float), cand.lon.to_numpy(float))
    n_days = np.array([len(results[s].r) for s in cand.sta])
    for i, k in sorted(twin_drops(d_c, cand.sta.to_numpy(str), n_days).items()):
        dropped["twin"].append({"sta": cand.sta[i], "kept": cand.sta[k], "km": round(float(d_c[i, k]), 3)})
        del results[cand.sta[i]]

    kept = stations[stations.sta.isin(results)].reset_index(drop=True)
    last = max(r.r.index.max() for r in results.values())
    days = pd.date_range(START, end or last, freq="D")
    S, T = len(kept), len(days)

    r = np.full((S, T, 3), np.nan, dtype=np.float32)
    sigma = np.full((S, T, 3), np.nan, dtype=np.float32)
    for i, sta in enumerate(kept.sta):
        res = results[sta]
        pos = days.get_indexer(res.r.index)
        ok = pos >= 0
        r[i, pos[ok]] = res.r.to_numpy(np.float32)[ok]
        sigma[i, pos[ok]] = res.sigma.to_numpy(np.float32)[ok]
    avail = np.isfinite(r).all(axis=2)

    qs, qd, qa = [], [], []
    for i, sta in enumerate(kept.sta):
        for date, amp in results[sta].quake_steps:
            pos = days.searchsorted(date)
            if pos < T:
                qs.append(i); qd.append(int(pos)); qa.append(amp)

    lat, lon = kept.lat.to_numpy(float), kept.lon.to_numpy(float)
    d, az = distance_azimuth(lat, lon)
    nbr_idx, nbr_dist, nbr_az = neighbour_graphs(d, az)

    split_sta = np.zeros(S, dtype=np.int8)
    n_held = max(1, round(HELDOUT_FRACTION * S))
    split_sta[farthest_point_sample(d, n_held)] = 1
    fallback = np.array([results[s].fallback_fit for s in kept.sta])
    split_sta[fallback] = 1

    exclude = np.zeros((S, T), dtype=bool)
    in_window = (days >= RIDGECREST_WINDOW[0]) & (days <= RIDGECREST_WINDOW[1])
    exclude[np.ix_(near_rc[stations.sta.isin(results).to_numpy()], in_window)] = True

    cube = Cube(
        r=r, sigma=sigma, avail=avail, sta=kept.sta.to_numpy(str), lat=lat, lon=lon,
        days=days.to_numpy("datetime64[D]"), radii=np.array(RADII_KM, dtype=np.float32),
        nbr_idx=nbr_idx, nbr_dist=nbr_dist, nbr_az=nbr_az,
        split_day=day_splits(days, freeze_date), split_sta=split_sta, exclude=exclude,
        qstep_sta=np.array(qs, dtype=np.int32), qstep_day=np.array(qd, dtype=np.int32),
        qstep_amp=np.array(qa, dtype=np.float32).reshape(-1, 3),
    )

    n_nbr = (nbr_idx >= 0).sum(axis=2)
    summary = {
        "stations_in": int(len(stations)),
        "stations_kept": S,
        "stations_dropped": dropped,
        "days": T,
        "first_day": str(days[0].date()),
        "last_day": str(days[-1].date()),
        "station_days": int(avail.sum()),
        "coverage": round(float(avail.mean()), 4),
        "heldout_stations": int(split_sta.sum()),
        "heldout_from_fallback_fit": int(fallback.sum()),
        "screened_days": int(sum(r.n_screened for r in results.values())),
        "kept_quake_offsets": len(qs),
        "ridgecrest_stations": int(exclude.any(axis=1).sum()),
        "colocated_pairs_under_1km": int(((d < TWIN_DISTANCE_KM).sum() - S) // 2),
        "neighbours_per_radius": {
            f"{rad:g}": {
                "median": float(np.median(n_nbr[ri])),
                "stations_with_fewer_than_4": int((n_nbr[ri] < 4).sum()),
            }
            for ri, rad in enumerate(RADII_KM)
        },
        "config": {
            "start": str(START.date()), "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()), "radii_km": list(RADII_KM),
            "freeze_date": str(freeze_date.date()) if freeze_date is not None else None,
            "k": K_NEIGHBOURS, "neighbour_search_km_beyond_radius": NEIGHBOUR_SEARCH_KM,
            "heldout_fraction": HELDOUT_FRACTION, "heldout_seed": HELDOUT_SEED,
            "min_train_epochs": MIN_TRAIN_EPOCHS, "fallback_fit_days": FALLBACK_FIT_DAYS,
            "qc_max_horizontal_scale_mm": QC_MAX_HORIZONTAL_SCALE_MM,
            "qc_max_vertical_scale_mm": QC_MAX_VERTICAL_SCALE_MM,
            "twin_distance_km": TWIN_DISTANCE_KM,
            "screen_window_days": SCREEN_WINDOW_DAYS, "screen_k": SCREEN_K,
            "ridgecrest_latlon": list(RIDGECREST_LATLON),
            "ridgecrest_radius_km": RIDGECREST_RADIUS_KM,
            "ridgecrest_window": [str(t.date()) for t in RIDGECREST_WINDOW],
        },
    }
    return cube, summary


def write_summary(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n")
