"""Signal-preservation tests (contract §5): planted transients, seasonal kept, real offsets, slow-slip episodes.

Both measure how much real signal survives cleaning, i.e. stays in the
leave-one-out residual r - r̂ instead of being absorbed into the prediction.

Injections are paired: every injected run is compared with the same model on
the uninjected cube, so

    ρ = ⟨Δ, s⟩ / ⟨s, s⟩,   Δ = (r+s − P(r+s)) − (r − P(r))

measures the model's response to the injected signal s alone, without the
noise term ⟨r − P(r), s⟩ that the unpaired definition carries. For a linear
model this is exactly ⟨s − P(s), s⟩ / ⟨s, s⟩.

Many transients are injected per run. Within a run they are separated in
time (slots) and, within a slot, in space, far enough that no station near
one transient uses a neighbour near another.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gnssdl.bench.base import Reconstructor
from gnssdl.bench.score import _split_fast_slow
from gnssdl.dataset import EARTH_RADIUS_KM, Cube, crop_days, distance_azimuth

AMPLITUDES_MM = (2.0, 5.0, 10.0)
FOOTPRINTS_KM = (10.0, 25.0, 50.0, 100.0)
DURATIONS_DAYS = (10, 30, 90)
CENTRES_PER_CELL = 50
HOLD_DAYS = 60
SLOT_DAYS = 2 * max(DURATIONS_DAYS) + HOLD_DAYS + 60
FOOTPRINT_CUTOFF = 3.0         # s is set to zero beyond 3 L
SCORE_RADIUS = 2.0             # ρ uses stations within 2 L of the centre
SEED = 20260102

CROP_MARGIN_DAYS = 60          # context kept around cropped windows (M2's running median)



@dataclass(frozen=True)
class Injection:
    id: int
    amplitude_mm: float
    footprint_km: float
    duration_days: int
    lat: float
    lon: float
    t0: int            # day index where the rise starts
    ue: float          # horizontal unit vector, east
    un: float          # horizontal unit vector, north


def time_profile(n_days: int, duration: int) -> np.ndarray:
    """Raised-cosine rise over `duration` days, HOLD_DAYS at 1, raised-cosine
    fall over `duration` days, then zero. Length n_days."""
    rise = 0.5 * (1 - np.cos(np.pi * np.arange(duration) / duration))
    g = np.r_[rise, np.ones(HOLD_DAYS), rise[::-1]]
    out = np.zeros(n_days)
    out[: min(n_days, len(g))] = g[:n_days]
    return out


def signal_of(cube: Cube, inj: Injection) -> tuple[np.ndarray, np.ndarray, slice]:
    """(station indices, s[stations, days, 3] in mm, day slice) for one injection."""
    d = distance_azimuth(np.r_[inj.lat, cube.lat], np.r_[inj.lon, cube.lon])[0][0, 1:]
    sta = np.flatnonzero(d <= FOOTPRINT_CUTOFF * inj.footprint_km)
    length = 2 * inj.duration_days + HOLD_DAYS
    days = slice(inj.t0, min(inj.t0 + length, cube.r.shape[1]))
    g = time_profile(days.stop - days.start, inj.duration_days)
    f = inj.amplitude_mm * np.exp(-d[sta] ** 2 / (2 * inj.footprint_km**2))
    s = np.zeros((len(sta), len(g), 3), dtype=np.float32)
    s[..., 0] = (f[:, None] * g[None, :] * inj.ue)
    s[..., 1] = (f[:, None] * g[None, :] * inj.un)
    return sta, s, days


PERIODS = {"validation": 1, "test": 2, "prospective": 3}


def footprint(cube: Cube, inj: Injection) -> tuple[np.ndarray, np.ndarray]:
    """(distance of every station to the centre in km, bool mask of stations
    inside the footprint, where the planted signal is nonzero)."""
    d = distance_azimuth(np.r_[inj.lat, cube.lat], np.r_[inj.lon, cube.lon])[0][0, 1:]
    return d, d <= FOOTPRINT_CUTOFF * inj.footprint_km


def affected(cube: Cube, inside: np.ndarray, radius_km: float | None, model=None) -> np.ndarray:
    """Stations whose cleaned series a transient can change: those inside its
    footprint and those with a neighbour inside it (neighbour lists at
    `radius_km`). None means a filter that uses every station (e.g. a
    regional stack), which every transient can affect everywhere. A model
    that defines `affected_by` (the audit filters) decides for itself."""
    if model is not None and hasattr(model, "affected_by"):
        return model.affected_by(cube, inside)
    if radius_km is None:
        return np.ones_like(inside)
    k = getattr(model, "n_neighbours", 16)
    nbr = cube.nbr_idx[cube.radius_index(radius_km)][:, :k]
    return inside | ((nbr >= 0) & inside[np.where(nbr >= 0, nbr, 0)]).any(axis=1)


def plan_injections(
    cube: Cube, keep_sta: np.ndarray, centres_per_cell: int = CENTRES_PER_CELL, seed: int = SEED,
    period: str = "validation", radius_km: float | None = 0.0, model=None,
) -> list[list[Injection]]:
    """All injections of the grid, packed into runs. Centres are placed at
    random scored stations; transient start times fall in slots of `period`.

    Within a slot, two transients share a run only if no station is affected
    by both (`affected` at the filter's exclusion radius), so neither changes
    a station the other is scored on. Larger radii reach further, so they
    pack fewer transients per run. A transient's id is given when it is
    drawn, before packing, so the same id is the same transient (centre,
    direction, width, amplitude, duration) for every filter and radius; only
    its start slot depends on the packing. Until 2026-11-12 ids were given in
    packing order and differed between filters. Design decisions use the validation
    years; the test and prospective periods are for final reporting."""
    rng = np.random.default_rng(seed)
    test_days = np.flatnonzero(cube.split_day == PERIODS[period])
    n_slots = len(test_days) // SLOT_DAYS
    if n_slots < 1:
        raise ValueError(f"{period} period shorter than one injection slot")
    candidates = np.flatnonzero(keep_sta & cube.avail[:, test_days].any(axis=1))

    runs: list[list[Injection]] = []
    next_id = 0
    for L in FOOTPRINTS_KM:
        pending = []
        for A in AMPLITUDES_MM:
            for D in DURATIONS_DAYS:
                for _ in range(centres_per_cell):
                    c = int(rng.choice(candidates))
                    ang = rng.uniform(0, 2 * np.pi)
                    inj = Injection(next_id, A, L, D, float(cube.lat[c]), float(cube.lon[c]), 0,
                                    float(np.cos(ang)), float(np.sin(ang)))
                    next_id += 1
                    pending.append((inj, affected(cube, footprint(cube, inj)[1], radius_km, model)))
        rng.shuffle(pending)
        while pending:
            run: list[Injection] = []
            for slot in range(n_slots):
                taken = np.zeros(len(cube.sta), dtype=bool)
                rest = []
                for inj, aff in pending:
                    if not (aff & taken).any():
                        taken |= aff
                        run.append(replace(inj, t0=int(test_days[slot * SLOT_DAYS])))
                    else:
                        rest.append((inj, aff))
                pending = rest
                if not pending:
                    break
            runs.append(run)
    return runs


def clean(model: Reconstructor, cube: Cube) -> np.ndarray:
    """Leave-one-out prediction for every station at once (contract §2.1).

    Valid only for models that never read the target station's own values:
    their prediction for a station on the unhidden cube already equals the
    prediction with that station hidden. The harness's leak checks verify
    this property (the target is poisoned in `check_radius`). Reference
    filters (contract §2.1 exception) read the target on purpose and are
    applied to the unhidden cube as they are."""
    if not (model.never_reads_target or model.reference_filter):
        raise NotImplementedError(f"{model.name}: per-station leave-one-out cleaning not implemented")
    return model.predict(cube, np.zeros(cube.avail.shape, dtype=bool))


def injection_scores(
    model: Reconstructor, cube: Cube, keep_sta: np.ndarray, runs: list[list[Injection]], log=None,
) -> pd.DataFrame:
    """One row per injection: ρ over the scored stations within 2 L and the
    transient's days, horizontal components. Works on the cube cropped to
    the injected days (plus a margin), which gives the same ρ faster."""
    t_all = [i.t0 for run in runs for i in run]
    lo = max(0, min(t_all) - CROP_MARGIN_DAYS)
    hi = min(cube.r.shape[1], max(t_all) + 2 * max(DURATIONS_DAYS) + HOLD_DAYS + CROP_MARGIN_DAYS)
    cube = crop_days(cube, lo, hi)
    runs = [[replace(i, t0=i.t0 - lo) for i in run] for run in runs]
    base = cube.r - clean(model, cube)
    rows = []
    for k, run in enumerate(runs):
        r = cube.r.copy()
        parts = []
        for inj in run:
            sta, s, days = signal_of(cube, inj)
            ok = cube.avail[np.ix_(sta, np.arange(days.start, days.stop))]
            r[sta, days] += np.where(ok[..., None], s, 0.0)
            parts.append((inj, sta, s, days))
        resid = r - clean(model, replace(cube, r=r))
        for inj, sta, s, days in parts:
            d_all, inside = footprint(cube, inj)
            d = d_all[sta]
            near = (d <= SCORE_RADIUS * inj.footprint_km) & keep_sta[sta]
            idx = sta[near]
            if not len(idx):
                continue
            ok = cube.avail[idx, days]
            delta = (resid[idx, days] - base[idx, days])[..., :2]
            sig = s[near][..., :2]
            num = float(np.sum(np.where(ok[..., None], delta * sig, 0.0)))
            den = float(np.sum(np.where(ok[..., None], sig * sig, 0.0)))
            if den <= 0:
                continue
            # centre, start day and direction make each row a full record of the planted signal
            row = {"id": inj.id, "amplitude_mm": inj.amplitude_mm, "footprint_km": inj.footprint_km,
                   "duration_days": inj.duration_days, "lat": round(inj.lat, 4), "lon": round(inj.lon, 4),
                   "start": str(pd.Timestamp(cube.days[inj.t0]).date()), "ue": round(inj.ue, 4),
                   "un": round(inj.un, 4), "stations": int(len(idx)), "rho": num / den}
            row.update(_spurious(cube, base, resid, inj, inside, days, keep_sta, model.radius_km, model))
            rows.append(row)
        if log:
            log(k + 1, len(runs))
    return pd.DataFrame(rows)


def _spurious(cube: Cube, base: np.ndarray, resid: np.ndarray, inj: Injection, inside: np.ndarray,
              days: slice, keep_sta: np.ndarray, radius_km: float | None, model=None) -> dict:
    """Signal created by the filter: at scored stations outside the
    footprint (where nothing was planted) that the transient can still
    affect through their neighbours, the peak horizontal change in the
    cleaned series, as a fraction of the planted amplitude."""
    outside = affected(cube, inside, radius_km, model) & ~inside & keep_sta
    idx = np.flatnonzero(outside)
    if not len(idx):
        return {"spurious_stations": 0, "spurious_max": 0.0, "spurious_median": 0.0}
    delta = (resid[idx, days] - base[idx, days])[..., :2]
    ok = cube.avail[idx, days]
    mag = np.where(ok, np.hypot(delta[..., 0], delta[..., 1]), 0.0)
    peak = mag.max(axis=1) / inj.amplitude_mm
    return {"spurious_stations": int(len(idx)), "spurious_max": float(peak.max()),
            "spurious_median": float(np.median(peak))}


EPISODE_WINDOW_DAYS = 10       # days averaged before the start and after the end of an episode
EPISODE_MIN_DAYS = 5           # available days needed in each window
EPISODE_BAND_KM = 150.0        # stations this close to the tremor band axis are scored
EPISODE_MIN_STATIONS = 10      # episodes with fewer scored stations are dropped in the summary
EPISODE_FAR_KM = 300.0         # scored stations farther than this from the band axis give the far reference
EPISODE_MIN_FAR = 10           # far stations needed for the reference; the episode is skipped with fewer
EPISODE_FLOOR_FACTOR = 2.0     # an episode is counted if its median slip estimate is this many noise floors
EPISODE_COLUMNS = ["episode", "timing", "sta", "lat", "band_km", "obs_mm", "slip_mm", "floor_mm", "far",
                   "num", "den", "retained"]


def episodes_in_period(cube: Cube, episodes: pd.DataFrame, period: str) -> pd.DataFrame:
    """The episodes a run on `period` may score. Each episode's measure window
    (EPISODE_WINDOW_DAYS before its start to the same after its end) must lie
    on days of the allowed splits: for "validation", the training days (the
    catalogued episodes there are excluded from training) and the validation
    days; for "test" or "prospective", that split only. So a development run
    never reads a test-year episode."""
    allowed = {"validation": (0, 1), "test": (2,), "prospective": (3,)}[period]
    days = pd.DatetimeIndex(cube.days)
    ok_day = np.isin(cube.split_day, allowed)
    keep = []
    for ep in episodes.itertuples(index=False):
        w0 = pd.Timestamp(ep.start) - pd.Timedelta(days=EPISODE_WINDOW_DAYS)
        w1 = pd.Timestamp(ep.end) + pd.Timedelta(days=EPISODE_WINDOW_DAYS)
        sel = (days >= w0) & (days <= w1)
        keep.append(bool(sel.any()) and bool(ok_day[sel].all()) and w0 >= days[0] and w1 <= days[-1])
    return episodes[np.array(keep, dtype=bool)].reset_index(drop=True)


def episode_retention(model: Reconstructor, cube: Cube, keep_sta: np.ndarray,
                      episodes: pd.DataFrame, band: pd.DataFrame) -> pd.DataFrame:
    """Contract §1.6 (measure revised 2026-11-12). For each listed slow-slip
    episode, a station's displacement d is its mean horizontal position over
    the 10 days after the episode's end minus the mean over the 10 days
    before its start. The windows are 40 to 144 days apart, so d also holds
    the motion the whole network shares over them (seasonal cycle, common
    mode). The far reference is the component-wise median of d over the
    scored stations more than EPISODE_FAR_KM east or west of the tremor band
    axis, at any latitude, where no slow slip is expected; the slow-slip
    estimate at a scored station within the episode's latitude range and
    EPISODE_BAND_KM of the axis is s = d - reference, taken on the
    unfiltered residuals only. One row per (episode, band station):
    `num` = ⟨kept, s⟩ and `den` = ⟨s, s⟩, with `kept` the displacement of the
    cleaned series (no reference subtracted), and `retained` = num / den.
    `floor_mm` is the median size of s over the far stations, the size the
    estimate has where there is no slip; `far` their number. Episodes with
    fewer than EPISODE_MIN_FAR far stations are skipped.

    The unfiltered series does not score 1: its `kept` is s + reference, and
    the reference projects on s by chance. A filter that removes exactly the
    reference scores 1."""
    from gnssdl.bench.episodes import band_longitude
    days = pd.DatetimeIndex(cube.days)
    lon_axis = band_longitude(band, cube.lat)
    # east-west distance from the band axis, along the parallel
    off_km = np.abs(np.radians(cube.lon - lon_axis)) * np.cos(np.radians(cube.lat)) * EARTH_RADIUS_KM
    far_sta = keep_sta & (off_km > EPISODE_FAR_KM)
    rows = []
    for ep in episodes.itertuples(index=False):
        start, end = pd.Timestamp(ep.start), pd.Timestamp(ep.end)
        w0 = start - pd.Timedelta(days=EPISODE_WINDOW_DAYS)
        w1 = end + pd.Timedelta(days=EPISODE_WINDOW_DAYS)
        if w0 < days[0] or w1 > days[-1]:
            continue
        sel = keep_sta & (cube.lat >= ep.lat_min) & (cube.lat <= ep.lat_max) & (off_km <= EPISODE_BAND_KM)
        if not sel.any():
            continue
        lo = int(days.searchsorted(w0 - pd.Timedelta(days=CROP_MARGIN_DAYS)))
        hi = int(days.searchsorted(w1 + pd.Timedelta(days=CROP_MARGIN_DAYS), side="right"))
        small = crop_days(cube, lo, hi)
        sdays = pd.DatetimeIndex(small.days)
        before = (sdays >= w0) & (sdays < start)
        after = (sdays > end) & (sdays <= w1)
        a0, a1 = small.avail & before, small.avail & after
        ok = (a0.sum(axis=1) >= EPISODE_MIN_DAYS) & (a1.sum(axis=1) >= EPISODE_MIN_DAYS)
        far = np.flatnonzero(far_sta & ok)
        if len(far) < EPISODE_MIN_FAR:
            continue

        def displacement(x: np.ndarray, i: int) -> np.ndarray:
            return (x[i, a1[i], :2].mean(axis=0) - x[i, a0[i], :2].mean(axis=0)).astype(np.float64)

        far_d = np.array([displacement(small.r, i) for i in far])
        ref = np.median(far_d, axis=0)
        floor = float(np.median(np.hypot(*(far_d - ref).T)))
        resid = small.r - clean(model, small)
        for i in np.flatnonzero(sel & ok):
            obs = displacement(small.r, i)
            slip = obs - ref
            den = float(np.sum(slip * slip))
            if den <= 0:
                continue
            num = float(np.sum(displacement(resid, i) * slip))
            rows.append({"episode": ep.id, "timing": ep.timing, "sta": str(small.sta[i]),
                         "lat": round(float(small.lat[i]), 3), "band_km": round(float(off_km[i]), 1),
                         "obs_mm": round(float(np.hypot(*obs)), 2), "slip_mm": round(float(np.sqrt(den)), 2),
                         "floor_mm": round(floor, 3), "far": int(len(far)),
                         "num": num, "den": den, "retained": num / den})
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (timing, episode): scored band stations, their median slip
    estimate, the noise floor, the pooled retention Σ⟨kept, s⟩ / Σ⟨s, s⟩ over
    the stations (the form of the planted-transient retention), and
    `counted`: at least EPISODE_MIN_STATIONS stations and a median slip
    estimate of at least EPISODE_FLOOR_FACTOR floors."""
    missing = {"num", "den", "slip_mm", "floor_mm"} - set(df.columns)
    if missing:
        raise ValueError(f"episode rows lack {sorted(missing)}: they were written before the measure of "
                         "2026-11-12 (contract §1.6); rerun `bench signal MODEL --episodes`")
    per = (df.groupby(["timing", "episode"], sort=True)
             .agg(stations=("num", "size"), slip_mm=("slip_mm", "median"), floor_mm=("floor_mm", "first"),
                  far=("far", "first"), num=("num", "sum"), den=("den", "sum")).reset_index())
    per["retained"] = per.num / per.den
    per["counted"] = (per.stations >= EPISODE_MIN_STATIONS) & (per.slip_mm >= EPISODE_FLOOR_FACTOR * per.floor_mm)
    return per.drop(columns=["num", "den"])


def episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per timing group: the median of the pooled retention over the counted
    episodes (see `episode_table`), their number and stations, and `weak`,
    the episodes with enough stations whose slip estimate is below the
    floor rule and which are therefore left out."""
    per = episode_table(df)
    out = []
    for timing, g in per.groupby("timing"):
        c = g[g.counted]
        out.append({"timing": timing, "episodes": len(c), "stations": int(c.stations.sum()),
                    "retained": float(c.retained.median()) if len(c) else np.nan,
                    "weak": int(((g.stations >= EPISODE_MIN_STATIONS) & ~g.counted).sum())})
    return pd.DataFrame(out, columns=["timing", "episodes", "stations", "retained", "weak"])


NOISE_MIN_DAYS = 180           # scored days a station needs for the noise-removed measure
NOISE_BLOCK_DAYS = 30          # block length of the noise bootstrap (bench.stats)


def noise_parts(model: Reconstructor, cube: Cube, keep_sta: np.ndarray, period: str = "validation",
                scale: np.ndarray | None = None, skip: np.ndarray | None = None,
                ) -> tuple[dict[str, float], pd.DataFrame]:
    """The noise-removed measure and what its bootstrap needs. Per scored
    station over `period`: 1 - rms(fast(r - r̂)) / rms(fast(r)), averaged over
    E/N/U, then the median over stations (the dict). The frame has one row
    per station and 30-day block: the number of days and the sums of squares
    of the fast part of the cleaned series (`clean_*`) and of r (`raw_*`)
    per component, from which `bench.stats` recomputes the measure on
    resampled stations and blocks.

    With `scale` (S×3 station noise scale) and `skip` (S×T cells near
    earthquakes), and for a filter that never reads the target, the frame
    also holds the leave-one-out error of `bench compare`: `err_days` and
    `err_*`, the sums of squares of the fast part of r - r̂ divided by the
    station's scale, over the period's cells outside `skip` (stations with at
    least one block of them). They are zero where this does not apply."""
    with_error = scale is not None and skip is not None and model.never_reads_target
    code = PERIODS[period]
    test = np.flatnonzero(cube.split_day == code)
    lo = max(0, test[0] - CROP_MARGIN_DAYS)
    hi = min(cube.r.shape[1], test[-1] + CROP_MARGIN_DAYS + 1)
    small = crop_days(cube, lo, hi)
    resid = small.r - clean(model, small)
    days = pd.DatetimeIndex(small.days)
    in_test = small.split_day == code
    first = int(np.flatnonzero(in_test)[0])
    out, rows = [], []
    for i in np.flatnonzero(keep_sta):
        cols = np.flatnonzero(small.avail[i] & in_test & ~small.exclude[i])
        if len(cols) < NOISE_MIN_DAYS:
            continue
        f_r, _ = _split_fast_slow(small.r[i, cols].astype(np.float64), days[cols])
        f_e, _ = _split_fast_slow(resid[i, cols].astype(np.float64), days[cols])
        ratio = np.sqrt(np.nanmean(f_e**2, axis=0)) / np.sqrt(np.nanmean(f_r**2, axis=0))
        out.append(1.0 - float(np.mean(ratio)))
        block = (cols - first) // NOISE_BLOCK_DAYS
        err = {}
        if with_error:
            ecols = cols[~skip[i, lo:hi][cols]]
            if len(ecols) >= NOISE_BLOCK_DAYS:
                fz, _ = _split_fast_slow(resid[i, ecols].astype(np.float64), days[ecols])
                z = fz / scale[i]
                ok = np.isfinite(z).all(axis=1)
                eb = (ecols[ok] - first) // NOISE_BLOCK_DAYS
                for b in np.unique(eb):
                    err[int(b)] = (int((eb == b).sum()), (z[ok][eb == b] ** 2).sum(axis=0))
        for b in np.unique(block):
            m = block == b
            c2, r2 = np.nansum(f_e[m] ** 2, axis=0), np.nansum(f_r[m] ** 2, axis=0)
            n_err, e2 = err.get(int(b), (0, np.zeros(3)))
            rows.append({"sta": str(small.sta[i]), "block": int(b), "days": int(m.sum()),
                         **{f"clean_{c}": float(c2[k]) for k, c in enumerate("enu")},
                         **{f"raw_{c}": float(r2[k]) for k, c in enumerate("enu")},
                         "err_days": n_err, **{f"err_{c}": float(e2[k]) for k, c in enumerate("enu")}})
    summary = {"noise_removed": float(np.median(out)) if out else np.nan, "stations": len(out)}
    return summary, pd.DataFrame(rows, columns=["sta", "block", "days", "clean_e", "clean_n", "clean_u",
                                                "raw_e", "raw_n", "raw_u", "err_days", "err_e", "err_n", "err_u"])


def noise_removed(model: Reconstructor, cube: Cube, keep_sta: np.ndarray,
                  period: str = "validation") -> dict[str, float]:
    """How much of the fast (< ~2 months) scatter cleaning removes, over
    `period`: per scored station, 1 - rms(fast(r - r̂)) / rms(fast(r)),
    averaged over E/N/U, then the median over stations. Computed the same
    way for every model, including reference filters that cannot be scored
    on the masks, so it is the x-axis of the noise-vs-signal trade-off."""
    return noise_parts(model, cube, keep_sta, period)[0]


OFFSET_WINDOW_DAYS = 10        # days averaged on each side of an earthquake
OFFSET_MIN_DAYS = 5            # available days needed on each side
OFFSET_MIN_MM = 3.0            # observed horizontal offset needed to score it
OFFSET_CLEAR_DAYS = 15         # no other catalogued step this close


def quake_offsets(model: Reconstructor, cube: Cube, keep_sta: np.ndarray, steps: pd.DataFrame,
                  period: str = "validation") -> pd.DataFrame:
    """Real earthquake offsets kept by cleaning: for every earthquake step in
    NGL's catalogue during `period`, at every scored station, the offset in
    the raw residual (mean of the 10 days after minus the 10 days before,
    horizontal) and the share of it left in the leave-one-out cleaned series,
    ⟨kept, observed⟩ / ⟨observed, observed⟩. Only offsets of at least
    OFFSET_MIN_MM with no other catalogued step within OFFSET_CLEAR_DAYS are
    scored, because the observed offset includes noise and there is no other
    ground truth."""
    code = PERIODS[period]
    pdays = np.flatnonzero(cube.split_day == code)
    lo = max(0, pdays[0] - CROP_MARGIN_DAYS)
    hi = min(cube.r.shape[1], pdays[-1] + CROP_MARGIN_DAYS + 1)
    small = crop_days(cube, lo, hi)
    days = pd.DatetimeIndex(small.days)
    resid = small.r - clean(model, small)
    pos = {s: i for i, s in enumerate(small.sta)}
    in_period = days[small.split_day == code]
    q = steps[(steps.kind == "quake") & steps.sta.isin(pos) & (steps.date >= in_period[0])
              & (steps.date <= in_period[-1])]
    rows = []
    for sta, date, mag, eid in zip(q.sta, q.date, q.mag, q["info"]):
        i = pos[sta]
        if not keep_sta[i]:
            continue
        others = steps[(steps.sta == sta) & (steps.date != date)]
        if (abs(others.date - date) <= pd.Timedelta(days=OFFSET_CLEAR_DAYS)).any():
            continue
        t = int(days.searchsorted(date))
        before = np.arange(max(0, t - OFFSET_WINDOW_DAYS), t)
        after = np.arange(t + 1, min(len(days), t + 1 + OFFSET_WINDOW_DAYS))
        b, a = before[small.avail[i, before]], after[small.avail[i, after]]
        if len(b) < OFFSET_MIN_DAYS or len(a) < OFFSET_MIN_DAYS:
            continue
        obs = small.r[i, a, :2].mean(0) - small.r[i, b, :2].mean(0)
        size = float(np.hypot(*obs))
        if size < OFFSET_MIN_MM:
            continue
        kept = resid[i, a, :2].mean(0) - resid[i, b, :2].mean(0)
        rows.append({"sta": str(sta), "date": str(date.date()), "event": eid, "mag": mag,
                     "offset_mm": round(size, 2), "kept": float(np.dot(kept, obs) / np.dot(obs, obs))})
    return pd.DataFrame(rows)


VELOCITY_WINDOWS = (("2022-01-01", "2023-12-31"), ("2024-01-01", "2025-12-31"))
VELOCITY_PAIR_DAYS = 365       # MIDAS pairs: values one year apart
VELOCITY_MIN_PAIRS = 100       # pairs needed in each window to score a station-component


def midas_velocity(x: np.ndarray, ok: np.ndarray, steps_before: np.ndarray) -> np.ndarray:
    """MIDAS-style velocity (Blewitt et al. 2016), mm/yr, per station and
    component, of `x` (S × T × 3) on the days where `ok` (S × T): the median
    of the slopes of all pairs exactly VELOCITY_PAIR_DAYS apart that do not
    span a step (`steps_before`, S × T: number of catalogued steps on or
    before each day), then the median again after dropping slopes more than
    2 σ (1.4826 × MAD) from it. NaN with fewer than VELOCITY_MIN_PAIRS pairs."""
    lag = VELOCITY_PAIR_DAYS
    pair = ok[:, :-lag] & ok[:, lag:] & (steps_before[:, lag:] == steps_before[:, :-lag])
    slope = (x[:, lag:] - x[:, :-lag]) / (lag / 365.25)
    slope = np.where(pair[..., None], slope, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v0 = np.nanmedian(slope, axis=1, keepdims=True)
        sd = 1.4826 * np.nanmedian(np.abs(slope - v0), axis=1, keepdims=True)
        v = np.nanmedian(np.where(np.abs(slope - v0) <= 2 * sd, slope, np.nan), axis=1)
    v[pair.sum(axis=1) < VELOCITY_MIN_PAIRS] = np.nan
    return v


def velocity_repeatability(model: Reconstructor, cube: Cube, keep_sta: np.ndarray,
                           steps: pd.DataFrame, skip: np.ndarray) -> pd.DataFrame:
    """Contract §5.2: MIDAS velocity of the leave-one-out cleaned series in
    2022–2023 and in 2024–2025, per scored station, and the difference
    (second minus first). Days left out of scoring (`skip`: ±30 days of
    M ≥ 6 earthquakes) are not used. Pairs spanning a catalogued earthquake
    step are skipped, as in MIDAS; equipment steps are not, because the
    residuals have them removed already (contract §1.1), whereas MIDAS works
    on raw positions."""
    days = pd.DatetimeIndex(cube.days)
    lo = max(0, int(days.searchsorted(pd.Timestamp(VELOCITY_WINDOWS[0][0]))) - CROP_MARGIN_DAYS)
    hi = min(len(days), int(days.searchsorted(pd.Timestamp(VELOCITY_WINDOWS[-1][1]), side="right"))
             + CROP_MARGIN_DAYS)
    small = crop_days(cube, lo, hi)
    sdays = pd.DatetimeIndex(small.days)
    resid = small.r - clean(model, small)
    pos = {s: i for i, s in enumerate(small.sta)}
    count = np.zeros(small.avail.shape, dtype=np.int32)
    quakes = steps[steps.kind == "quake"]
    for sta, date in zip(quakes.sta, quakes.date):
        if sta in pos:
            t = int(sdays.searchsorted(date))
            if t < len(sdays):
                count[pos[sta], t] += 1
    steps_before = np.cumsum(count, axis=1)
    ok_all = small.avail & ~skip[:, lo:hi]
    vel = []
    for start, end in VELOCITY_WINDOWS:
        win = (sdays >= pd.Timestamp(start)) & (sdays <= pd.Timestamp(end))
        a, b = np.flatnonzero(win)[[0, -1]]
        vel.append(midas_velocity(resid[:, a:b + 1], ok_all[:, a:b + 1], steps_before[:, a:b + 1]))
    rows = []
    for i in np.flatnonzero(keep_sta):
        d = vel[1][i] - vel[0][i]
        if np.isnan(d).all():
            continue
        rows.append({"sta": str(small.sta[i]),
                     **{f"v1_{c}": vel[0][i, k] for k, c in enumerate("enu")},
                     **{f"v2_{c}": vel[1][i, k] for k, c in enumerate("enu")},
                     **{f"dv_{c}": d[k] for k, c in enumerate("enu")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Seasonal retention (added 2026-10-30)
# --------------------------------------------------------------------------- #

SEASONAL_MIN_DAYS = 365        # available days needed in the period to fit the harmonic


def _harmonic_design(t: np.ndarray, step_t: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * t
    cols = [np.ones_like(t), t - t.mean(), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
    cols += [(t >= s).astype(float) for s in step_t]
    return np.column_stack(cols)


def seasonal_kept(model: Reconstructor, cube: Cube, keep_sta: np.ndarray, steps: pd.DataFrame | None = None,
                  period: str = "validation") -> pd.DataFrame:
    """How much of each station's seasonal cycle a filter keeps (contract §5.6).

    Since 2026-11-05 the annual and semi-annual harmonic stays in r. Over
    `period`, per scored station and component, constant + rate + annual +
    semi-annual (+ Heaviside steps at the station's catalogued earthquakes
    inside the period) is fitted by least squares to r and to the
    leave-one-out residual r - r̂ on the same available days; with c the four
    harmonic coefficients, ρ = ⟨c_kept, c_obs⟩ / ⟨c_obs, c_obs⟩ per component
    (`rho_e`, `rho_n`, `rho_u`), horizontal (`rho_h`) and all three
    (`rho_all`), with the observed annual and semi-annual amplitudes in mm.
    ρ = 1: the filter leaves the station's seasonal cycle alone; ρ = 0: it
    removes it (the part shared with the stations it averages)."""
    from gnssdl.model import _to_dec
    code = PERIODS[period]
    cells = np.flatnonzero(cube.split_day == code)
    lo = max(0, cells[0] - CROP_MARGIN_DAYS)
    hi = min(cube.r.shape[1], cells[-1] + CROP_MARGIN_DAYS + 1)
    small = crop_days(cube, lo, hi)
    days = pd.DatetimeIndex(small.days)
    t_all = np.array([_to_dec(d) for d in days])
    resid = small.r - clean(model, small)
    in_period = (small.split_day == code)[None, :] & small.avail
    quakes = steps[steps.kind == "quake"] if steps is not None and len(steps) else None
    rows = []
    for i in np.flatnonzero(keep_sta):
        ok = in_period[i]
        if ok.sum() < SEASONAL_MIN_DAYS:
            continue
        t = t_all[ok]
        step_t = np.zeros(0)
        if quakes is not None:
            mine = pd.to_datetime(quakes.loc[quakes.sta == str(small.sta[i]), "date"])
            step_t = np.array([_to_dec(d) for d in mine if days[ok][0] < d <= days[ok][-1]])
        G = _harmonic_design(t, step_t)
        c_obs, *_ = np.linalg.lstsq(G, small.r[i, ok].astype(np.float64), rcond=None)
        c_kept, *_ = np.linalg.lstsq(G, resid[i, ok].astype(np.float64), rcond=None)
        h_obs, h_kept = c_obs[2:6], c_kept[2:6]                       # 4 × 3 (E, N, U)
        num = np.sum(h_kept * h_obs, axis=0)
        den = np.sum(h_obs * h_obs, axis=0)
        comp = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        h_num, h_den = num[:2].sum(), den[:2].sum()
        row = {"sta": str(small.sta[i]), "days": int(ok.sum()),
               "rho_e": float(comp[0]), "rho_n": float(comp[1]), "rho_u": float(comp[2]),
               "rho_h": float(h_num / h_den) if h_den > 0 else np.nan,
               "rho_all": float(num.sum() / den.sum()) if den.sum() > 0 else np.nan}
        for k, c in enumerate("enu"):
            p1, q1, p2, q2 = h_obs[:, k]
            row[f"annual_{c}_mm"] = float(np.hypot(p1, q1))
            row[f"semiannual_{c}_mm"] = float(np.hypot(p2, q2))
        rows.append(row)
    return pd.DataFrame(rows)
