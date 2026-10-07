"""Signal-preservation tests (contract §5): injected transients and Ridgecrest.

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

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gnssdl.bench.base import Reconstructor
from gnssdl.bench.score import _split_fast_slow
from gnssdl.dataset import RIDGECREST_LATLON, Cube, crop_days, distance_azimuth

AMPLITUDES_MM = (2.0, 5.0, 10.0)
FOOTPRINTS_KM = (10.0, 25.0, 50.0, 100.0)
DURATIONS_DAYS = (10, 30, 90)
CENTRES_PER_CELL = 50
HOLD_DAYS = 60
SLOT_DAYS = 2 * max(DURATIONS_DAYS) + HOLD_DAYS + 60
SEPARATION_MARGIN_KM = 200.0   # covers the neighbour reach at every radius
FOOTPRINT_CUTOFF = 3.0         # s is set to zero beyond 3 L
SCORE_RADIUS = 2.0             # ρ uses stations within 2 L of the centre
SEED = 20260102

CROP_MARGIN_DAYS = 60          # context kept around cropped windows (M2's running median)

RIDGECREST_MAINSHOCK = pd.Timestamp("2019-07-06")
RIDGECREST_RADIUS_KM = 80.0


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


def plan_injections(
    cube: Cube, keep_sta: np.ndarray, centres_per_cell: int = CENTRES_PER_CELL, seed: int = SEED,
    period: str = "validation",
) -> list[list[Injection]]:
    """All injections of the grid, packed into runs. Centres are placed at
    random scored stations; transient start times fall in slots of `period`.
    Design decisions use the validation years; the test and prospective
    periods are for final reporting."""
    rng = np.random.default_rng(seed)
    test_days = np.flatnonzero(cube.split_day == PERIODS[period])
    n_slots = len(test_days) // SLOT_DAYS
    if n_slots < 1:
        raise ValueError(f"{period} period shorter than one injection slot")
    candidates = np.flatnonzero(keep_sta & cube.avail[:, test_days].any(axis=1))
    d_all = distance_azimuth(cube.lat, cube.lon)[0]

    runs: list[list[Injection]] = []
    next_id = 0
    for L in FOOTPRINTS_KM:
        pending = []
        for A in AMPLITUDES_MM:
            for D in DURATIONS_DAYS:
                for _ in range(centres_per_cell):
                    pending.append((A, D, int(rng.choice(candidates)), rng.uniform(0, 2 * np.pi)))
        rng.shuffle(pending)
        separation = 2 * FOOTPRINT_CUTOFF * L + SEPARATION_MARGIN_KM
        while pending:
            run: list[Injection] = []
            for slot in range(n_slots):
                centres: list[int] = []
                rest = []
                for A, D, c, ang in pending:
                    if all(d_all[c, o] >= separation for o in centres):
                        centres.append(c)
                        t0 = int(test_days[slot * SLOT_DAYS])
                        run.append(Injection(next_id, A, L, D, float(cube.lat[c]), float(cube.lon[c]),
                                             t0, float(np.cos(ang)), float(np.sin(ang))))
                        next_id += 1
                    else:
                        rest.append((A, D, c, ang))
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
            d = distance_azimuth(np.r_[inj.lat, cube.lat[sta]], np.r_[inj.lon, cube.lon[sta]])[0][0, 1:]
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
            rows.append({"id": inj.id, "amplitude_mm": inj.amplitude_mm, "footprint_km": inj.footprint_km,
                         "duration_days": inj.duration_days, "stations": int(len(idx)),
                         "rho": num / den})
        if log:
            log(k + 1, len(runs))
    return pd.DataFrame(rows)


def ridgecrest_retention(model: Reconstructor, cube: Cube, keep_sta: np.ndarray) -> pd.DataFrame:
    """Per station within 80 km of Ridgecrest: the fraction of the observed
    postseismic displacement (days 7-365 after the mainshock, relative to
    days 1-6) that remains in the leave-one-out residual, horizontal."""
    days = pd.DatetimeIndex(cube.days)
    lo = int(days.searchsorted(RIDGECREST_MAINSHOCK - pd.Timedelta(days=CROP_MARGIN_DAYS)))
    hi = int(days.searchsorted(RIDGECREST_MAINSHOCK + pd.Timedelta(days=365 + CROP_MARGIN_DAYS)))
    cube = crop_days(cube, lo, hi)
    days = pd.DatetimeIndex(cube.days)
    ref = (days >= RIDGECREST_MAINSHOCK + pd.Timedelta(days=1)) & (days <= RIDGECREST_MAINSHOCK + pd.Timedelta(days=6))
    post = (days >= RIDGECREST_MAINSHOCK + pd.Timedelta(days=7)) & (days <= RIDGECREST_MAINSHOCK + pd.Timedelta(days=365))
    d = distance_azimuth(np.r_[RIDGECREST_LATLON[0], cube.lat], np.r_[RIDGECREST_LATLON[1], cube.lon])[0][0, 1:]
    resid = cube.r - clean(model, cube)
    rows = []
    for i in np.flatnonzero((d <= RIDGECREST_RADIUS_KM) & keep_sta):
        a_ref, a_post = cube.avail[i] & ref, cube.avail[i] & post
        if a_ref.sum() < 3 or a_post.sum() < 60:
            continue
        obs = cube.r[i, a_post, :2] - cube.r[i, a_ref, :2].mean(axis=0)
        kept = resid[i, a_post, :2] - resid[i, a_ref, :2].mean(axis=0)
        den = float(np.sum(obs * obs))
        if den <= 0:
            continue
        rows.append({"sta": str(cube.sta[i]), "dist_km": round(float(d[i]), 1),
                     "post_mm": round(float(np.sqrt(np.mean(np.sum(obs**2, axis=1)))), 2),
                     "retained": float(np.sum(kept * obs)) / den})
    return pd.DataFrame(rows)


def noise_removed(model: Reconstructor, cube: Cube, keep_sta: np.ndarray,
                  period: str = "validation") -> dict[str, float]:
    """How much of the fast (< ~2 months) scatter cleaning removes, over
    `period`: per scored station, 1 - rms(fast(r - r̂)) / rms(fast(r)),
    averaged over E/N/U, then the median over stations. Computed the same
    way for every model, including reference filters that cannot be scored
    on the masks, so it is the x-axis of the noise-vs-signal trade-off."""
    code = PERIODS[period]
    test = np.flatnonzero(cube.split_day == code)
    lo = max(0, test[0] - CROP_MARGIN_DAYS)
    hi = min(cube.r.shape[1], test[-1] + CROP_MARGIN_DAYS + 1)
    small = crop_days(cube, lo, hi)
    resid = small.r - clean(model, small)
    days = pd.DatetimeIndex(small.days)
    in_test = small.split_day == code
    out = []
    for i in np.flatnonzero(keep_sta):
        cols = np.flatnonzero(small.avail[i] & in_test & ~small.exclude[i])
        if len(cols) < 180:
            continue
        f_r, _ = _split_fast_slow(small.r[i, cols].astype(np.float64), days[cols])
        f_e, _ = _split_fast_slow(resid[i, cols].astype(np.float64), days[cols])
        ratio = np.sqrt(np.nanmean(f_e**2, axis=0)) / np.sqrt(np.nanmean(f_r**2, axis=0))
        out.append(1.0 - float(np.mean(ratio)))
    return {"noise_removed": float(np.median(out)) if out else np.nan, "stations": len(out)}
