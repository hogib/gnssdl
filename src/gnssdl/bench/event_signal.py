"""Event library, part 3: realistic transients through the signal tests
(contract §5.5).

Templates (data/events/*.toml) are placed on random suitable CFM faults
(`gnssdl.bench.events`), added to the real residuals, and scored with the
paired ρ of §5.1:

    Δ = (r + s − P(r + s)) − (r − P(r)),   ρ = ⟨Δ, s⟩ / ⟨s, s⟩

over the horizontal components, the event's days (onset to end plus a
60-day hold) and the scored stations whose peak horizontal displacement
exceeds 1 mm. Each station's own ρ is also kept, with its distance from the
slip patch and its side of the fault, so absorption can be read against
both.

Differences from the Gaussian blobs:
- slip is permanent, so the planted offset stays to the end of the cube,
  and two events share a run only if no station is affected by both at
  any time (a later event must not sit on an earlier one's offset);
- an event is kept only if at least MIN_SEEN scored stations move by more
  than SEEN_MM: signal loss is measurable only where the network sees the
  event;
- spurious signal is a fraction of the event's largest station
  displacement, at stations moving less than INSIDE_MM.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnssdl.bench import events as ev_mod
from gnssdl.bench.base import Reconstructor
from gnssdl.bench.signal import CROP_MARGIN_DAYS, HOLD_DAYS, PERIODS, affected, clean
from gnssdl.dataset import Cube, crop_days, distance_azimuth

EVENT_DIR = Path("data/events")
SEEN_MM = 1.0                  # a station "sees" the event above this peak horizontal motion
MIN_SEEN = 3                   # scored stations that must see it
INSIDE_MM = 0.1                # stations moving more than this are inside the event (packing)
EVENTS_PER_TEMPLATE = 20
SEED = 20260103


@dataclass
class EventInjection:
    id: int
    event: ev_mod.PlacedEvent
    t0: int                    # day index of the onset
    window: int                # days scored from t0 (event duration + hold)
    inside: np.ndarray         # S bool, stations moving more than INSIDE_MM
    seen: np.ndarray           # S bool, scored stations moving more than SEEN_MM


def load_templates(directory: Path = EVENT_DIR) -> list[ev_mod.EventTemplate]:
    """Every *.toml template in `directory`. Keys are EventTemplate's fields;
    `slip_senses` is a list of CFM codes."""
    names = {f.name for f in fields(ev_mod.EventTemplate)}
    out = []
    for path in sorted(Path(directory).glob("*.toml")):
        raw = tomllib.loads(path.read_text())
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        raw["slip_senses"] = tuple(raw["slip_senses"])
        out.append(ev_mod.EventTemplate(**raw))
    return out


def _peak_horizontal(event: ev_mod.PlacedEvent) -> np.ndarray:
    """S peak horizontal displacement in mm (slip only grows, so the final
    value is the peak)."""
    u = event.final_mm()
    return np.hypot(u[:, 0], u[:, 1])


def plan_events(
    cube: Cube, keep_sta: np.ndarray, templates: list[ev_mod.EventTemplate],
    catalogue: list[ev_mod.FaultInfo], per_template: int = EVENTS_PER_TEMPLATE, seed: int = SEED,
    period: str = "validation", radius_km: float | None = 0.0, model=None, max_tries: int = 400,
) -> list[list[EventInjection]]:
    """Place `per_template` events of each template and pack them into runs
    in which no station is affected by two events."""
    rng = np.random.default_rng(seed)
    period_days = np.flatnonzero(cube.split_day == PERIODS[period])
    has_data = cube.avail[:, period_days].any(axis=1)
    scored = keep_sta & has_data
    pending: list[tuple[EventInjection, np.ndarray]] = []
    next_id = 0
    for tp in templates:
        placed = tries = 0
        while placed < per_template:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"{tp.name}: only {placed} of {per_template} placements seen by "
                                   f"{MIN_SEEN} stations after {max_tries} tries")
            event = ev_mod.random_event(tp, catalogue, rng, cube.lon, cube.lat)
            peak = _peak_horizontal(event)
            seen = scored & (peak > SEEN_MM)
            if seen.sum() < MIN_SEEN:
                continue
            window = int(np.ceil(event.duration_days)) + HOLD_DAYS
            if window >= len(period_days):
                raise ValueError(f"{tp.name}: event lasts longer than the {period} period")
            t0 = int(period_days[rng.integers(0, len(period_days) - window)])
            u = event.final_mm()
            inside = np.linalg.norm(u, axis=1) > INSIDE_MM
            inj = EventInjection(-1, event, t0, window, inside, seen)
            pending.append((inj, affected(cube, inside, radius_km, model)))
            placed += 1
    rng.shuffle(pending)
    runs: list[list[EventInjection]] = []
    while pending:
        taken = np.zeros(len(cube.sta), dtype=bool)
        run, rest = [], []
        for inj, aff in pending:
            if not (aff & taken).any():
                taken |= aff
                run.append(replace(inj, id=next_id))
                next_id += 1
            else:
                rest.append((inj, aff))
        runs.append(run)
        pending = rest
    return runs


def event_signal(cube: Cube, inj: EventInjection) -> tuple[np.ndarray, np.ndarray, slice]:
    """(station indices, s[stations, days, 3] in mm, day slice) for one
    event: the displacement from the onset, held at its final value to the
    end of the cube."""
    sta = np.flatnonzero(inj.inside)
    days = slice(inj.t0, cube.r.shape[1])
    n = days.stop - days.start
    t = np.arange(n, dtype=float)
    sub = replace(inj.event, green=inj.event.green[sta])
    m = min(n, inj.window + 1)
    s = np.empty((len(sta), n, 3), dtype=np.float32)
    s[:, :m] = sub.displacement_mm(t[:m])
    s[:, m:] = sub.final_mm()[:, None, :]                       # permanent once the event is over
    return sta, s, days


def _side_and_distance(cube: Cube, inj: EventInjection, sta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For stations `sta`: distance (km) to the nearest triangle centroid of
    the slip patch (surface projection), and the side of the fault (+1 / −1)
    relative to the patch's mean strike through its centre."""
    ev = inj.event
    lon0, lat0 = ev.centre
    frame = ev_mod.LocalFrame(lon0, lat0)
    fault = ev.meta["fault"]
    tris = ev_mod.fault_triangles(fault, frame, ev.tri)
    cxy = tris.mean(axis=1)[:, :2]
    _, strike, _, _ = ev_mod.triangle_frames(tris)
    s = strike.mean(axis=0)[:2]
    s /= np.linalg.norm(s) or 1.0
    x, y = frame.xy(cube.lon[sta], cube.lat[sta])
    p = np.stack([x, y], axis=1)
    dist = np.sqrt(((p[:, None, :] - cxy[None]) ** 2).sum(-1)).min(axis=1) / 1000.0
    side = np.sign(s[0] * p[:, 1] - s[1] * p[:, 0])                # left of strike = +1
    return dist, side


def event_scores(
    model: Reconstructor, cube: Cube, keep_sta: np.ndarray, runs: list[list[EventInjection]], log=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per event, one row per event × seeing station)."""
    lo = max(0, min(i.t0 for run in runs for i in run) - CROP_MARGIN_DAYS)
    hi = cube.r.shape[1]
    cube = crop_days(cube, lo, hi)
    runs = [[replace(i, t0=i.t0 - lo) for i in run] for run in runs]
    base = cube.r - clean(model, cube)
    rows, sta_rows = [], []
    for k, run in enumerate(runs):
        r = cube.r.copy()
        parts = []
        for inj in run:
            sta, s, days = event_signal(cube, inj)
            ok = cube.avail[sta, days]
            r[sta, days] += np.where(ok[..., None], s, 0.0)
            parts.append((inj, sta, s, days))
        resid = r - clean(model, replace(cube, r=r))
        for inj, sta, s, days in parts:
            win = slice(days.start, min(days.start + inj.window, cube.r.shape[1]))
            w = win.stop - win.start
            use = inj.seen[sta] & keep_sta[sta]
            idx = sta[use]
            ok = cube.avail[idx, win]
            delta = (resid[idx, win] - base[idx, win])[..., :2]
            sig = s[use][:, :w, :2]
            num_i = np.sum(np.where(ok[..., None], delta * sig, 0.0), axis=(1, 2))
            den_i = np.sum(np.where(ok[..., None], sig * sig, 0.0), axis=(1, 2))
            if den_i.sum() <= 0:
                continue
            peak = _peak_horizontal(inj.event)
            row = {"id": inj.id, "template": inj.event.template.name, "fault": inj.event.fault,
                   "rake": inj.event.rake, "t0": str(cube.days[days.start]),
                   "stations": int(len(idx)), "peak_mm": float(peak[idx].max()),
                   "rho": float(num_i.sum() / den_i.sum())}
            row.update(_event_spurious(cube, base, resid, inj, win, keep_sta, model, float(peak[idx].max())))
            rows.append(row)
            dist, side = _side_and_distance(cube, inj, idx)
            for j, i in enumerate(idx):
                if den_i[j] > 0:
                    sta_rows.append({"id": inj.id, "template": row["template"], "sta": str(cube.sta[i]),
                                     "peak_mm": float(peak[i]), "dist_km": float(dist[j]),
                                     "side": int(side[j]), "rho": float(num_i[j] / den_i[j])})
        if log:
            log(k + 1, len(runs))
    return pd.DataFrame(rows), pd.DataFrame(sta_rows)


def _event_spurious(cube: Cube, base: np.ndarray, resid: np.ndarray, inj: EventInjection, win: slice,
                    keep_sta: np.ndarray, model, scale_mm: float) -> dict:
    """Peak horizontal change in the cleaned series at scored stations the
    event can affect through their neighbours but that move by less than
    INSIDE_MM themselves, as a fraction of the event's largest scored
    displacement."""
    outside = affected(cube, inj.inside, model.radius_km, model) & ~inj.inside & keep_sta
    idx = np.flatnonzero(outside)
    if not len(idx):
        return {"spurious_stations": 0, "spurious_max": 0.0, "spurious_median": 0.0}
    delta = (resid[idx, win] - base[idx, win])[..., :2]
    ok = cube.avail[idx, win]
    mag = np.where(ok, np.hypot(delta[..., 0], delta[..., 1]), 0.0)
    peak = mag.max(axis=1) / scale_mm
    return {"spurious_stations": int(len(idx)), "spurious_max": float(peak.max()),
            "spurious_median": float(np.median(peak))}
