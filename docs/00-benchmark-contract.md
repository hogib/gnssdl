# 00 — Benchmark contract

Everything every model shares. A model that needs to deviate from this
document is not comparable and must say so in its own doc.

## 0. Signal and nuisance

The benchmark's target is fixed here so that "good filtering" has one
meaning.

- Signal (to be kept): tectonic and other aseismic transients — slow slip,
  creep events, afterslip and other postseismic deformation, volcanic and
  magmatic deformation, and coseismic offsets.
- Nuisance (to be removed): common-mode error from processing (orbits,
  clocks, reference frame) and environmental loading (atmospheric,
  hydrological, non-tidal ocean), plus station-specific noise.

Loading is a real geophysical signal and the object of other studies; here
it is nuisance because it is not the target. A filter that removes shared
loading is doing its job. Daily loading products (Li et al. 2025, e.g. GFZ
or EOST) can be used to measure how much of the removed common mode is
loading.

## 1. The data cube

Built once by `gnssdl build` (module `gnssdl.dataset`), stored as
`data/cube/california.npz`.

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `r` | S × T × 3 | float32 | Residual displacement in mm (E, N, U). NaN where missing. |
| `sigma` | S × T × 3 | float32 | NGL formal 1σ in mm. NaN where missing. |
| `avail` | S × T | bool | Observation exists and passed outlier screening. |
| `sta` | S | str | 4-char station ids, sorted. |
| `lat`, `lon` | S | float64 | Degrees, lon in −180..180. |
| `days` | T | datetime64[D] | Daily grid, 2008-01-01 … last NGL epoch. |
| `radii` | NR | float32 | Exclusion radii R in km: 0, 10, 25, 50, 100, 200, 400 (§1.2). |
| `nbr_idx` | NR × S × K | int32 | For each R, indices of the K = 16 nearest stations with R < d ≤ R + 300 km, nearest first; −1 pads. |
| `nbr_dist` | NR × S × K | float32 | Great-circle distance, km. |
| `nbr_az` | NR × S × K | float32 | Azimuth from station to neighbour, radians. |
| `split_day` | T | int8 | 0 train, 1 validation, 2 test. |
| `split_sta` | S | int8 | 0 normal, 1 held-out station. |
| `exclude` | S × T | bool | Cells barred from training (Ridgecrest window). |

S ≈ 1,381, T ≈ 6,900; the cube is ~80 MB per float array.

### 1.1 How `r` is made

Per station and component, `gnssdl.model.departure` fits
constant + rate + annual + semi-annual + Heaviside steps by least squares.

- The fit uses training days only (`split_day == 0`, not `exclude`). A
  station with fewer than 365 training epochs is fitted on its first 730
  available days instead and is forced into `split_sta = 1`, so its baseline
  never touches data a model is scored on as "seen".
- Equipment steps (NGL `steps.txt` type 1) are estimated and removed
  everywhere, including after the training period (local ±30-day jump
  estimate, as in `departure`).
- Earthquake steps (type 2) are not removed. They are spatially coherent and
  the models should learn to predict them from neighbours.
- Screening: a value more than 5 robust σ (1.4826 × MAD of a 61-day running
  window) from the running median is set to missing.
- Fallback fit (stations with < 365 training epochs): the first 730 days or
  the first 365 epochs of the record, whichever reaches further.
- Station QC: a second fit removes earthquake steps too, so real coseismic
  offsets don't count. A station is dropped if its robust scale on the fit
  epochs exceeds 10 mm horizontal or 30 mm vertical. This removes
  non-tectonic site motion: groundwater subsidence (Central Valley), unstable
  monuments, landslides, volcanic sources. The dropped list, with scales and
  coordinates, is written to the cube's `.json` summary for review.
- Twins: stations closer than 1 km are one site; only the one with the most
  station-days is kept, so no model can copy a co-located antenna and no
  "unseen" station has a twin in training.

Known limitation (accepted): two preprocessing steps look at a station's
surrounding days in every period, including days that later become hidden
evaluation cells. These are the local jump estimate for equipment changes
after the training period (±30 days) and outlier screening (61-day running
median). Neither uses another station or any label, and the effect on a
visible cell is at most a small constant shift (an equipment jump estimate)
or the removal of a spike. The cube is built once, before masks exist, so
this is not reproduced per mask. Results should mention it.

### 1.2 Exclusion radius

Neighbours closer than R to the target are never used to predict it. R is an
experimental variable, not a hyper-parameter: every model is run and scored
at each R in `radii` = {0, 10, 25, 50, 100, 200, 400} km. R = 400 km
reproduces the fixed distance of Bachelot et al. (2025).

- Why: shared network noise is coherent over hundreds of km, while a
  transient of footprint L reaches stations only out to roughly L. Excluding
  near neighbours should keep more transient signal at the cost of removing
  less noise. The trade-off as a function of R is one of the main results.
- Large R also mimics a sparse network: California at R = 50–100 km
  resembles Türkiye's NGL network (median nearest neighbour ~62 km). This
  makes the sweep the first step of any transfer study.
- Neighbours are searched up to 300 km beyond R (700 km at R = 400); every
  station keeps 16 neighbours at every radius in the current cube. Every
  score table still reports, per R, the median neighbour count and the
  fraction of predictions that fell back (see each model doc), since a
  sparser network may run short.

### 1.3 Temporal context

Which days a prediction for target i at day t may use:

| Mode | Neighbours' data | Target's own unhidden data |
|---|---|---|
| `same-day` | day t only | none |
| `causal` | days ≤ t | days < t (only if own-history is on) |
| `two-sided` | whole window | whole window (only if own-history is on) |

Own-history is a separate on/off switch. With it on, a model can carry a
transient's onset from the target's visible days into the hidden ones, which
moves signal from the residual into the prediction (absorption through time
rather than space). It is therefore off for every signal-preservation test
(§5) and for producing cleaned series (§2.1). It may be on only for the
gap-filling scores (`scatter`, `block`), and results always state which
setting was used.

M1–M3 are `same-day` by construction. M4 is `two-sided` (non-causal
filters). M5 is run in all three modes, with and without own-history. The
`causal` mode is the one the later continual detector needs.

### 1.4 Splits

| Name | Days |
|---|---|
| train | 2008-01-01 – 2019-12-31 |
| validation | 2020-01-01 – 2021-12-31 |
| test | 2022-01-01 – freeze date, or end |
| prospective | after the freeze date (`gnssdl build --freeze-date`) |

Decisions use the validation years only: hyper-parameters, design choices,
model selection and any change to the scoring rules. The test years were
looked at while the scoring rules were being revised (October 2026), so they
are no longer untouched; they remain for reporting. The final, untouched
evaluation is prospective: freeze the method on a date, rebuild the cube
later with `--freeze-date` so that days recorded afterwards become split 3,
and score them once. `gnssdl bench report` defaults to validation and
prints a reminder when another split is shown.

- Held-out stations: 10% of stations, chosen by farthest-point sampling over
  station coordinates with seed 0, so they are spread out rather than
  clustered.
- Ridgecrest exclusion: stations within 150 km of (35.77 N, 117.60 W), days
  2019-07-01 – 2020-06-30, have `exclude = True`. No model trains on them.
  They are scored separately (section 5.3).

### 1.5 Training data for learned models

Scoring uses the real residuals `r`, earthquake offsets included. Learned
models (M3 onward) train on something quieter (`gnssdl.bench.train`):

- quiet residuals: `r` minus the earthquake offsets estimated in the
  training-period fit (stored in the cube as `qstep_sta`, `qstep_day`,
  `qstep_amp`);
- training cells: training days, available, outside the Ridgecrest
  exclusion and outside ±30 days of listed M ≥ 6 earthquakes, on stations
  that are not held out;
- per-station normalisation by the robust noise scale.

Why: with offsets of tens to hundreds of centimetres in the training years
(El Mayor-Cucapah, 2010), a squared-error loss would be dominated by a few
stations, and weights that reproduce shared offsets are weights that absorb
signal. Training on quiet data teaches the models the noise, not the signal.

## 2. Model interface

`gnssdl.bench.base`:

```python
class Reconstructor(Protocol):
    name: str

    def fit(self, cube: Cube) -> None:
        """Learn from train days (and validation days for early stopping /
        hyper-parameter choice only). Must ignore held-out stations,
        test days and excluded cells."""

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        """hide: S×T bool (or S×T×3) cells the model must not see.
        Return S×T×3 float32 predictions, finite wherever `hide` is True
        and the truth exists. Values elsewhere are ignored."""

    # Run settings, set by the harness before fit/predict:
    radius_km: float      # exclusion radius R (§1.2); selects nbr_idx[R]
    context: str          # "same-day" | "causal" | "two-sided" (§1.3)
    own_history: bool     # §1.3
```

A model that doesn't support a setting (e.g. M1 with `causal`) raises
instead of silently ignoring it.

`predict` receives the full cube; the harness, not the model, is
responsible for masking: before calling `predict` it sets `r`, `sigma` and
`avail` to NaN/False on hidden cells in a copy. A model that reads a hidden
value is a bug the tests must catch (section 6).

### 2.1 Producing cleaned series (leave-one-out)

The cleaned series of station i is r_i − r̂_i, where r̂_i is predicted with
station i hidden for the whole period, neighbours within R excluded, and
own-history off. A prediction made with the target visible is meaningless as
a common-mode estimate: a flexible model can copy the input, leaving a zero
residual. Every transient or Ridgecrest result is computed this way, and
`gnssdl.bench.signal.clean` is the only code path that produces cleaned
series. For models that never read the target (`never_reads_target`,
verified by the radius leak check), one prediction on the unhidden cube is a
leave-one-out cleaning of every station at once.

Exception: reference filters. A labelled reference filter
(`reference_filter`, e.g. M2-self) reads the target on purpose, to measure
what a published filter of that kind removes. It goes through the same
`clean` function, is refused by `gnssdl bench run`, and is reported only on
signal kept (§5.1, §5.3) and noise removed (§5.4), always under its own
name.

Every trained model saves `data/models/<name>/` with its parameters and a
`config.json` holding every hyper-parameter and the git commit.

## 3. Masks

### Training masks

Generated on the fly by each trainable model, never reused. The three
patterns below are the minimum; a model doc may add more.

### Evaluation masks

Generated once by `gnssdl bench masks` with seed 20260101, saved in
`data/cube/masks.npz`, shared by all models.

| Pattern | Definition | Applies to |
|---|---|---|
| `scatter` | 10% of available station-days, uniformly | seen stations, val and test days |
| `block` | One 30-day block per station per year, start uniform | seen stations, val and test days |
| `station` | All days of every held-out station | held-out stations, val and test days |
| `ridgecrest` | All excluded cells | Ridgecrest window |

All three components of a station-day are hidden together.

## 4. Scoring

All scores use hidden cells where the truth exists, per pattern, per
component, per split (`gnssdl.bench.score`).

Per station first, then the median. Each station with at least 20 hidden
cells gets its own scores; the headline is the median over stations, with
the interquartile range. Pooling all cells at once lets a handful of
stations with large earthquake offsets dominate (it did in the first run),
so pooled numbers are kept only as secondary `pooled_*` columns.

Fast and slow parts. For each station, the error on its hidden cells, in
time order, is split into

- slow: 61-day running median of the error (trend drift since the training
  period, earthquake offsets, slow transients);
- fast: error minus slow (day-to-day and week-to-week scatter).

`nrmse_fast` is the primary score (RMS of fast / s_i, with s_i the
station's robust training-period scale); it is also the tuning criterion,
computed leave-one-out over all validation days (see Tuning criterion below).
`nrmse_slow` and `nrmse_total` are reported. Signal a model should leave
alone lands mostly in the slow part and is judged by the §5 tests, not
rewarded or penalised here. M0's fast score is below 1 (about 0.7) because
the split also removes some of each station's own noise; compare models
against M0, not against 1.

Left out of scoring:

- cells within ±30 days of an M ≥ 6 earthquake listed in NGL's steps file
  for that station (fast and slow parts only);
- stations with site trouble after the training period (scoring QC): a more
  than 3× increase in day-to-day scatter relative to the training years, or
  a jump of more than 50 mm horizontal / 100 mm vertical between
  consecutive 30-day medians with no listed step within 45 days. The QC uses
  only each station's own data and the steps file, so it is the same for
  every model; the list is stored with each result.

`cmr` (common-mode reduction, 1 − var(r − r̂) / var(r)) is reported in the
pooled columns for comparison with published numbers, not ranked.

Tuning criterion. Models that never read the target are tuned on the
leave-one-out fast score over every available validation day of the seen,
scored stations (`tuning_cells` in `gnssdl.bench.run`); one prediction on
the unhidden cube is leave-one-out for every station, so nothing needs
hiding. This replaced tuning on the validation `scatter` cells, whose
sparse sampling (about 1 day in 10) made the fast/slow split unreliable.
Models that cannot produce leave-one-out predictions in one pass fall back
to the `scatter` cells.

Uncertainty (`gnssdl bench compare A B`, module `gnssdl.bench.stats`):

- noise: each station's leave-one-out fast error is summarised per 30-day
  block; 1,000 resamples of blocks (the same blocks for every station) and
  of stations give a 95% interval for the difference in median score;
- signal: injections are resampled, paired by id, for the difference in
  median ρ per footprint.

A difference whose interval includes 0 is reported as not
distinguishable. Trainable models are run with 5 seeds; report mean and
spread.

## 5. Signal-preservation tests

These decide the ranking together with `nrmse`.

### 5.1 Injected transients

`gnssdl bench signal MODEL` (module `gnssdl.bench.signal`) adds synthetic
signals s to `r` in one period: validation by default, for decisions
(`--period test` or `prospective` for reporting). The validation years hold
only two 300-day slots, so design runs may use fewer injections per cell
(`--centres 20`). Phase 1 uses Gaussian footprints:

  s_i(t) = A · exp(−d_i² / 2L²) · g(t) · û

where d_i is the distance to a centre placed at a random scored station, û a
random horizontal unit vector, and g a raised-cosine rise over D days, a
60-day hold at 1 and a raised-cosine fall over D days. The fall keeps each
transient inside its own time slot, so no permanent offset can contaminate
later injections at the same stations. s is zero beyond 3L.

Grid: A ∈ {2, 5, 10} mm, L ∈ {10, 25, 50, 100} km, D ∈ {10, 30, 90} days,
50 centres per cell (1,800 injections). These are deliberately idealised:
every station inside the footprint moves the same way. Realistic,
fault-consistent signals are the event library (§5.5).

Many transients share one model pass. The period is cut into 300-day slots.
A transient affects a station's cleaned series if the station is inside its
footprint (d ≤ 3L) or has a neighbour there, at the filter's exclusion
radius; filters that use every station are affected everywhere. Within a
slot, two transients share a run only if no station is affected by both, so
packing is exact and adapts to the radius (larger radii pack fewer per run).

ρ is paired with an uninjected run of the same model, leave-one-out as in
§2.1:

  Δ = (r + s − P(r + s)) − (r − P(r)),    ρ = ⟨Δ, s⟩ / ⟨s, s⟩

over the horizontal components, the transient's days and the scored stations
within 2L of the centre. Pairing removes the noise term ⟨r − P(r), s⟩; for a
linear model ρ = ⟨s − P(s), s⟩ / ⟨s, s⟩ exactly. ρ ≈ 1 means the transient
survives; ρ ≈ 0 means the model absorbed it. Report ρ over (L, R) for each D
and amplitude. The expected pattern is that ρ rises towards 1 as R exceeds L.

Spurious signal: at scored stations outside the footprint that the
transient still affects through their neighbours, nothing was planted, so
any change is created by the filter. Each injection records the largest and
the median peak horizontal change at those stations, as a fraction of the
planted amplitude A (`spurious_max`, `spurious_median`). Retention is also
reported per amplitude, which tests whether a filter treats small and large
signals differently.

### 5.2 Velocity repeatability

MIDAS-style velocity of (r − r̂) on 2022–2023 vs 2024–2025, per station.
Smaller differences mean less long-period distortion.

### 5.3 Ridgecrest

Fraction of the observed postseismic displacement (days 7–365 after
2019-07-06, relative to the mean of days 1–6) retained in the leave-one-out
residual r − r̂ at scored stations within 80 km, horizontal components:
⟨kept, observed⟩ / ⟨observed, observed⟩ per station, median over stations.
These days are excluded from training, so no model has seen them.

### 5.4 Noise removed

The x-axis of the noise-vs-signal trade-off, computed identically for every
model including reference filters: on the test years, per scored station,
1 − rms(fast(r − r̂)) / rms(fast(r)) averaged over E/N/U, with r̂ the
`clean` prediction; the median over stations. M0 scores 0.

### 5.5 Event library (realistic injections)

The Gaussian blobs of §5.1 are a controlled sweep of width against exclusion
radius, but real transients are not round: slip on a strike-slip fault moves
the two sides in opposite directions, and a neighbour average partly cancels
such a signal instead of reproducing it. Blob results are therefore stated
with their scope ("spatially uniform transients of width L"), and
conclusions about real transients, and any judgement of M5, wait for this
test.

Principle: copy the source, not the recording. An observed displacement
field exists only at the stations that recorded it, mixed with noise and
other signals. A template is instead a source model (fault patch, slip,
depth, time evolution) taken from a published study of a real California
event. It is moved to other faults of the same type, and the displacement
each real station would record is computed from it.

Templates (values from abstracts; verify against the full papers before
use):

| Template | Source | Time function | Reference |
|---|---|---|---|
| Shallow slow slip | cm-scale strike-slip from the surface to ~2 km depth | slip front propagating along strike at ~9 km/day, bilaterally, over 2-3 weeks | 2023 Superstition Hills / Imperial faults (Materna et al. 2024, GRL) |
| Afterslip | slip next to the ends of a rupture | logarithmic decay | 2019 Ridgecrest (published afterslip inversions) |
| Triggered creep | shallow creep on a neighbouring fault | lasting > 178 days | Garlock fault after Ridgecrest |
| Large afterslip | near-field afterslip | log decay τ ≈ 20 d (exp. τ ≈ 66 d) | 2010 El Mayor-Cucapah (Gonzalez-Ortega et al. 2014) |

Templates live in `data/events/*.yaml` (geometry relative to the fault,
slip, depth range, time function, reference), so each one is reviewable.

Transplanting:

- Fault geometry: SCEC Community Fault Model 7.0 triangulated surfaces
  (statewide; Zenodo record 13685611), stored under `data/cfm/` (not
  tracked).
- Target faults: same style as the template (strike-slip for all four), dip
  within ±15° of the template's, long enough to host the patch. The patch is
  placed at a random along-strike position within the template's depth
  range; slip and time function are kept.
- Displacements: east, north and up at every station's real position from
  triangular dislocations in an elastic half-space (`cutde`, after Nikkhoo
  & Walter 2015; Poisson's ratio 0.25).
- Timing and packing: random start inside a test-period slot, as in §5.1.
  Two events share a slot only if no station sees more than 0.1 mm from
  both.

Scoring: the paired ρ of §5.1, over the scored stations where the event's
peak horizontal displacement exceeds 1 mm (there is no single width L).
Also report ρ against each station's distance from the fault trace, and ρ
separately for stations on the two sides of the fault.

Limitations, stated with every result: elastic half-space only (no layered
crust, no viscoelastic relaxation, so the far field after large earthquakes
is underrepresented); published slip models are non-unique and are used in
simplified form.

Implementation: `gnssdl.bench.events`, run as `gnssdl bench signal MODEL
--events`. New dependency: `cutde`.

## 6. Required tests (for every model)

1. Leak test: perturb hidden cells in the input copy by +10⁶ mm; predictions
   must not change.
2. Shape and finiteness: output S×T×3, finite on every hidden cell with
   truth.
3. Split discipline: `fit` on a cube whose validation/test/held-out cells
   are NaN gives identical parameters to `fit` on the full cube.
4. Toy recovery: on a synthetic cube where every station equals a shared
   common mode plus small white noise, the model must reach `nrmse` well
   below M0.
5. Radius discipline: perturb every station within R of the target by
   +10⁶ mm; the target's prediction must not change.
6. Context discipline: in `causal` mode, perturb all cells after day t;
   predictions at t must not change. With own-history off, perturb the
   target's unhidden cells; its predictions must not change.

## 7. Layout

```
src/gnssdl/
  dataset.py        cube construction (gnssdl build)
  bench/
    base.py         Cube, Reconstructor, mask application
    masks.py        evaluation masks
    inject.py       synthetic transients
    score.py        metrics + bootstrap
    m0_zero.py  m1_stack.py  m2_robust.py  m3_ridge.py  m4_band.py
    nn/             M5
```
