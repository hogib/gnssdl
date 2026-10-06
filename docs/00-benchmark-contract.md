# 00 — Benchmark contract

Everything every model shares. A model that needs to deviate from this
document is not comparable and must say so in its own doc.

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
| `radii` | NR | float32 | Exclusion radii R in km: 0, 10, 25, 50, 100 (§1.2). |
| `nbr_idx` | NR × S × K | int32 | For each R, indices of the K = 16 nearest stations with R < d ≤ 300 km, nearest first; −1 pads. |
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
at each R in `radii` = {0, 10, 25, 50, 100} km.

- Why: shared network noise is coherent over hundreds of km, while a
  transient of footprint L reaches stations only out to roughly L. Excluding
  near neighbours should keep more transient signal at the cost of removing
  less noise. The trade-off as a function of R is one of the main results.
- Large R also mimics a sparse network: California at R = 50–100 km
  resembles Türkiye's NGL network (median nearest neighbour ~62 km). This
  makes the sweep the first step of any transfer study.
- At large R some stations run out of neighbours within 300 km. Every score
  table reports, per R, the median neighbour count and the fraction of
  predictions that fell back (see each model doc).

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
| test | 2022-01-01 – end |

- Held-out stations: 10% of stations, chosen by farthest-point sampling over
  station coordinates with seed 0, so they are spread out rather than
  clustered.
- Ridgecrest exclusion: stations within 150 km of (35.77 N, 117.60 W), days
  2019-07-01 – 2020-06-30, have `exclude = True`. No model trains on them.
  They are scored separately (section 5.3).

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
`gnssdl bench clean` is the only code path that produces cleaned series.

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

All metrics are computed on hidden cells where the truth exists, per
pattern, per component, per split.

- `nrmse`: RMSE of (r − r̂) / s_i, where s_i is station i's robust scale
  (1.4826 × MAD of its training residuals). Lower is better. M0 scores ≈ 1
  by construction.
- `wmse`: mean of ((r − r̂)/σ)². Reported, not ranked.
- `cmr` (common-mode reduction): 1 − var(r − r̂) / var(r), over hidden cells,
  to compare with published numbers. Not ranked, because it rewards removing
  signal.

Every score is reported per exclusion radius R. The headline trade-off
plot shows `cmr` (noise removed) against ρ (signal kept, §5.1) as R varies,
one curve per model.

Uncertainty: paired bootstrap over days (blocks of 30 days, 1,000
resamples) for every model-vs-model difference. Trainable models are run
with 5 seeds; report mean and spread.

## 5. Signal-preservation tests

These decide the ranking together with `nrmse`.

### 5.1 Injected transients

`gnssdl bench inject` adds synthetic signals s to the test-period `r`
before masking. Phase 1 uses Gaussian footprints:

  s_i(t) = A · exp(−d_i² / 2L²) · g(t) · û

where d_i is distance to a random centre, g a smooth ramp (raised-cosine) of
duration D, and û a random horizontal unit vector.

Grid: A ∈ {2, 5, 10} mm, L ∈ {10, 25, 50, 100} km, D ∈ {10, 30, 90} days,
50 random centres per cell. Phase 2 swaps the footprint for Okada
dislocations on strike-slip faults.

For each injection, every station within 2L of the centre is cleaned
leave-one-out as in §2.1 (whole station hidden, neighbours within R
excluded, own-history off), and

  ρ = ⟨r − r̂, s⟩ / ⟨s, s⟩

measures the fraction of signal left in the residual. ρ ≈ 1 means the
transient survives; ρ ≈ 0 means the model absorbed it. Report ρ maps over
(L, R) for each D and amplitude. The expected pattern is that ρ rises
towards 1 as R exceeds L.

### 5.2 Velocity repeatability

MIDAS-style velocity of (r − r̂) on 2022–2023 vs 2024–2025, per station.
Smaller differences mean less long-period distortion.

### 5.3 Ridgecrest

Fraction of the observed postseismic displacement (days 7–365 after
2019-07-06, relative to days 1–6) retained in r − r̂ at stations within
80 km.

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
