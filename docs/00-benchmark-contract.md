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
| `nbr_idx` | S × K | int32 | Indices of the K = 16 nearest stations within 150 km, nearest first; −1 pads. |
| `nbr_dist` | S × K | float32 | Great-circle distance, km. |
| `nbr_az` | S × K | float32 | Azimuth from station to neighbour, radians. |
| `split_day` | T | int8 | 0 train, 1 validation, 2 test. |
| `split_sta` | S | int8 | 0 normal, 1 held-out station. |
| `exclude` | S × T | bool | Cells barred from training (Ridgecrest window). |

S ≈ 1,381, T ≈ 6,900; the cube is ~80 MB per float array.

### How `r` is made

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

### Splits

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
```

`predict` receives the full cube; the harness, not the model, is
responsible for masking: before calling `predict` it sets `r`, `sigma` and
`avail` to NaN/False on hidden cells in a copy. A model that reads a hidden
value is a bug the tests must catch (section 6).

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

For each injection the model reconstructs the whole affected area (all
stations within 2L hidden in turn, leave-one-out), and

  ρ = ⟨r − r̂, s⟩ / ⟨s, s⟩

measures the fraction of signal left in the residual. ρ ≈ 1 means the
transient survives; ρ ≈ 0 means the model absorbed it. Report ρ maps over
(L, D) per amplitude.

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
