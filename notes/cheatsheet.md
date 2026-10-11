# Cheat sheet: models, metrics, results

Personal notes for cross-checking, not text for the paper. Numbers were
pulled from `data/results/` on 2026-10-11.

Citations: "contract §x" means `docs/00-benchmark-contract.md` section x;
other docs are named by file. Where a rule lives only in code, the code is
cited instead. Unless stated otherwise:
- every number is from the validation years;
- every filter excludes 0 km around the station (R = 0).

Test-year signal runs and the extra attention-network runs are still going;
see the end of this file.

---

## 1. How to read the metrics

All of these are medians over stations (or over injections), never pooled.
Pooled numbers let a handful of stations with big earthquake jumps dominate.

**Residual.** A station's position minus its fitted trajectory model, which
removes the trend, seasonal terms and equipment jumps (earthquake jumps stay
in). Everything below works on residuals. *Contract §1.1.*

**Filter.** Anything that predicts a station's residual (the "common mode")
from other stations. The cleaned series is the residual minus that
prediction. *Contract §2, §2.1.*

**Exclusion radius R (km).** The filter may not use stations closer than R
km to the one being cleaned. R = 0 means only the station itself is left
out ("leave-one-out"). A bigger R protects local signals but leaves the
filter fewer, more distant stations. *Contract §1.2.*

### Noise-side metrics (lower error / higher noise removed = better)

**Error** (fast leave-one-out error, "nrmse_fast"):
- This is the RMS of the prediction error in units of that station's normal
  noise level. Only the day-to-day and week-to-week part is counted.
- The slow part, a 61-day running median of the error, is split off and not
  counted here, because it is where real signals live.
- The baseline is predicting zero, which scores **0.723**, not 1 (the split
  itself eats some noise), so compare against 0.723.
- The differences are small in absolute terms. 0.51 vs 0.48 is a real gap,
  and 0.504 vs 0.508 usually is not; the bootstrap decides that.
- *Contract §4 (fast/slow split, per-station medians, "Tuning criterion",
  "Uncertainty" for the bootstrap); why zero prediction scores about 0.7, not 1, is
  explained in contract §4, "Fast and slow parts". `docs/01-m0-zero.md` still says
  "≈ 1": it was written before the fast/slow split and is out of date on this
  point. The 0.723 itself is from the Table 2 caption in `paper/pilot.tex`.*

**Noise removed:** 1 − (scatter after cleaning / scatter before), for the
fast part, averaged over E/N/U, median over stations. 26% means the cleaned
series has about a quarter less day-to-day scatter. Predicting zero scores
0%. *Contract §5.4.* (§5.4 says "on the test years"; the numbers in this
sheet are the validation-year version of the same formula.)

### Signal-side metrics (closer to 1 = better for kept, closer to 0 = better for spurious)

**Kept (ρ, "retention"):**
- We plant a fake signal, clean the series with and without it, and
  measure how much of it survives.
- ρ = 1 means it is fully kept. ρ = 0 means the filter absorbed it, which
  means it treated the signal as noise and deleted it.
- ρ > 1 can happen; the filter then adds to the signal.
- *Contract §5.1 (the ρ formula and the pairing with an uninjected run).*

**Kept 25 / 100 km:** the planted signals are round Gaussian blobs, where
every station within the width moves the same way. They come in four
widths: 10, 25, 50 and 100 km. A wide blob looks like network noise to the
filter, so it is the hard case. *Contract §5.1, "Grid".*

**Spurious:**
- When a signal is planted, stations *outside* it can change, because the
  filter spreads part of the signal to them.
- The number is the largest such change, as a fraction of the planted
  amplitude, for 100-km blobs.
- 0.11 means a station with nothing planted got a fake bump of 11% of the
  real signal's size. *Contract §5.1, "Spurious signal" (`spurious_max`).*

**Afterslip kept (15–40 km):**
- We plant realistic afterslip, computed from slip on a real fault patch
  in the SCEC fault model.
- The number is the median ρ at stations 15–40 km from the fault patch,
  where the signal is a few mm and easy to lose.
- Close to the fault, almost everything keeps the signal (see section 5).
- *The templates are in contract §5.5 and `src/gnssdl/bench/event_templates/`.
  The 15–40 km band is a summary choice made for the paper; it is defined in
  code (`_point` in `src/gnssdl/paper_figures.py`), not in a design doc.*

**Real offsets kept:**
- These are real earthquake jumps (≥ 3 mm, no other jump within 15 days).
- The number is the fraction of the jump still present after cleaning,
  measured over ±10 days.
- This is one of the two signal tests on real, not planted, signal (Ridgecrest
  is the other).
- *Not in a design doc: it was added after the contract was written. The rules
  are the `OFFSET_*` constants and `quake_offsets` in
  `src/gnssdl/bench/signal.py`; it is also described in `paper/proposal.tex`.*

**Ridgecrest kept:**
- This is the real postseismic motion after the 2019 Ridgecrest earthquake
  (days 7–365), at stations within 80 km.
- The number is the fraction kept after cleaning. No model trained on
  those days.
- These numbers are the same for every period, because 2019 is in neither
  the validation nor the test years.
- *Contract §5.3.*

### Two kinds of filters, scored differently

- **Leave-one-out filters** never read the station they clean. They get
  an error score and all the signal tests.
- **Target-inclusive filters** use the station's own value in its own
  correction (common practice). The error score would be meaningless for
  them, shown as "–". They get only noise removed and the signal tests.
- *Contract §2.1 (the exception for reference filters); `docs/08-audit-filters.md`,
  introduction.*

---

## 2. The filters, in plain words

| Code | Plain name | What it does | Parameters | Design doc |
|---|---|---|---|---|
| M0 | Zero | Predicts nothing; the floor | 0 | `01-m0-zero.md` |
| C1 | Network average | Mean of all stations, *including* the station itself (Wdowinski et al. 1997 style) | 0 | `08-audit-filters.md`, "C1" |
| C3 | Regional PCA | Network split into 1, 4 or 8 regions; 1–3 principal components removed per region; includes the station itself | ~3.7k per component | `08-audit-filters.md`, "C3" |
| M2-self | Robust median, self-included | Weighted median of the 16 nearest plus the station itself (CMC Imaging style) | 2 | `03-m2-robust-neighbour.md`, "Reference variant: M2-self" |
| FS | Far stack | Mean of *every* station beyond R. Simplest leave-one-out filter | 0 | `08-audit-filters.md`, "Far-stack sweep and neighbour count" |
| D1 | Beyond-400 km average | Far stack with R = 400 km (Bachelot et al. 2025 design) | 0 | `08-audit-filters.md`, "D1" |
| M1 | Weighted neighbour average | Distance-weighted mean of the 16 nearest beyond R; also run with 64 and 256 nearest | 1 | `02-m1-weighted-stack.md`; 64/256 in `08-audit-filters.md`, "Far-stack sweep and neighbour count" |
| M2 | Robust neighbour median | Weighted median of the 16 nearest beyond R | 2 | `03-m2-robust-neighbour.md` |
| M3a | Per-station regression | Ridge regression on the 16 nearest; separate weights for each station | 65 per station | `04-m3-ridge.md`, "Two variants", "As implemented" |
| M3b | Shared regression | Same, one set of weights for all stations, using neighbour distance/direction | 257 | `04-m3-ridge.md` |
| M4a / M4b | Band regression | M3a / M3b done separately in four period bands (<7, 7–60, 60–400, >400 days) | 4 × M3 | `05-m4-band-ridge.md`, "As implemented" |
| M3k | Learned distance kernel | Small neural net that sets each neighbour's weight from distance, direction and noise level; 16 or 256 neighbours | 1,411 | `07-m3k-learned-kernel.md`, "Model", "As implemented" |
| M5 | Attention network | Transformer over 32 neighbours + the far stack, 64-day window; learns a *correction to the far stack* | 212,048 | `06-m5-graph-attention.md`, "As implemented (first version)" |
| M5-aug | Attention network trained to ignore transients | M5, but during training fake transients are added to the neighbours' inputs (not the answers), in half the samples | 212,048 | `06-m5-graph-attention.md`, "Variant M5-aug" |

---

## 3. Main results (R = 0, validation years)

Error = leave-one-out fast error (lower is better; zero prediction = 0.723).
Same numbers as Table 2 of `paper/pilot.tex`, except the regional PCA offsets
range (see section 6).

| Filter | Error | Noise removed | Kept 25 / 100 km | Spurious | Afterslip 15–40 km | Real offsets | Ridgecrest |
|---|---|---|---|---|---|---|---|
| Network average (C1) | – | 26.7% | 0.99 / 0.83 | 0.11 | 1.00 | 0.98 | 0.92 |
| Regional PCA (C3, 9 settings) | – | 25.1–26.4% | 0.88–0.99 / 0.30–0.84 | 0.27–1.52 | 0.97–1.00 | 0.90–0.98 | 0.79–0.94 |
| Robust median, self-included | – | 25.0% | 0.24 / 0.00 | 0.10 | 0.19 | 0.60 | 0.34 |
| Far stack | 0.509 | 26.6% | 0.99 / 0.83 | 0.11 | 1.00 | 0.98 | 0.92 |
| Beyond-400 km (D1) | 0.526 | 24.9% | 1.00 / 1.00 | 0.24 | 1.00 | 0.99 | 0.91 |
| Weighted avg, 16 | 0.531 | 22.2% | 0.26 / 0.00 | 0.02 | 0.09 | 0.72 | 0.31 |
| Weighted avg, 256 | 0.507 | 26.4% | 0.84 / 0.21 | 0.10 | 0.96 | 0.83 | 0.84 |
| Robust median | 0.558 | 18.4% | 0.27 / 0.00 | 0.11 | 0.23 | 0.69 | 0.37 |
| Per-station regression | 0.527 | 22.6% | 0.43 / 0.17 | 0.04 | 0.22 | 0.77 | 0.50 |
| Shared regression | 0.534 | 21.4% | 0.30 / 0.07 | 0.04 | 0.24 | 0.74 | 0.50 |
| Band regression, per-station | 0.517 | 24.9% | 0.56 / 0.37 | 0.03 | 0.53 | 0.75 | 0.85 |
| Band regression, shared | 0.532 | 21.5% | 0.33 / 0.11 | 0.04 | 0.31 | 0.72 | 0.50 |
| Learned kernel, 16 | 0.517 | 23.9% | 0.28 / 0.01 | 0.03 | 0.11 | 0.81 | 0.36 |
| Learned kernel, 256 | 0.499 | 27.1% | 0.71 / 0.22 | 0.18 | 0.73 | 0.88 | 0.76 |
| Attention network | 0.504 | 26.8% | 0.80 / 0.56 | 0.15 | 0.87 | 0.88 | 0.91 |
| Attention, transient-trained | 0.508 | 26.5% | 0.97 / 0.80 | 0.17 | 0.99 | 0.98 | 0.92 |

How to read a row:
- Read left to right as "how much noise it removes" and then "what it costs
  in signal".
- A good filter has low error and high noise removed *and* kept close to 1
  with low spurious.
- The weighted average of 16 neighbours is the cautionary row: decent
  noise removal, but it deletes nearly every 100-km signal (0.00) and
  91% of distant afterslip (0.09).

---

## 4. Main findings, one line each

1. **How many stations you average matters more than how close they
   are.** Going from 16 to 256 neighbours lowers the error (0.531 → 0.507)
   and raises how much signal is kept. Averaging everything (the far stack)
   is as good as anything we tried at removing noise.
2. **The exclusion radius is a trade-off.** For the far stack, R from 0 to
   400 km:
   - 100-km signal kept goes from 0.83 to 1.00;
   - spurious signal goes from 0.11 to 0.24;
   - noise removed goes from 26.6% to 24.9%.

   No radius wins on everything. *The sweep's design: `08-audit-filters.md`,
   "Far-stack sweep and neighbour count".*
3. **Local filters erase the far field.** Filters built on the nearest 16
   stations keep afterslip next to the fault but delete it 15–40 km away
   (0.09–0.31 kept). Wide averages keep it (≥ 0.96).
4. **The learned kernel beats the plain weighted average at the same
   signal kept.** With 256 neighbours it removes the most noise of
   anything (27.1%), but it loses signal (0.22 of 100-km blobs).
5. **The attention network** sits between the learned kernel and the far
   stack. Trained with fake transients, it keeps signal like the far
   stack, but does not beat it on noise. That bootstrap comparison is
   still to run. *What it was meant to test: `06-m5-graph-attention.md`,
   "Variant M5-aug".*
6. **Planted and real signal agree.** The filters rank the same way on
   planted signals as on real ones: Spearman 0.94 for real earthquake
   offsets and 0.92 for Ridgecrest. That supports using planted signals for
   the audit.
7. **Regional PCA with one region and one component** creates fake signal
   larger than the real one (spurious 1.52). More regions make less fake
   signal but keep less of the real one.

---

## 5. Supporting numbers

### Radius sweeps (validation)

Error here is the mask-based "scatter" score from `data/results/*.csv`,
*not* the leave-one-out error of section 3; see the note in section 6.

| Filter | R (km) | Error (mask) | Noise removed | Kept 25 / 100 km | Afterslip 15–40 km |
|---|---|---|---|---|---|
| Far stack | 0 | 0.486 | 27% | 0.99 / 0.83 | 1.00 |
| | 25 | 0.487 | 27% | 0.99 / 0.84 | |
| | 50 | 0.488 | 27% | 1.00 / 0.85 | |
| | 100 | 0.490 | 26% | 1.00 / 0.89 | 1.00 |
| | 200 | 0.495 | 26% | 1.00 / 0.96 | |
| | 400 | 0.509 | 25% | 1.00 / 1.00 | 1.00 |
| | 600 | 0.544 | 22% | 1.00 / 1.00 | |
| Weighted avg, 16 | 0 | 0.545 | 22% | 0.26 / 0.00 | 0.09 |
| | 10 | 0.550 | 22% | 0.31 / 0.00 | |
| | 25 | 0.560 | 21% | 0.48 / 0.02 | |
| | 50 | 0.576 | 20% | 0.84 / 0.07 | |
| | 100 | 0.609 | 20% | 1.00 / 0.24 | 1.02 |
| | 200 | 0.651 | 17% | 1.00 / 0.69 | |
| | 400 | 0.632 | 16% | 1.00 / 1.00 | 1.00 |
| Weighted avg, 256 | 0 / 100 / 400 | 0.494 / 0.519 / 0.519 | 26 / 26 / 25% | 0.84→1.00 / 0.21→0.45→1.00 | 0.96 / 1.01 / 1.00 |
| Learned kernel, 16 | 0 / 100 / 400 | 0.510 / 0.559 / 0.581 | 24 / 22 / 18% | 0.28→1.00 / 0.01→0.25→1.00 | 0.11 / 1.02 / 1.00 |
| Learned kernel, 256 | 0 / 100 / 400 | 0.483 / 0.513 / 0.502 | 27 / 26 / 25% | 0.71→1.00 / 0.22→0.50→1.00 | 0.73 / 1.01 / 1.00 |

The attention network has only R = 0 so far; the other radii are queued
(planned in `06-m5-graph-attention.md`, "Run settings").

### Regional PCA settings (k = regions, p = components)

*Settings grid: `08-audit-filters.md`, "C3".*

| Setting | Noise | Kept 25 / 100 | Spurious | Afterslip | Offsets | Ridgecrest |
|---|---|---|---|---|---|---|
| k1 p1 | 26% | 0.99 / 0.84 | 1.52 | 1.00 | 0.98 | 0.92 |
| k1 p2 | 26% | 0.98 / 0.83 | 0.56 | 1.00 | 0.96 | 0.92 |
| k1 p3 | 26% | 0.98 / 0.82 | 0.56 | 1.00 | 0.96 | 0.92 |
| k4 p1 | 26% | 0.96 / 0.62 | 0.38 | 0.99 | 0.96 | 0.94 |
| k4 p2 | 26% | 0.95 / 0.55 | 0.39 | 0.99 | 0.91 | 0.94 |
| k4 p3 | 26% | 0.94 / 0.54 | 0.46 | 0.99 | 0.95 | 0.93 |
| k8 p1 | 25% | 0.91 / 0.33 | 0.27 | 0.98 | 0.90 | 0.84 |
| k8 p2 | 26% | 0.89 / 0.30 | 0.28 | 0.97 | 0.90 | 0.83 |
| k8 p3 | 26% | 0.88 / 0.30 | 0.28 | 0.97 | 0.91 | 0.79 |

### Event library, whole-event retention (median ρ over events)

*Contract §5.5 (templates, coverage rule, packing).*

| Filter | Afterslip | Shallow slow slip |
|---|---|---|
| Network avg, far stack, beyond-400, regional PCA | 0.99–1.00 | 0.98–1.00 |
| Weighted avg 16 / 256 | 0.88 / 0.99 | 1.00 / 1.00 |
| Robust median / self-included | 0.84 / 0.79 | 1.00 / 0.97 |
| Per-station / shared regression | 0.91 / 0.88 | 0.99 / 1.00 |
| Band regression per-station / shared | 0.94 / 0.89 | 1.00 / 1.00 |
| Learned kernel 16 / 256 | 0.85 / 0.93 | 0.98 / 0.97 |
| Attention / transient-trained | 0.97 / 1.00 | 1.00 / 1.00 |

Why everything looks good here:
- The whole-event number is dominated by the stations near the fault,
  which move a lot and are easy to keep.
- Shallow slow slip is small and local (8–25 km long), so even 16
  neighbours mostly sit outside it.
- The far-field column in section 3 (15–40 km) is where filters differ.

### Seeds (training repeats; mask-based "scatter" error, R = 0)

*5 seeds required by contract §4 ("Uncertainty"), `06-m5-graph-attention.md`
"Training", and `07-m3k-learned-kernel.md` "Training".*

| Model | Validation, seeds 0–4 | Test years, seeds 0–4 |
|---|---|---|
| Learned kernel, 16 | 0.510 0.510 0.509 0.509 0.510 | 0.619 0.620 0.618 0.619 0.618 |
| Learned kernel, 256 | 0.483 0.481 0.480 0.478 0.480 | 0.559 0.558 0.559 0.564 0.567 |
| Attention network | 0.479 0.479 0.480 0.480 (seed 4 running) | 0.548 0.548 0.548 0.553 |
| Attention, transient-trained | 0.484 0.482 0.485 (seeds 3–4 queued) | 0.552 0.551 0.553 |

Seed-to-seed spread is about ±0.002 on validation. Differences between
models smaller than that are noise.

---

## 6. Traps when cross-checking

- **Two different "error" numbers exist:**
  - Section 3 uses the leave-one-out error over *every* validation day
    (the tuning measure; it is what Table 2 in `pilot.tex` uses).
  - `data/results/<model>.csv` holds the mask-based scores (patterns
    `scatter`, `block`, `station`), computed only on hidden cells.

  They rank the filters almost the same way but are not equal (e.g. far
  stack 0.509 vs 0.486). Don't mix them in one table.
  *Masks: contract §3; tuning criterion: contract §4.*
- **Regional PCA offsets range:** Table 2 of `pilot.tex` says 0.90–0.95, but
  the results files give 0.90–0.98 (one region, one component: 0.98). This
  sheet uses the files. Table 2 will be refilled from the test years anyway.
- **Test-year errors are higher across the board** (mask-based: far stack
  0.553, attention network 0.548, transient-trained 0.552, learned kernel
  256 0.559, weighted avg 16 0.695). The test years are noisier. Compare
  models within a period, never across.
- **Ridgecrest numbers do not change between periods.** 2019 is in neither
  the validation nor the test years.
- **The weighted average with 64 neighbours** has blob, noise and Ridgecrest
  results only (no event or offset runs). It is a step in the
  neighbour-count story, not a headline filter.
- **The beyond-400 km filter (D1) is the far stack at R = 400.** The two
  are computed separately, and their numbers agree. *`08-audit-filters.md`, "D1".*
- **Kept > 1** (e.g. 1.02) is real: the filter added a little of the
  signal back. It is not a bug.

---

## 7. Still running / not yet in this sheet

- Test-year signal tests (blobs, events, offsets) for every filter: CPU
  chain, `data/logs/cpu_freeze.log`.
- Attention network seed 4 and transient-trained seeds 3–4, then their
  test-year signal tests: `data/logs/gpu_freeze.log`.
- Queued after that (`data/logs/gpu_radii.log`), in this order:
  - transient-trained attention at R = 25;
  - attention network at R = 25, 100, 50, 10;
  - each with 5 seeds, then validation and test-year signal tests.
- Bootstrap comparison: transient-trained attention vs far stack, and
  attention network vs far stack.
- Leave-one-out error for the test years (the section 3 error column on
  test years), to be computed when the figures are redrawn.
- Not planned for the pilot (listed in the design docs, not run): attention
  network own-history and causal settings and its 8 ablations
  (`06-m5-graph-attention.md`, "Run settings", "Ablations"); transient
  training at 25% of samples and the consistency-loss variant (same doc,
  "Variant M5-aug"); learned-kernel ablations (`07-m3k-learned-kernel.md`,
  "Variants"); correlation-weighted robust median
  (`03-m2-robust-neighbour.md`, "Notes").
