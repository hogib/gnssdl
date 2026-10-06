# 05 — M4: frequency-dependent (band-split) regression

## Purpose

Let neighbour weights depend on time scale. Neighbours may share a station's
day-to-day jitter but not its annual cycle, or the other way round. GNSS
noise has known structure at specific periods (fortnightly, ~351-day
draconitic), so a single same-day weight (M3) is a compromise. M4 is the
linear model that removes that compromise, standing in for a multichannel
Wiener filter on gappy data.

## Why not a true Wiener filter

A frequency-domain Wiener filter needs gap-free series. The cube has gaps
everywhere and the masks add more. Band-splitting in the time domain gives
the same effect while staying gap-tolerant.

## Algorithm

1. Gap fill (input only). On the masked input, fill every missing or hidden
   cell with M3b's prediction. This makes series continuous so they can be
   filtered. The fill never uses hidden values, because M3b is itself
   leak-tested.
2. Band split. Decompose each filled series into B = 4 bands with
   zero-phase filters that sum to identity:

   | Band | Periods |
   |---|---|
   | b1 | < 7 days |
   | b2 | 7 – 60 days |
   | b3 | 60 – 400 days |
   | b4 | > 400 days |

   Implementation: differences of Gaussian smoothers (σ chosen per cut-off)
   so the bands add up exactly to the input.
3. Per-band regression. For each band, fit an M3a-style ridge (and an
   M3b-style pooled version for unseen stations) predicting the target's band
   from the neighbours' same band. Same features, dropout augmentation and
   weighting as M3.
4. Prediction: r̂ = Σ_b r̂_b.

Variants mirror M3: M4a per-station, M4b pooled.

## Fitting details

- Filter edges: the band split is computed over the whole record before
  splitting into periods, but only from the masked input. Validation and
  test predictions must use the cube with all test-period hidden cells
  masked before filtering. Otherwise filter tails leak hidden values into
  neighbouring days. The leak test (contract §6.1) covers this and must be
  run with a block mask.
- λ per band, chosen on validation `nrmse`.

## Notes

- The filters here are non-causal, which is fine for the offline benchmark.
  The continual detector (paper §3.6) will need a causal version; that is
  out of scope here.
- Cost: four M3 fits plus filtering; minutes on CPU.

## Expected behaviour

Gains over M3 concentrated in b3/b4 (seasonal and draconitic content) and at
stations whose neighbours have different monuments or receivers. If M4 ≈ M3,
time-scale dependence is not important at this density, and M5's temporal
machinery has less to offer.

## Risks

- The step-1 fill makes M4 depend on M3b. A bad fill smears into the bands.
  Report M4's scores restricted to days with no hidden neighbours as a
  sanity check.
- Band edges are fixed by hand. A sensitivity run with shifted cut-offs
  (×0.5, ×2) goes in the appendix, not the tuning loop.
