# 04 — M3: ridge regression on neighbours

## Purpose

Learn how much each neighbour should count, instead of assuming it from
distance. This is the "predict station 4 from stations 1–3" idea, done for
every station. It is the strongest purely spatial, same-day linear model,
and the main bar for M4 and M5.

## Two variants

M3 comes in two variants, because per-station weights cannot be applied to a
station the model has never seen.

| Variant | Weights | Scored on |
|---|---|---|
| M3a, per-station | One weight vector per station and component | seen stations (`scatter`, `block`) |
| M3b, pooled | One weight vector shared by all stations, applied to neighbour "slots" | all patterns, including `station` |

## Features

Neighbour slots come from `nbr_idx[R]`, so a separate model is fitted for
every exclusion radius R (contract §1.2). Temporal context is `same-day`
only and the target's own values never enter its features, so own-history is
off by construction.

For target i, day t, target component c:

- for each neighbour slot k = 1…16 and each component c' ∈ {E, N, U}:
  r_{j_k,c'}(t) with missing values set to 0, plus the availability bit
  a_{j_k}(t)
- intercept

That gives 16 × 3 + 16 + 1 = 65 features. Using all three neighbour
components lets east at one station inform north at another, which matters
when the common mode is not aligned with the axes.

M3b adds per-slot geometry so a shared model can weight by geometry: for each
slot, log(d), sin(az), cos(az) and their products with the slot's three
residuals. That gives about 65 + 16 × 3 + 16 × 9 ≈ 260 features.

## Fitting

- Rows: training days where the target is available and not excluded.
- Missing-neighbour robustness: zero-filling biases a model fitted only on
  complete rows. For each training row, independently drop each neighbour
  (set to 0 and its bit to 0) with probability p_drop ~ U(0, 0.5). Replicate
  each row 4× with fresh drops. This teaches the weights to cope with any gap
  pattern without fitting one model per pattern.
- Loss: ridge with sample weights 1/σ²_i(t), closed form via the normal
  equations (65×65 per station-component for M3a; one ~260×260 system for
  M3b).
- λ: chosen on validation `nrmse` from a log grid 10⁻² … 10⁴ (in units of
  the trace of the Gram matrix / n_features). One λ for all stations in M3a,
  so it can't overfit per station.
- M3b trains on all non-held-out stations' rows. Held-out stations appear
  only as targets at evaluation time.

## Prediction

r̂_i(t) = features_i(t) · β, built from the masked input. If the target's
own value is hidden it doesn't appear in its own features, so nothing
changes. But if a neighbour is also hidden (the `station` pattern hides only
the target, `scatter` hides random cells), its slot is zeroed exactly as in
training.

## Cost

M3a: 1,381 stations × 3 components × a 65×65 solve; seconds. Building the
design matrices dominates; stream by station to keep memory low.

## Expected behaviour

Better than M1/M2 where the common mode varies across the neighbourhood and
where neighbours differ in quality. Same-day only: it cannot use the
target's own history, so on long `block` gaps it is no better than on
`scatter`. M4 and M5 add the time dimension.

## Risks

- Collinearity: neighbours are highly correlated; ridge handles it, but λ
  matters. Log the effective degrees of freedom per station.
- Stations with short training records: M3a weights are noisy. Require at
  least 365 training rows, otherwise fall back to M3b for that station and
  report the count.
