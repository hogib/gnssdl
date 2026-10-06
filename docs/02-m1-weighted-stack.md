# 02 — M1: distance-weighted neighbour mean

## Purpose

The classical regional stack (Wdowinski et al. 1997): a station's common mode
is the average of its neighbours' residuals on the same day. This is the
simplest model that uses the network.

## Algorithm

For station i, day t, component c:

  r̂_i(t) = Σ_j w_ij · a_j(t) · r_j(t) / Σ_j w_ij · a_j(t)

over the K = 16 neighbours j in `nbr_idx[R][i]` (neighbours beyond the
exclusion radius R, contract §1.2), where a_j(t) is availability
after masking and

  w_ij = exp(−d_ij² / 2L²) / σ̄_j²

- d_ij: distance from `nbr_dist`.
- σ̄_j: station j's median formal σ over the training period, per
  component, so noisy stations count less.
- L: length scale. L = ∞ gives the uniform stack of the original method.

If no neighbour is available, r̂_i(t) = 0 (falls back to M0) and the cell is
counted in a `fallback` statistic that is reported with the scores.

Components are treated independently.

## Fitting

The only parameter is L. Choose
L ∈ {10, 25, 50, 100, 200, 500, ∞} km by validation `nrmse` on the `scatter`
pattern; one L for all stations and components. The choice and the full
validation curve go in `config.json`. L is chosen separately for each
exclusion radius R; the curve of best L against R is reported, because it
shows whether the optimal stacking scale simply tracks the excluded zone.

## Notes

- Temporal context: `same-day` only; own-history is not used. Requests
  for `causal` or `two-sided` raise.
- Neighbours are the static `nbr_idx[R]` list. Stations that start or stop
  inside the record simply drop out through a_j(t).
- The target never contributes to its own prediction, so the leak test is
  structural, but it must still pass.
- Cost: O(S·T·K), well under a minute on CPU in NumPy.

## Expected behaviour

Strong on day-to-day common-mode scatter at dense sites. Weak where the
common mode varies spatially faster than L, near network edges (coast,
Nevada), and on large-footprint injections: anything coherent over ~L is
absorbed, so ρ should fall sharply once the injection footprint exceeds L.
That fall-off is the reference curve the other models are compared against.
