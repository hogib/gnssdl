# 03 — M2: robust neighbour statistic

## Purpose

M1 with outlier resistance, in the spirit of common-mode component imaging
(CMC Imaging; Kreemer & Blewitt 2021). One misbehaving neighbour (equipment
problem, local transient, bad day) should not leak into the prediction.
This is the stand-in for the best published robust filter, so it is the
main bar a learned model has to clear. (Kreemer & Blewitt do not state that
NGL applies CMC Imaging to its distributed series, so the docs and paper do
not call it "operational".)

It is not a reimplementation of CMC Imaging, whose exact procedure we have
not reproduced. The doc and the paper must call it "robust neighbour
statistic", not "CMC Imaging".

## Algorithm

For station i, day t, component c, take the available neighbour values
{r_j(t)} from `nbr_idx[R][i]` (beyond the exclusion radius R) with weights
w_ij from M1 (same kernel, its own L).

1. Neighbour pre-screening: discard r_j(t) if |r_j(t) − m_j(t)| > k · s_j,
   where m_j(t) is neighbour j's 31-day running median and s_j its robust
   scale. This removes neighbour-local spikes.
2. Prediction: weighted median of the remaining values with weights w_ij.
3. If fewer than n_min = 3 neighbours remain, fall back to M1, and if that
   is impossible, to 0; count both cases.

The running median in step 1 must be computed from the masked input only, so
a hidden cell never influences a neighbour's screening.

## Fitting

Grid search on validation `nrmse` (`scatter` and `block` averaged):

- L ∈ {25, 50, 100, 200, 500, ∞} km
- k ∈ {2, 3, 5, ∞} (∞ disables pre-screening)

Tuned separately for each exclusion radius R.

## Notes

- Temporal context: `same-day` for the prediction itself. The 31-day running
  median in step 1 uses neighbours' surrounding days, which is allowed: it
  never touches the target's own data, so own-history stays off.
- Weighted median: follow Kreemer & Blewitt (2021, §3.6). Normalise the
  weights, sort the values, give each value the percentile
  p_j = 100 · (c_j − w'_j / 2), where c_j is the cumulative normalised weight,
  and interpolate linearly between the two values that straddle p = 50.
  Taking "the first value whose cumulative weight reaches half" instead
  disagrees with the plain median for equal weights (four equal values give
  the second, not the mean of the second and third), and the interpolation
  stops the result from snapping to a single neighbour's value.
  Vectorise over stations per day; NumPy is enough.
- Cost: a few minutes on CPU.

## Expected behaviour

Close to M1 on clean days, clearly better on days with a bad neighbour. On
injections it should behave like M1, since a smooth transient that hits most
neighbours is not an outlier. If M2 and M3 end up close, that is a finding:
the robust-median approach is already near the linear optimum for this
network.

## Differences from CMC Imaging

Read from Kreemer & Blewitt (2021). Kept deliberately, but listed so the
paper can state them:

| | CMC Imaging | M2 |
|---|---|---|
| Neighbours | Delaunay neighbours of the day, plus any station within their median distance (cap 1,325 km); co-located stations (< 20 m) excluded | 16 nearest with R < d ≤ 300 km |
| Target station | Included in its own median when it is a filter station, weighted by its zero-distance correlation | Never (leave-one-out, contract §2.1) |
| Weights | Correlation predicted from a robust (Theil–Sen) linear fit of correlation vs distance, divided by the formal σ | M1 weights (Gaussian distance kernel × 1/σ̄²) |
| Bad stations | Whole stations dropped as filters if their zero-distance correlation is > 3σ below their neighbours' | Per-day spike screening only (step 1) |
| Trend bias | Hierarchical correction of short series for long-period common mode | None (trajectory fitted on the training period) |
| Evaluation | Scatter and velocity repeatability, no injected signals | nrmse plus the §5 signal-preservation tests |

Optional extension of the grid: weights w_ij = ĉ(d_ij) / σ_j, with ĉ the
Theil–Sen fit of training-period correlation against distance. This is the
CMC Imaging weighting and costs one extra grid value.

## Reference variant: M2-self

The most consequential difference is the target station. CMC Imaging puts
the station's own residual into its common mode, so any local transient at
that station is partly removed by construction. The authors say as much:
much of the method's scatter reduction comes from local common-mode
transients being absorbed into the common mode. M2 at R = 0 therefore
removes *less* signal than the published method, and the benchmark would
flatter it.

To measure this, add a reference filter, M2-self:

- Same as M2 at R = 0, but the target's same-day residual r_i(t) joins the
  weighted median with the M1 weight at zero distance, 1/σ̄_i² (or
  ĉ(0)/σ_i with the correlation weights).
- It sees the target, so it is **not** a `Reconstructor` and is never scored
  on `nrmse`, `wmse` or the masks: those numbers would be trivially good.
- It is scored only on `cmr`, ρ (§5.1) and Ridgecrest retention (§5.3),
  and appears as a single labelled point on the cmr-vs-ρ plot.

Expected: higher `cmr` and lower ρ than M2 at R = 0, most strongly for
small footprints L. The gap is the signal a CMC-Imaging-style filter
removes that a leave-one-out filter keeps.

## Open questions

- M2-self needs a code path outside `gnssdl bench clean`, which the
  contract (§2.1) currently names as the only producer of cleaned series.
  Either add an explicit exception for labelled reference filters to the
  contract, or compute M2-self only inside the injection and Ridgecrest
  scorers.
- Comparing against NGL's own common-mode-filtered series: the filtered
  series released with Kreemer & Blewitt (2021) (Harvard Dataverse,
  doi:10.7910/DVN/ONATFP) cover western Europe only. No filtered product
  for California has turned up in the literature reviewed so far; check
  the NGL website before deciding. M2-self is the fallback.
