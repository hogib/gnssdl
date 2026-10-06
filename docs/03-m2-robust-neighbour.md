# 03 — M2: robust neighbour statistic

## Purpose

M1 with outlier resistance, in the spirit of NGL's common-mode component
imaging (Kreemer & Blewitt 2021). One misbehaving neighbour (equipment
problem, local transient, bad day) should not leak into the prediction.
This is the stand-in for the operational standard, so it is the main bar a
learned model has to clear.

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
- Weighted median: sort by value, take the value where cumulative weight
  reaches half. Vectorise over stations per day; NumPy is enough.
- Cost: a few minutes on CPU.

## Expected behaviour

Close to M1 on clean days, clearly better on days with a bad neighbour. On
injections it should behave like M1, since a smooth transient that hits most
neighbours is not an outlier. If M2 and M3 end up close, that is a finding:
the operational approach is already near the linear optimum for this
network.

## Open questions

- Should we also compare against NGL's own common-mode-filtered series, if
  NGL publishes them for these stations? To check before implementation.
