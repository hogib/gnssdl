# 08 — Audit filters: C1, C3, D1, D2

The filters in common use, run through the same signal-loss audit as
M0–M5 (proposal §3.4). They come in two families:

- C, target-inclusive: the station's own value enters its own common-mode
  estimate. They are reference filters (contract §2.1 exception): never
  scored on the masks, only on signal kept, spurious signal and noise
  removed (contract §5).
- D, fixed distance exclusion (Bachelot et al. 2025): only stations at
  least 400 km away are used. D1 never reads the target and is scored like
  any model; D2 is scored in two versions (below).

Packing of planted transients (contract §5.1) needs to know which stations
a transient can affect. Each filter provides it:

| Filter | A transient inside footprint F can affect station x if |
|---|---|
| C1 | always (the stack uses every station) |
| C3 | x is in the same sub-region as any station of F |
| D1, D2 | x is in F, or some station of F is at least 400 km from x |

## C1 — regional stack (Wdowinski et al. 1997)

For each day and component, the common mode is the mean of the residuals of
every available station in the network, the target included:

  ĉ(t) = mean_j r_j(t),   r̂_i(t) = ĉ(t)

Unweighted, as in the original. The network here (California and
neighbours, ~1,000 km) is larger than the original southern California
network, which is part of what the audit shows: one stack for a large
region absorbs regional signals into a network-wide estimate.

## C3 — sub-regional probabilistic PCA (Gruszczynski et al. 2018; Liu et al. 2026)

1. Sub-regions: cluster the stations by location (k-means on projected
   coordinates) into regions of roughly 200–300 km, so the common mode can
   vary in space as Liu et al. (2026) recommend. The number of regions is a
   hyper-parameter chosen on validation noise removed.
2. In each region, fit probabilistic PCA by expectation-maximisation on the
   training-period residuals, treating missing days as latent (no
   interpolation), per component. Keep the leading components that explain
   the common mode; the number is chosen on validation noise removed (in
   the literature usually one or two).
3. On any day, estimate the component scores by weighted least squares from
   all available stations of the region, the target included, and
   reconstruct the common mode at every station.

Standard PCA filtering estimates each day's scores from all stations,
including the one being filtered, so C3 is target-inclusive like C1.

## D1 — mean of stations beyond 400 km (Bachelot et al. 2025)

As in their released code (`stack_filtering/stack_400km.ipynb`): for each
station, the unweighted mean of every available station at least 400 km
away, per day and component. Not limited to the nearest few. The target is
never used (distance 0), so D1 is leave-one-out and is scored on the masks
and the signal tests like M1. Its exclusion radius is fixed at 400 km.

## D2 — the graph neural network of Bachelot et al. (2025)

Reimplemented in plain PyTorch inside the harness, following their released
code (v1.0.1, `training/NN_architecture.py`, `GNN_training.py`,
`graph_construction/`), because their pipeline depends on their own NetCDF
preprocessing and the released training script does not run unmodified (an
undefined `lambda_center` argument).

Faithful elements:

- Windows of 30 days, shifted by 3; a station enters a window if at least
  80% of its days are present; missing days set to 0 (no availability
  flag); per-station min-max scaling to [-1, 1] on the training period.
- Graph: for each station, the 8 nearest stations more than 400 km away
  (their candidate set is the 300 nearest by latitude/longitude, then
  pruned); edge feature: distance in km; undirected candidates, the limit
  of 8 applying to edges leaving a station, as in their code.
- Model: MLP [90, 512, h] (ReLU, batch norm, as torch_geometric's MLP
  defaults), one GATv2 layer (2 heads, edge feature, no self-loops), MLP
  [2h, 512] (LeakyReLU, dropout 0.3), linear to 90. h = 64 as in their
  example command.
- Training: 30% of nodes masked at random (inputs set to 0), MSE on masked
  nodes, Adam, batches of 64 windows, early stopping with patience 50.
- Inference: 10 overlapping windows averaged per day.

Two versions:

| | D2-faithful | D2-protocol |
|---|---|---|
| Validation for early stopping | random 80/20 split of windows, as released | validation years (contract §1.4) |
| Target at inference | not masked, as released | masked (leave-one-out, contract §2.1) |
| Scored on | signal kept, spurious, noise removed (reads the target, so a reference filter) | everything, like any model |

The difference between the two measures the effect of the two choices we
flag in their code: validation on windows that overlap the training
windows by up to 90%, and a target that is unmasked at inference, which
GATv2's attention can read because its attention scores depend on the
target's own features even without self-loops.

Not reproduced: their preprocessing (maintenance-log corrections, tremor
co-location), which is specific to Cascadia; our cube's preprocessing is
used instead, the same for every filter.

## Implementation

`gnssdl.bench.audit`: C1, D1 (done first: simple, no training), C3, D2.
Each provides `affected_by(cube, inside)` for the packing above, and the
reference filters set `reference_filter = True`.
