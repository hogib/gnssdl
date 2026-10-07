# 08 — Audit filters: C1, C3, D1

The filters in common use, run through the same signal-loss audit as
M0–M5 (proposal §3.4). They come in two families:

- C, target-inclusive: the station's own value enters its own common-mode
  estimate. They are reference filters (contract §2.1 exception): never
  scored on the masks, only on signal kept, spurious signal and noise
  removed (contract §5).
- D, fixed distance exclusion: only stations at least 400 km away are used,
  the design of Bachelot et al. (2025). D1 never reads the target and is
  scored like any model.

For the graph neural network of Bachelot et al. (2025) we use the results
as they report them; it is not reimplemented here.

Packing of planted transients (contract §5.1) needs to know which stations
a transient can affect. Each filter provides it:

| Filter | A transient inside footprint F can affect station x if |
|---|---|
| C1 | always (the stack uses every station) |
| C3 | x is in the same sub-region as any station of F |
| D1 | x is in F, or some station of F is at least 400 km from x |

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

As described in their paper: for each station, the average displacement of
the stations more than 400 km away, per day and component, unweighted and
not limited to the nearest few. The target is never used (distance 0), so
D1 is leave-one-out and is scored on the masks and the signal tests like
M1. Its exclusion radius is fixed at 400 km, which is also one point of the
leave-one-out models' radius sweep.

## Implementation

`gnssdl.bench.audit`: C1 and D1 (simple, no training), then C3. Each
provides `affected_by(cube, inside)` for the packing above, and the
reference filters set `reference_filter = True`.
