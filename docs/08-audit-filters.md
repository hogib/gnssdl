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
   coordinates, best of 10 k-means++ starts), so the common mode can vary
   in space as Liu et al. (2026) recommend.
2. In each region, fit probabilistic PCA by expectation-maximisation
   (Tipping & Bishop 1999), per component, treating missing days as latent
   (no interpolation). Data: training-period quiet residuals (contract
   §1.5), divided by each station's robust scale, with the Ridgecrest year
   and ±30 days around listed M ≥ 6 earthquakes left out. Zero mean, since
   the residuals are already detrended.
3. On any day, the component scores are the pPCA posterior mean given all
   available stations of the region, the target included, and the common
   mode at each station is its loadings times the scores.

Stations with fewer than 365 training days get loadings by regressing their
first 730 available days on the scores of the other stations, the same
fallback window the cube uses for their trajectory fit. They are listed in
the config.

Number of regions and components: swept, not tuned. C3 reads the target,
so the noise-removed measure rewards it for absorbing whatever the target
shares with its region, signal included; choosing the setting on noise
removed would favour absorption. (More components do not always remove more
noise in the evaluation years, since the loadings are fitted on 2008–2019:
with one region, 3 components removed less than 2.) Every setting (1, 4 or
8 regions × 1, 2 or 3 components) is instead reported as its own point on
the noise-removed vs signal-kept plot (`c3-k4p1` = 4 regions, 1 component),
as the exclusion radius is for the leave-one-out models. One region is a
network-wide PCA filter; with 4 and 8, regions are a few hundred km across.

Standard PCA filtering estimates each day's scores from all stations,
including the one being filtered, so C3 is target-inclusive like C1. Common
practice also fits the loadings on the series being filtered; fitting them
on the training years instead is the more lenient choice for C3, because a
transient in the evaluation period cannot shape the loadings.

## D1 — mean of stations beyond 400 km (Bachelot et al. 2025)

As described in their paper: for each station, the average displacement of
the stations more than 400 km away, per day and component, unweighted and
not limited to the nearest few. The target is never used (distance 0), so
D1 is leave-one-out and is scored on the masks and the signal tests like
M1. Its exclusion radius is fixed at 400 km, which is also one point of the
leave-one-out models' radius sweep.

## Far-stack sweep and neighbour count

D1 differs from M1 in two ways at once: it excludes more (400 km) and it
averages far more stations (hundreds rather than 16). Two controls separate
the effects:

- fs, the far stack at any radius: the mean of every available station at
  least R away, for R = 0, 25, 50, 100, 200, 400 and 600 km (all other
  stations at R = 0). D1 is its 400 km point.
- M1 with K = 64 and K = 256 nearest neighbours beyond R instead of 16.
  Neighbour lists stop at R + 300 km, so at small R the largest K means
  "every station within 300 km".

The cube stores 256 neighbours per station and radius, nearest first; each
model uses its first `n_neighbours` columns (16 unless stated), so results
with K = 16 are unchanged.

## Implementation

`gnssdl.bench.audit`: C1, C3, D1 and the fs sweep. Each
provides `affected_by(cube, inside)` for the packing above, and the
reference filters set `reference_filter = True`.
