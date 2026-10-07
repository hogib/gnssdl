# Design docs

Design documents for the network-reconstruction benchmark described in
`paper/proposal.tex`. Each doc fixes the decisions an implementation must
follow so models stay comparable.

| Doc | Contents |
|---|---|
| [00-benchmark-contract.md](00-benchmark-contract.md) | Data cube, splits, masks, model interface, scoring. Every model doc assumes it. |
| [01-m0-zero.md](01-m0-zero.md) | M0: predict zero (the floor) |
| [02-m1-weighted-stack.md](02-m1-weighted-stack.md) | M1: distance-weighted neighbour mean |
| [03-m2-robust-neighbour.md](03-m2-robust-neighbour.md) | M2: robust neighbour statistic |
| [04-m3-ridge.md](04-m3-ridge.md) | M3: ridge regression on neighbours (per-station and pooled) |
| [05-m4-band-ridge.md](05-m4-band-ridge.md) | M4: frequency-dependent (band-split) regression |
| [07-m3k-learned-kernel.md](07-m3k-learned-kernel.md) | M3k: learned blind-spot kernel (normalized convolution with a learned distance/azimuth kernel) |
| [06-m5-graph-attention.md](06-m5-graph-attention.md) | M5: divided space-time attention network (conv patch stem), and the M5-aug variant |

## Plan

| Step | Status |
|---|---|
| Contract: data cube, masks, scoring, leak checks | done |
| M0, M1, M2 and the M2-self reference | done |
| Signal tests: Gaussian injections (§5.1), Ridgecrest (§5.3), noise removed (§5.4) | done |
| Protocol: validation-only decisions, prospective split, leave-one-out tuning, quiet training data, bootstrap comparisons | done |
| Exclusion radii to 400 km, exact injection packing, spurious signal, retention by amplitude | done |
| Audit filters: C1 regional stack, C3 sub-regional probabilistic PCA, D1 400 km stack, D2 Bachelot et al. GNN | to do |
| Event library: realistic, fault-consistent injections (§5.5) | designed |
| M3: ridge regression on neighbours | designed |
| M3k: learned blind-spot kernel (07 doc) | designed |
| M4: band-split ridge regression | designed |
| M5: divided space-time attention | designed; not judged until the event library exists |
| M5-aug: M5 trained with planted transients in the neighbours' inputs (06 doc) | designed |

The audit filters and the event library come before the learned models, so
that every model from M3 on is judged on realistic signals as well as the
blobs and against the filters in common use. The scoreboard after M3 and
M3k decides how much effort M4 and M5 deserve.
