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
| [06-m5-graph-attention.md](06-m5-graph-attention.md) | M5: graph-attention temporal network |

## Plan

| Step | Status |
|---|---|
| Contract: data cube, masks, scoring, leak checks | done |
| M0, M1 | done |
| Signal tests: Gaussian injections (§5.1), Ridgecrest (§5.3), noise removed (§5.4) | done |
| M2 and the M2-self reference | implemented, results pending |
| Event library: realistic, fault-consistent injections (§5.5) | designed |
| M3 | designed |
| M4 | designed |
| M5 | designed; not judged until the event library exists |

The event library comes before M3 so that every model from M3 on is judged
on realistic signals as well as the blobs, and M1/M2 are rerun on it once.
The scoreboard after M3 decides how much effort M4 and M5 deserve.
