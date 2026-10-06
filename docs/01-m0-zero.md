# 01 — M0: zero prediction

## Purpose

The floor. Predicts r̂ = 0 everywhere, i.e. "the station did exactly what its
training-period trajectory model says". Any model that does not beat M0 has
learned nothing from the network.

## Algorithm

`predict` returns a zero array of shape S×T×3. `fit` does nothing.

## Expected scores

- `nrmse` ≈ 1 on all patterns (by definition of the normalisation), slightly
  above 1 on test days, because the trajectory model drifts out of date.
- ρ = 1 for every injection: nothing is removed.
- `cmr` = 0.

The amount by which M0's test `nrmse` exceeds 1 measures how much the
extrapolated trajectory degrades over 2022–2026. That number is worth
reporting on its own, because it sets how often the cube's baselines must be
refitted in the continual setting.

## Why keep it

It anchors the scoreboard and doubles as the reference for the
signal-preservation tests: any ρ < 1 in another model is signal that model
removed.

## Tests

The contract tests in `00-benchmark-contract.md` §6, except toy recovery
(M0 is expected to fail it, and the test asserts that it does).
