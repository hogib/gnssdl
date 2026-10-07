# 06 — M5: graph-attention temporal network

## Purpose

The deep model. It sees a station, its neighbours and a window of time
together, so it can learn rules the linear models can't express: weights
that change with which neighbours are present, with the time scale, and
with the situation (e.g. postseismic periods). The question it answers is
whether that flexibility beats M3/M4 on held-out data without absorbing real
transients.

## Sample definition

One training sample is a subgraph × window:

- Target station i (not held-out), plus its neighbours. Neighbours are the
  K = 16 nearest stations beyond the exclusion radius R (contract §1.2) with
  ≥ 50% availability in the window (dynamic, unlike M3's static slots),
  padded with masked dummies if fewer exist. Stations within R are never in
  the subgraph, so the model cannot see them.
- A window of W = 64 consecutive days, entirely inside the training split.
  Windows may touch excluded cells; those are treated as missing input and
  never as targets.

## Input features (per node, per day)

| Channel | Count |
|---|---|
| r (E, N, U), masked cells set to 0 | 3 |
| log σ (E, N, U), masked set to 0 | 3 |
| availability after masking | 1 |
| "hidden by mask" flag (vs naturally missing) | 1 |
| static: log robust noise scale s_i (E, N, U), repeated over days | 3 |
| static: fraction of days available in the station's training record | 1 |
| day of year, sin and cos | 2 |

14 channels. r and σ are divided by the station's robust scale s_i before
input; predictions are multiplied back. The static channels describe the
station itself (how noisy, how complete), so the network can weigh a
neighbour by its quality as well as its distance; day of year gives
seasonal context. Absolute latitude and longitude are deliberately not
inputs: they let a network memorise individual stations, which hurts on
held-out ones. Geometry enters only relative to the target, through the
edge features. Held-out stations get s_i from their
own unhidden data in the period being scored. That uses no hidden value, and
it is documented in the results.

Edge features (target → neighbour): log distance, sin/cos azimuth.

## Architecture

```
per node:  input 14×W ──Linear──▶ h ∈ R^{64×W}
repeat ×3:
  temporal block:  dilated residual 1D conv (kernel 3, dilations 1,2,4,8),
                   weights shared across nodes, GELU, LayerNorm
  spatial block:   for each day, every node attends to all nodes in the
                   subgraph (4 heads, d=64); attention logits get an
                   additive bias from an MLP(edge features); masked/dummy
                   nodes are keys with -inf where unavailable
output:   per node, per day Linear 64 → 3   (r̂ for E, N, U)
```

About 0.4–0.5 M parameters. The size is deliberately small: the network-wide
noise is one realisation per day, so there are only ~4,400 independent
training days (2008–2019).

## Run settings

Every M5 model is trained for one combination of the contract's run
settings (§1.2–1.3) and evaluated only at that combination:

| Setting | Values |
|---|---|
| exclusion radius R | 0, 10, 25, 50, 100 km |
| temporal context | `same-day` (W = 1), `causal`, `two-sided` |
| own-history | off, on |

How the context modes are implemented:
- `causal`: the temporal convolutions are left-padded (no access to later
  days), and spatial attention at day t only uses keys from days ≤ t.
- `two-sided`: symmetric padding.
- `same-day`: W = 1, so the temporal block is skipped.

The full grid is 5 × 3 × 2 = 30 configurations × 5 seeds, which is too many
to run blindly. Planned order:
1. two-sided, own-history off, all R. This is the main comparison with M1–M4.
2. Own-history on vs off at R = 0 and R = 25 km, all three contexts.
3. causal, own-history off, all R. This is what the continual detector will
   use.

## Masking during training (fresh every sample)

With own-history off, the target is always hidden for the whole window
(pattern `node` below with probability 1). The model learns to predict a
station purely from its neighbours, which is what cleaned series need
(contract §2.1).

With own-history on, pick one pattern per sample:

| Pattern | Probability | What is hidden |
|---|---|---|
| scatter | 0.4 | 15% of all node-days in the subgraph |
| block | 0.3 | 8–30 consecutive days at the target |
| node | 0.3 | the whole target for the whole window |

Independently, neighbour dropout: hide each neighbour entirely with
probability U(0, 0.3), so the model learns to cope with sparse
neighbourhoods.

## Loss

Over hidden cells with truth, all nodes in the subgraph:

  L = mean[ (r − r̂)² / (σ² + σ_floor²) ]

with σ_floor = 0.5 mm, so a few epochs with tiny formal σ can't dominate.
This is the weighted squared error of the proposal (eq. 3). A
learned-variance head is a later option, not in v1.

## Training

- Data: quiet residuals on training cells, normalised per station (contract
  §1.5): earthquake offsets estimated in the training-period fit removed,
  ±30 days around listed M ≥ 6 earthquakes and the Ridgecrest year skipped,
  held-out stations never used, as targets or as inputs.
- Optimiser: AdamW, lr 1e-3, weight decay 1e-4, cosine decay, 1k-step warm-up.
- Batch: 64 subgraphs; mixed precision. Fits in 8 GB (RTX 3060 Ti).
- Steps: up to 200k; validate every 2k steps on fixed validation masks;
  early stopping with patience 10 evaluations on validation `nrmse`.
- Seeds: 5 runs; report mean and spread.
- Logging: train/val loss, per-pattern val `nrmse`, and every 10k steps ρ
  on a small fixed validation injection set. That gives early warning of
  over-smoothing.

## Inference on the full cube

Slide the 64-day window with stride 32 for every station as target. Average
the two overlapping predictions with a triangular taper. Only the target
node's outputs are kept from each pass.

The target must be hidden in every pass that produces a cleaned series or a
signal-preservation score: the whole window is hidden, neighbours within R
are excluded, and own-history is off (contract §2.1). Running inference with
the target visible lets the network copy its input, so r − r̂ would be close
to zero and look like perfect denoising. Only the gap-filling scores
(`scatter`, `block`) with own-history on may leave the target's unhidden
days visible.

## Ablations (cheap, decided in advance)

1. No spatial block: temporal-only. Shows how much comes from neighbours.
2. No temporal block (W = 1): same-day only. This is the direct nonlinear
   counterpart to M3.
3. Static neighbours (M3's slots) instead of dynamic.
4. Static station features and day of year removed (back to 8 channels).
5. Own-history on vs off. Scored on gap filling and on ρ, to measure how
   much a transient's onset is carried from the target's visible days into
   the prediction (absorption through time).

## Variant M5-aug: transient-augmented training

Every other filter in the audit controls signal absorption structurally, by
choosing which neighbours it sees (a fixed 400 km, or the swept radius R).
M5-aug tries to learn it instead.

Idea. The plain M5 loss rewards predicting anything the neighbours share,
real transients included. M5-aug adds synthetic transients to the
neighbours' inputs during training but leaves the targets' labels
unchanged. Copying a planted transient into a prediction then increases the
loss, so the network learns to use the shared noise while ignoring
transient-shaped shared signal. It is the evaluation's planted-signal test
turned into a training objective.

Augmentation, per training sample (subgraph × 64-day window), with
probability p_aug:

- one transient field s added to the inputs of every node in the subgraph
  (the target's input is hidden anyway when own-history is off);
- centre at a random point within R + 300 km of the target, so the field can
  reach neighbours at any radius;
- width L log-uniform in 5–150 km, amplitude log-uniform in 1–15 times the
  station's noise scale, duration 5–60 days, raised-cosine rise and fall
  placed anywhere in the window, random horizontal direction;
- two shape families, half each: a Gaussian blob (all stations move
  together) and a fault-type dipole (the blob with its sign flipped across a
  random line through the centre, so the two sides move in opposite
  directions), so the network does not learn only one shape;
- labels unchanged: every hidden cell's target is the original residual,
  without s.

Optional consistency term (variant M5-aug-c): two forward passes on the
same sample, with and without s, and

  L = L_mse(clean pass) + λ · mean[(r̂(r + s) − sg(r̂(r)))²]

where sg stops the gradient. This states the invariance directly; λ ∈
{0.1, 1}.

Settings. p_aug ∈ {0.25, 0.5}; R ∈ {0, 25} km only, because the point is
whether learned invariance can replace a large exclusion radius. Each
setting is a point on the noise-removed vs signal-kept plot, compared with
the fixed-R curve of the other models.

Evaluation and leakage.

- The augmentation generator is separate from the evaluation injector
  (different seed, its own shape families) and runs on training days only.
  Evaluation injections and the event library are never seen in training.
- The decisive test is the event library (contract §5.5): if M5-aug keeps
  blobs and dipoles but not transplanted real events, it has learned the
  synthetic shapes rather than the general principle.
- Watch noise removal: a network that ignores every coherent anomaly also
  stops removing real common-mode bursts. The audit measures both sides.

Expected. If it works, M5-aug at R = 0 keeps much more of a 25–50 km
transient than M5 at R = 0, while keeping most of its noise removal; its
point sits above the fixed-R curve. That would be the main deep-learning
contribution of the study. If it does not, the fixed exclusion radius is
the better tool, which is also a clear result.

## Expected behaviour and how it could fail

- Wins expected at stations with intermittent neighbours, on `block` gaps
  (it can use the target's own history around the gap), and in band b3/b4
  content.
- Main failure mode: over-smoothing. Lower `nrmse` than M3 together with
  lower ρ on injections means it learned to absorb coherent signal. That
  result would be reported as such, and it argues against using M5 for
  transient detection.
- Memorisation of the common-mode history: watch the gap between train and
  validation `nrmse`; if it is large, reduce width or add dropout before
  adding data.

## Tests

The contract tests in `00-benchmark-contract.md` §6, run on a 50-station toy
cube on CPU in CI, plus:

- a gradient check that hidden cells receive zero gradient through the
  input path;
- in `causal` mode, a gradient check that outputs at day t receive zero
  gradient from inputs at days > t;
- determinism with a fixed seed (same loss after 100 steps).
