# 07 — M3k: learned blind-spot kernel

## Purpose

M1 is a normalized convolution: each day is treated as an image whose
pixels are stations, and a missing or withheld pixel is filled with a
distance-weighted average of the pixels that exist,

  r̂_i(t) = Σ_j w(d_ij) · a_j(t) · r_j(t) / Σ_j w(d_ij) · a_j(t),

with a fixed Gaussian kernel w and a blind spot (the target and every
station within the exclusion radius R get zero weight). M3k keeps that form
and learns the kernel. It sits between M3 and M5 and answers two questions:

- does a learned kernel beat a fixed one (M3k vs M1)?
- in space, does attention beat a learned kernel (M5's spatial block vs
  M3k)? Same question as convolution versus attention, asked in space.

The stations are an irregular grid, so the kernel is a continuous one, as
in point-cloud convolutions (KPConv, parametric continuous convolution): a
small network maps each neighbour's offset to a weight, instead of a fixed
grid of weights.

## Model

For target i, day t, component c, over the K = 16 neighbours j beyond R
(`nbr_idx[R]`, contract §1.2):

  w_ijc = softplus(f_θ(g_ij, q_j))_c

  r̂_ic(t) = Σ_j w_ijc · a_j(t) · r_jc(t) / Σ_j w_ijc · a_j(t)

- g_ij, relative geometry: log distance, sin and cos of azimuth.
- q_j, neighbour quality (static): log robust noise scale per component and
  record completeness, the same static features as M5.
- f_θ: MLP, 2 hidden layers of 32, GELU, 3 outputs (one weight per
  component). About 2,000 parameters.
- softplus keeps weights positive, so the prediction is a weighted average
  of neighbours (a convex combination), like M1.

The kernel depends on where a neighbour is and what kind of station it is,
never on its values that day; that is the difference from attention (M5),
whose weights also depend on the content of the day.

Missing neighbours are handled by the normalisation itself, exactly as in
normalized convolution: a missing pixel contributes to neither sum, so no
dropout augmentation is needed (unlike M3). If no neighbour is available,
r̂ = 0 and the cell counts as a fallback, as in M1.

Blind spot. The target never enters its own sum, and stations within R are
not in `nbr_idx[R]`. One forward pass on the unhidden cube is therefore a
leave-one-out prediction for every station (`never_reads_target = True`),
like M1 and M2, so M3k uses the leave-one-out tuning and cleaning paths of
the harness unchanged.

Temporal context: `same-day` only. Own-history: off by construction.

## Training

- Data: quiet residuals on training cells, normalised per station
  (contract §1.5).
- Loss: the weighted squared error of M5, over every training cell of every
  station (the blind spot makes every station a target at once, so no
  masking is needed during training).
- Batching: one day is one "image"; a batch is 32 random training days,
  all stations at once. Each epoch visits every training day once.
- Optimiser: AdamW, lr 1e-3, weight decay 1e-4; early stopping on the
  validation leave-one-out fast score (contract §4, tuning criterion).
- One model per exclusion radius R. Seeds: 5.
- Runs on CPU in minutes; the GPU is optional.

## Interpretation

The learned kernel can be plotted: w as a function of distance and azimuth,
per component, for a typical station. That gives a data-driven picture of
how the common mode decays with distance and direction, to compare with the
correlation-distance relations of Tian & Shen (2016) and Kreemer & Blewitt
(2021). If the kernel is nearly isotropic and close to M1's Gaussian, M1
was already near optimal; anisotropy (for example along the plate
boundary) would be a finding in itself.

## Variants (ablations)

1. Geometry only: drop the neighbour-quality features q_j.
2. Signed kernel: weights not forced positive and no normalisation, which
   makes M3k a nonlinear relative of M3b's pooled regression.
3. K = 32 neighbours.
4. Multi-scale (M3k-ms), the image-pyramid idea: a second kernel over a
   coarse ring of distant stations (e.g. 300–1,000 km, beyond R) added to
   the local one, so the model can separate the network-wide common mode
   (reference frame, orbits) from the regional one (loading). Mirrors the
   sub-regional filtering of Liu et al. (2026) and the distance bands of
   Tian & Shen (2016).

## Expected behaviour

At least as good as M1, which it contains (a learned kernel equal to M1's
Gaussian is one solution), and close to M3b. Signal kept should be similar
to M1 at the same R: the kernel is still an average of neighbours, so a
transient that reaches them is still absorbed. Any gain should therefore
show as more noise removed at equal signal kept, which is what the
trade-off plot measures.

## Tests

The contract tests (`00-benchmark-contract.md` §6), plus:

- with f_θ fixed to output log of M1's Gaussian kernel (and q ignored), M3k
  reproduces M1's predictions to numerical precision;
- missing neighbours: predictions equal a direct computation of the
  normalized sum over available neighbours only;
- the target's prediction receives zero gradient from the target's own
  values.
