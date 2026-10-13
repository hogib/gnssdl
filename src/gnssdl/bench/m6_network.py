"""M6: whole-network space-time attention (post-freeze experiment).

Added after the method freeze of 2026-10-11 (contract §1.4): its test-year
numbers are exploratory, and only the prospective period is a clean test.

One sample is the whole network over a W-day window. A set of target
stations is hidden, together with every station within the exclusion radius
R of any target, and the network predicts the targets' residuals from every
other station. Training draws a fresh hidden set every sample (masked
autoencoding); prediction partitions the stations into fixed groups, one
pass per group, so every station is predicted from a pass in which it and
its R-disc were hidden. The prediction for a station therefore never reads
that station or any station within R (`never_reads_target`).

Inputs are the residuals only: per station and day, E, N, U divided by the
station's robust noise scale, and an availability bit. Geometry enters only
as a learned additive attention bias for every pair of stations, from their
log distance and azimuth; there are no absolute coordinates, so the network
cannot memorise stations.

Architecture:
- stem: 1-D convolution along time, shared by all stations (kernel = stride
  = P), giving W/P time tokens per station, plus a learned time-position
  embedding; hidden stations get a learned mask embedding;
- blocks: attention over a station's own time tokens (both directions in
  time), then attention over all stations at the same time token (both
  directions in space, with the pairwise geometry bias; hidden stations
  and stations without data in that token are never keys), each followed
  by an MLP; pre-LayerNorm;
- one extra always-available "register" node, so a query always has a key;
- head: per station and time token, Linear d -> P × 3, un-patched to days.

M6-aug adds planted fields (Gaussian blobs or fault-like dipoles) to the
visible stations' inputs in a share of the samples, leaving the targets'
labels unchanged, as M5-aug does.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gnssdl.bench.base import Reconstructor, station_scale
from gnssdl.bench.paths import RUNS, checkpoint_dir, log_dir
from gnssdl.bench.train import quiet_residuals
from gnssdl.dataset import Cube, crop_days, distance_azimuth

MODEL_DIR = RUNS                                    # runs root; see gnssdl.bench.paths
EARTH_KM = 6371.0
N_CHANNELS = 4                                      # E, N, U (noise-scale units), availability
N_PAIR = 5                                          # log distance, sin az, cos az, self, register


@dataclass
class M6Config:
    window: int = 64
    patch: int = 4
    d_model: int = 64
    heads: int = 4
    blocks: int = 3
    mlp_ratio: int = 2
    dropout: float = 0.1
    batch: int = 2                    # whole-network windows per step
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 500
    max_steps: int = 15000
    eval_every: int = 500
    patience: int = 6
    # share of stations hidden per sample (targets and their R-discs). Inference
    # hides 0.3 at every radius, inside the training range, so the radius sweep
    # stays comparable; smaller shares need many more passes at large R
    # (R = 100 km: ~400 groups at 0.1, ~50 at 0.3)
    train_hide: tuple[float, float] = (0.05, 0.3)
    infer_hide: float = 0.3
    min_target_days: int = 16
    group_seed: int = 0
    # M6-aug
    p_aug: float = 0.0
    aug_fields: tuple[int, int] = (1, 3)             # planted fields per augmented sample
    aug_width_km: tuple[float, float] = (5.0, 150.0)
    aug_amp: tuple[float, float] = (1.0, 15.0)       # times the centre target's horizontal noise scale
    aug_duration: tuple[int, int] = (5, 60)
    infer_batch: int = 2


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def build_net(cfg: M6Config):
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.h, self.dh = cfg.heads, cfg.d_model // cfg.heads
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
            self.out = nn.Linear(cfg.d_model, cfg.d_model)
            self.drop = cfg.dropout

        def forward(self, x, mask=None):              # x: (..., L, d); mask: broadcastable to (..., h, L, L)
            *lead, L, d = x.shape
            q, k, v = self.qkv(x).reshape(*lead, L, 3, self.h, self.dh).unbind(-3)
            q, k, v = (t.transpose(-2, -3) for t in (q, k, v))   # (..., h, L, dh)
            if mask is not None:
                mask = mask.to(q.dtype)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               dropout_p=self.drop if self.training else 0.0)
            return self.out(o.transpose(-2, -3).reshape(*lead, L, d))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))
            self.t_att, self.s_att = Attention(), Attention()
            self.m1 = nn.Sequential(nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(), nn.Linear(cfg.mlp_ratio * d, d))
            self.m2 = nn.Sequential(nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(), nn.Linear(cfg.mlp_ratio * d, d))

        def forward(self, x, s_mask):                 # x: (B, N, Tt, d); s_mask: (B, Tt, h, N, N)
            x = x + self.t_att(self.n1(x))            # over time, within a station
            x = x + self.m1(self.n2(x))
            xs = x.transpose(1, 2)                    # (B, Tt, N, d)
            xs = xs + self.s_att(self.n3(xs), s_mask)  # over stations, within a time token
            xs = xs + self.m2(self.n4(xs))
            return xs.transpose(1, 2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d, tt = cfg.d_model, cfg.window // cfg.patch
            self.stem = nn.Conv1d(N_CHANNELS, d, kernel_size=cfg.patch, stride=cfg.patch)
            self.time_pos = nn.Parameter(torch.zeros(tt, d))
            self.mask_emb = nn.Parameter(torch.zeros(d))
            self.register = nn.Parameter(torch.zeros(d))
            self.pair = nn.Sequential(nn.Linear(N_PAIR, 32), nn.GELU(), nn.Linear(32, cfg.heads))
            self.blocks = nn.ModuleList(Block() for _ in range(cfg.blocks))
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, cfg.patch * 3)
            nn.init.normal_(self.mask_emb, std=0.02)
            nn.init.normal_(self.register, std=0.02)

        def forward(self, feats, hidden, pair, targets):
            """feats: (B, S, W, C); hidden: (B, S) bool; pair: (S+1, S+1, N_PAIR);
            targets: (B, n) station indices to return. Returns (B, n, W, 3)."""
            B, S, W, C = feats.shape
            P = cfg.patch
            x = self.stem(feats.reshape(B * S, W, C).transpose(1, 2)).transpose(1, 2)   # (B*S, Tt, d)
            x = x.reshape(B, S, -1, x.shape[-1])
            x = x + hidden[:, :, None, None].to(x.dtype) * self.mask_emb
            reg = self.register.expand(B, 1, x.shape[2], -1)
            x = torch.cat([x, reg], 1) + self.time_pos                                   # (B, S+1, Tt, d)
            # keys: visible stations with data in the token's days, and the register
            a = feats[..., C - 1].reshape(B, S, -1, P).amax(-1) > 0                      # (B, S, Tt)
            key = (a & ~hidden[:, :, None]).transpose(1, 2)                              # (B, Tt, S)
            key = torch.cat([key, torch.ones_like(key[..., :1])], -1)                    # (B, Tt, S+1)
            # one mask for every block, built once in the compute dtype: at
            # B × Tt × h × (S+1)² it is the largest tensor in the network
            bias = self.pair(pair).permute(2, 0, 1)                                      # (h, S+1, S+1)
            s_mask = bias[None, None].masked_fill(~key[:, :, None, None, :], float("-inf"))
            for blk in self.blocks:
                # whole-network attention keeps B × Tt × h × (S+1)² weights per
                # block for the backward pass; recompute them instead (training only)
                if self.training and torch.is_grad_enabled():
                    x = torch.utils.checkpoint.checkpoint(blk, x, s_mask, use_reentrant=False)
                else:
                    x = blk(x, s_mask)
            xt = torch.gather(x, 1, targets[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3]))
            y = self.head(self.norm(xt))                                                 # (B, n, Tt, P*3)
            return y.reshape(B, targets.shape[1], W, 3)

    return Net()


def pair_features(cube: Cube) -> np.ndarray:
    """(S+1)×(S+1)×5 pair features from row (query) to column (key) station:
    log distance / 5, sin and cos of the azimuth, a self flag, and a flag for
    pairs with the register node (last row and column)."""
    d, az = distance_azimuth(cube.lat, cube.lon)
    S = len(cube.sta)
    out = np.zeros((S + 1, S + 1, N_PAIR), dtype=np.float32)
    out[:S, :S, 0] = np.log(np.maximum(d, 1.0)) / 5.0
    out[:S, :S, 1] = np.sin(az)
    out[:S, :S, 2] = np.cos(az)
    out[np.arange(S), np.arange(S), 1:3] = 0.0
    out[np.arange(S + 1), np.arange(S + 1), 3] = 1.0
    out[S, :, 4] = out[:, S, 4] = 1.0
    return out


def hide_groups(near: np.ndarray, share: float, seed: int) -> list[np.ndarray]:
    """Partition the stations into prediction groups. A group's hidden set
    is its targets and every station within R of them (`near`, S×S bool,
    the diagonal included); targets are added in a fixed random order while
    the hidden set stays within `share` of the network (at least one target
    per group). Depends only on geometry, never on data."""
    S = len(near)
    order = np.random.default_rng(seed).permutation(S)
    cap = max(1, int(round(share * S)))
    groups, left = [], list(order)
    while left:
        hidden = np.zeros(S, dtype=bool)
        group, rest = [], []
        for i in left:
            new = hidden | near[i]
            if not group or new.sum() <= cap:
                group.append(i)
                hidden = new
            else:
                rest.append(i)
        groups.append(np.asarray(group))
        left = rest
    return groups


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class M6Network(Reconstructor):
    name = "m6"
    never_reads_target = True
    supported_contexts = ("two-sided",)
    p_aug = 0.0
    post_freeze = True

    def __init__(self, radius_km: float = 0.0, seed: int = 0, config: M6Config | None = None, **kw):
        kw.setdefault("context", "two-sided")
        super().__init__(radius_km, **kw)
        self.seed = int(seed)
        self.cfg = config or M6Config(p_aug=self.p_aug)
        self.event_window: np.ndarray | None = None      # set by the harness
        self.history: list[dict] = []
        self._net = None

    # ---- static tables ---------------------------------------------------
    def _tables(self, cube: Cube) -> None:
        torch = _torch()
        scale = station_scale(cube)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nanmedian(scale, axis=0))
        d, _ = distance_azimuth(cube.lat, cube.lon)
        near = d <= self.radius_km
        np.fill_diagonal(near, True)
        lat, lon = np.deg2rad(cube.lat), np.deg2rad(cube.lon)
        xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        self._scale_np, self._near = scale, near
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self._dev)
        self._T = {"scale": t(scale), "pair": t(pair_features(cube)), "near": t(near, torch.bool),
                   "xyz": t(xyz)}

    def _inputs(self, r: np.ndarray, avail: np.ndarray) -> dict:
        torch = _torch()
        x = np.where(avail[..., None], np.nan_to_num(r, nan=0.0) / self._scale_np[:, None, :], 0.0)
        return {"x": torch.as_tensor(x.astype(np.float32), device=self._dev),
                "a": torch.as_tensor(avail, dtype=torch.float32, device=self._dev)}

    def _feats(self, D: dict, t0, hidden):
        """(B, S, W, 4) inputs for windows starting at `t0` (B), with the
        stations in `hidden` (B×S bool) blanked."""
        torch = _torch()
        W = self.cfg.window
        days = t0[:, None] + torch.arange(W, device=self._dev)[None, :]                # B×W
        x = D["x"][:, days].permute(1, 0, 2, 3)                                        # B×S×W×3
        a = D["a"][:, days].permute(1, 0, 2)                                           # B×S×W
        a = a * (~hidden)[..., None]
        return x * a[..., None], a

    # ---- training samples ------------------------------------------------
    def _draw_hidden(self, avail_w: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        """One training sample's targets and hidden set, from the window's
        S×W availability: targets have at least `min_target_days` days, and
        targets are added until the hidden share of active stations is
        reached."""
        cfg = self.cfg
        active = avail_w.any(axis=1)
        cand = np.flatnonzero(avail_w.sum(axis=1) >= cfg.min_target_days)
        share = rng.uniform(*cfg.train_hide)
        cap = max(1, int(round(share * active.sum())))
        hidden = np.zeros(len(active), dtype=bool)
        targets = []
        for i in rng.permutation(cand):
            new = hidden | self._near[i]
            if targets and (new & active).sum() > cap:
                continue
            targets.append(i)
            hidden = new
            if (hidden & active).sum() >= cap:
                break
        return np.asarray(targets, dtype=np.int64), hidden

    def _augment(self, x, a, targets, rng):
        """Add planted fields to the visible stations' horizontal inputs
        (x: B×S×W×3 in noise-scale units, a: B×S×W) for a p_aug share of
        the samples. Labels are taken separately and stay unchanged."""
        torch = _torch()
        T, cfg, dev = self._T, self.cfg, self._dev
        B, S, W, _ = x.shape
        x = x.clone()
        for b in range(B):
            if rng.random() >= cfg.p_aug:
                continue
            for _ in range(rng.integers(cfg.aug_fields[0], cfg.aug_fields[1] + 1)):
                tg = int(rng.choice(targets[b]))
                L = float(np.exp(rng.uniform(*np.log(cfg.aug_width_km))))
                amp = float(np.exp(rng.uniform(*np.log(cfg.aug_amp)))) * float(T["scale"][tg, :2].mean())
                # centre: a random point within R + 300 km of the target
                p0 = T["xyz"][tg]
                rnd = torch.randn(3, device=dev)
                tang = rnd - (rnd @ p0) * p0
                tang = tang / tang.norm()
                ang = float(np.sqrt(rng.uniform()) * (self.radius_km + 300.0) / EARTH_KM)
                ctr = p0 * math.cos(ang) + tang * math.sin(ang)
                dist = torch.arccos((T["xyz"] @ ctr).clamp(-1, 1)) * EARTH_KM            # S
                shape = torch.exp(-dist**2 / (2 * L**2))
                if rng.random() < 0.5:                                                    # fault-like dipole
                    north = torch.tensor([0.0, 0.0, 1.0], device=dev)
                    e_ax = torch.linalg.cross(north, ctr)
                    e_ax = e_ax / e_ax.norm().clamp_min(1e-9)
                    n_ax = torch.linalg.cross(ctr, e_ax)
                    th = rng.uniform(0, np.pi)
                    shape = shape * torch.sign(T["xyz"] @ (e_ax * math.cos(th) + n_ax * math.sin(th)))
                dur = int(rng.integers(cfg.aug_duration[0], cfg.aug_duration[1] + 1))
                start = int(rng.integers(-cfg.aug_duration[1], W))
                t = torch.arange(W, device=dev, dtype=torch.float32) - start
                up = 0.5 - 0.5 * torch.cos(math.pi * (t / dur).clamp(0, 1))
                down = 0.5 - 0.5 * torch.cos(math.pi * ((t - dur - dur // 2) / dur).clamp(0, 1))
                phi = rng.uniform(0, 2 * np.pi)
                u = torch.tensor([math.cos(phi), math.sin(phi)], device=dev)
                s = amp * shape[:, None, None] * (up - down)[None, :, None] * u           # S×W×2 (mm)
                x[b, :, :, :2] += s / T["scale"][:, None, :2] * (a[b][..., None] > 0)
        return x

    # ---- training ------------------------------------------------------
    def _checkpoint(self) -> Path:
        return checkpoint_dir(self.name, MODEL_DIR) / f"R{self.radius_km:g}_seed{self.seed}.pt"

    def _log(self, line: str) -> None:
        print(line, file=sys.stderr, flush=True)
        path = log_dir(self.name, MODEL_DIR) / "train.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(line + "\n")

    def fit(self, cube: Cube, val_hide=None, scorer=None) -> None:
        torch = _torch()
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        self._tables(cube)
        self._net = build_net(self.cfg).to(self._dev)
        path = self._checkpoint()
        if scorer is None:
            if not path.exists():
                raise FileNotFoundError(f"{path}: run `gnssdl bench run {self.name}` first")
            self._net.load_state_dict(torch.load(path, map_location=self._dev))
            self.history = json.loads(path.with_suffix(".json").read_text())["history"]
            return
        cfg, W = self.cfg, self.cfg.window
        # training cells: training days, not excluded, outside earthquake
        # windows, on stations that are not held out (contract §1.5); held-out
        # stations are therefore neither inputs nor targets
        cells = cube.avail & ~cube.exclude & (cube.split_day == 0)[None, :] & (cube.split_sta == 0)[:, None]
        if self.event_window is not None:
            cells &= ~self.event_window
        D = self._inputs(quiet_residuals(cube), cells)
        train_days = np.flatnonzero(cube.split_day == 0)
        starts = np.arange(0, train_days[-1] - W + 2)
        counts = np.lib.stride_tricks.sliding_window_view(cells, W, axis=1).sum(-1)[:, starts]
        starts = starts[(counts >= cfg.min_target_days).any(axis=0)]
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.AdamW(self._net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 *
                                                  (1 + math.cos(math.pi * min(s, cfg.max_steps) / cfg.max_steps)))
        n_par = sum(p.numel() for p in self._net.parameters())
        self._log(f"{self.name} R={self.radius_km:g} seed={self.seed}: {n_par} parameters, {len(cube.sta)} "
                  f"stations, {len(starts)} window starts, p_aug {cfg.p_aug}, device {self._dev} (post-freeze)")
        best, best_state, bad, t_start = np.inf, None, 0, time.time()
        run_loss, run_n = 0.0, 0
        amp = self._dev == "cuda"
        for step in range(1, cfg.max_steps + 1):
            self._net.train()
            t0 = rng.choice(starts, cfg.batch)
            hidden, targets = np.zeros((cfg.batch, len(cube.sta)), dtype=bool), []
            for b, s0 in enumerate(t0):
                tg, hidden[b] = self._draw_hidden(cells[:, s0:s0 + W], rng)
                targets.append(tg)
            n = max(len(t) for t in targets)
            tg_pad = np.stack([np.r_[t, np.full(n - len(t), t[0])] for t in targets])   # pad with a repeat
            weight = np.stack([np.r_[np.ones(len(t)), np.zeros(n - len(t))] for t in targets])
            t0_t = torch.as_tensor(t0, device=self._dev)
            hid_t = torch.as_tensor(hidden, device=self._dev)
            tg_t = torch.as_tensor(tg_pad, device=self._dev)
            x, a = self._feats(D, t0_t, hid_t)
            if cfg.p_aug > 0:
                x = self._augment(x, a, targets, rng)
            feats = torch.cat([x, a[..., None]], -1)
            days = t0_t[:, None] + torch.arange(W, device=self._dev)[None, :]
            y = D["x"][tg_t[:, :, None], days[:, None, :]]                               # B×n×W×3 (labels, unplanted)
            m = D["a"][tg_t[:, :, None], days[:, None, :]] * torch.as_tensor(weight, device=self._dev,
                                                                             dtype=torch.float32)[..., None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = self._net(feats, hid_t, self._T["pair"], tg_t)
            loss = (((pred.float() - y) ** 2) * m[..., None]).sum() / (3 * m.sum()).clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += float(loss.detach())
            run_n += 1
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                score = float(scorer(self.predict(cube, np.zeros(cube.avail.shape, dtype=bool), split=1)))
                improved = score < best - 1e-5
                if improved:
                    best, bad = score, 0
                    best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                else:
                    bad += 1
                self.history.append({"step": step, "train_loss": round(run_loss / run_n, 5),
                                     "val_nrmse_fast": round(score, 5)})
                self._log(f"  step {step:6d}  train loss {run_loss / run_n:.4f}  val nrmse_fast {score:.4f}"
                          f"{'  *' if improved else ''}  ({time.time() - t_start:.0f} s)")
                run_loss, run_n = 0.0, 0
                if bad >= cfg.patience:
                    self._log(f"  early stop: best val {best:.4f}")
                    break
        self._net.load_state_dict(best_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        path.with_suffix(".json").write_text(json.dumps({**self.config(), "history": self.history}, indent=1))

    # ---- inference -----------------------------------------------------
    def predict(self, cube: Cube, hide: np.ndarray, split: int | None = None) -> np.ndarray:
        """Leave-one-out prediction for every station: one pass per group of
        targets and window, with each group's targets and their R-discs
        hidden. With `split`, only that split's days (plus a window of
        context) are predicted, the rest left at zero."""
        torch = _torch()
        if self._net is None:
            raise RuntimeError("fit first")
        cfg, W = self.cfg, self.cfg.window
        S, T = cube.avail.shape
        out = np.zeros((S, T, 3), dtype=np.float32)
        lo, hi = 0, T
        if split is not None:
            d = np.flatnonzero(cube.split_day == split)
            lo, hi = max(0, d[0] - W), min(T, d[-1] + W + 1)
        if hi - lo < W:
            raise ValueError(f"cube shorter than the {W}-day window")
        sub = crop_days(cube, lo, hi)
        D = self._inputs(sub.r, sub.avail & ~hide[:, lo:hi])
        n = hi - lo
        starts = list(range(0, n - W + 1, W // 2))
        if starts[-1] != n - W:
            starts.append(n - W)
        taper = 1.0 - np.abs(np.arange(W) - (W - 1) / 2) / (W / 2) + 1e-3
        tp = torch.as_tensor(taper, dtype=torch.float32, device=self._dev)
        wsum = torch.zeros(n, device=self._dev)
        for s0 in starts:
            wsum[s0:s0 + W] += tp
        groups = hide_groups(self._near, cfg.infer_hide, cfg.group_seed)
        hidden_of = [self._near[g].any(axis=0) for g in groups]
        jobs = [(s0, k) for s0 in starts for k in range(len(groups))]
        acc = torch.zeros(S, n, 3, device=self._dev)
        self._net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._dev == "cuda"):
            for j in range(0, len(jobs), cfg.infer_batch):
                chunk = jobs[j:j + cfg.infer_batch]
                m = max(len(groups[k]) for _, k in chunk)
                tg = np.stack([np.r_[groups[k], np.full(m - len(groups[k]), groups[k][0])] for _, k in chunk])
                t0 = torch.as_tensor([s0 for s0, _ in chunk], device=self._dev)
                hid = torch.as_tensor(np.stack([hidden_of[k] for _, k in chunk]), device=self._dev)
                x, a = self._feats(D, t0, hid)
                tg_t = torch.as_tensor(tg, device=self._dev)
                pred = self._net(torch.cat([x, a[..., None]], -1), hid, self._T["pair"], tg_t).float()
                for b, (s0, k) in enumerate(chunk):
                    g = torch.as_tensor(groups[k], device=self._dev)
                    acc[g, s0:s0 + W] += pred[b, :len(groups[k])] * tp[None, :, None]
        res = acc / wsum[None, :, None] * self._T["scale"][:, None, :]
        out[:, lo:hi] = res.cpu().numpy()
        self.no_neighbour = np.zeros((S, T), dtype=bool)
        return out

    def affected_by(self, cube: Cube, inside: np.ndarray) -> np.ndarray:
        """Every station attends to the whole network beyond R, so a change
        anywhere can reach any station's prediction."""
        return np.ones(len(cube.sta), dtype=bool)

    def config(self) -> dict:
        best = min(self.history, key=lambda h: h["val_nrmse_fast"]) if self.history else None
        return {**super().config(), "seed": self.seed, "post_freeze": True, "m6": asdict(self.cfg),
                "steps": self.history[-1]["step"] if self.history else 0,
                "best_step": best and best["step"], "best_val_nrmse_fast": best and best["val_nrmse_fast"]}


class M6Aug(M6Network):
    """M6 trained with planted fields in the visible stations' inputs,
    labels unchanged."""
    name = "m6aug"
    p_aug = 0.5
