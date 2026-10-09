"""M5: divided space-time attention over a station's neighbourhood
(docs/06-m5-graph-attention.md), and M5-aug, trained with planted
transients in its inputs.

One sample is a target station, its K nearest neighbours beyond the
exclusion radius (chosen per window among the 2K nearest, by availability)
and one summary node, the far stack (mean of every available station beyond
R), over a W-day window. The target is hidden for the whole window
(own-history off), so a prediction never reads the target and one pass over
every station is a leave-one-out cleaning (`never_reads_target`).

Per node and day, 17 input channels (residual, log formal σ, availability,
hidden flag, day-to-day increment, static noise scale and completeness, day
of year), all residuals divided by the station's robust noise scale. A
1-D convolution cuts each node's window into W/P time tokens; blocks then
alternate attention over a node's own time tokens and attention over the
nodes at the same time token, the latter biased by each node's geometry
relative to the target (log distance, azimuth, node type).

Deviations from the design doc, recorded in it: the summary node and K = 32
(the far-stack and neighbour-count results); the output is a correction
added to the far stack (a residual connection), so training starts from the
strongest baseline; and the loss is squared error in units of the station
noise scale (as M3k), not σ-weighted.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from gnssdl.bench.audit import FarStackSweep
from gnssdl.bench.base import Reconstructor, station_scale
from gnssdl.bench.train import quiet_residuals
from gnssdl.dataset import Cube, crop_days

MODEL_DIR = Path("data/models")
EARTH_KM = 6371.0
N_CHANNELS = 17
N_EDGE = 5


@dataclass
class M5Config:
    window: int = 64
    patch: int = 4
    k: int = 32                       # neighbours shown in detail
    candidates: int = 64              # nearest stations beyond R to choose them from
    min_window_avail: float = 0.5
    d_model: int = 64
    heads: int = 4
    blocks: int = 3
    mlp_ratio: int = 2
    dropout: float = 0.1
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 500
    max_steps: int = 15000
    eval_every: int = 500
    patience: int = 6
    neighbour_dropout: float = 0.3    # each sample drops neighbours with p ~ U(0, this)
    min_target_days: int = 16
    # M5-aug
    p_aug: float = 0.0
    aug_width_km: tuple[float, float] = (5.0, 150.0)
    aug_amp: tuple[float, float] = (1.0, 15.0)       # times the target's horizontal noise scale
    aug_duration: tuple[int, int] = (5, 60)
    infer_batch: int = 512


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def build_net(cfg: M5Config):
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

        def forward(self, x, bias=None):              # x: (..., L, d); bias: (..., h, L, L) or broadcastable
            *lead, L, d = x.shape
            q, k, v = self.qkv(x).reshape(*lead, L, 3, self.h, self.dh).unbind(-3)
            q, k, v = (t.transpose(-2, -3) for t in (q, k, v))  # (..., h, L, dh)
            if bias is not None:
                bias = bias.to(q.dtype)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
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

        def forward(self, x, s_bias):                 # x: (B, N, Tt, d)
            x = x + self.t_att(self.n1(x))            # over time, within a node
            x = x + self.m1(self.n2(x))
            xs = x.transpose(1, 2)                    # (B, Tt, N, d)
            xs = xs + self.s_att(self.n3(xs), s_bias)  # over nodes, within a time token
            xs = xs + self.m2(self.n4(xs))
            return xs.transpose(1, 2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d, tt = cfg.d_model, cfg.window // cfg.patch
            self.stem = nn.Conv1d(N_CHANNELS, d, kernel_size=cfg.patch, stride=cfg.patch)
            self.time_pos = nn.Parameter(torch.zeros(tt, d))
            self.geo = nn.Sequential(nn.Linear(N_EDGE, d), nn.GELU(), nn.Linear(d, d))
            self.bias = nn.Sequential(nn.Linear(N_EDGE, 32), nn.GELU(), nn.Linear(32, cfg.heads))
            self.blocks = nn.ModuleList(Block() for _ in range(cfg.blocks))
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, cfg.patch * 3)

        def forward(self, feats, edge, key_ok):
            # feats: (B, N, W, C); edge: (B, N, E); key_ok: (B, N) bool
            B, N, W, C = feats.shape
            x = self.stem(feats.reshape(B * N, W, C).transpose(1, 2)).transpose(1, 2)   # (B*N, Tt, d)
            x = x.reshape(B, N, -1, x.shape[-1]) + self.time_pos + self.geo(edge)[:, :, None, :]
            b = self.bias(edge).permute(0, 2, 1)                                         # (B, h, N): per key
            b = b.masked_fill(~key_ok[:, None, :], float("-inf"))
            s_bias = b[:, None, :, None, :]                                              # (B, 1, h, 1, N)
            for blk in self.blocks:
                x = blk(x, s_bias)
            y = self.head(self.norm(x[:, 0]))                                            # target node: (B, Tt, P*3)
            # residual on the far stack (summary node, channels 0-2): the network
            # learns a correction to the strongest baseline, and starts from it
            return y.reshape(B, W, 3) + feats[:, -1, :, 0:3]

    return Net()


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class M5Attention(Reconstructor):
    name = "m5"
    never_reads_target = True
    supported_contexts = ("same-day", "two-sided")
    p_aug = 0.0

    def __init__(self, radius_km: float = 0.0, seed: int = 0, config: M5Config | None = None, **kw):
        kw.setdefault("context", "two-sided")
        super().__init__(radius_km, **kw)
        self.seed = int(seed)
        self.cfg = config or M5Config(p_aug=self.p_aug)
        self.n_neighbours = self.cfg.candidates          # what `affected` should assume
        self.event_window: np.ndarray | None = None      # set by the harness
        self.history: list[dict] = []
        self._net = None

    # ---- static tables ---------------------------------------------------
    def _tables(self, cube: Cube):
        """Per-station constants fixed at fit time: noise scale, static
        features, unit position vectors, candidate neighbours and the far
        matrix for this radius."""
        torch = _torch()
        dev = self._dev
        scale = station_scale(cube)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nanmedian(scale, axis=0))
        train = cube.split_day == 0
        comp = cube.avail[:, train].mean(axis=1) if train.any() else cube.avail.mean(axis=1)
        q = np.c_[np.log(scale), comp].astype(np.float32)
        lat, lon = np.deg2rad(cube.lat), np.deg2rad(cube.lon)
        xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        ri = cube.radius_index(self.radius_km)
        c = self.cfg.candidates
        idx = cube.nbr_idx[ri][:, :c].astype(np.int64)
        dist = cube.nbr_dist[ri][:, :c].astype(np.float32)
        az = cube.nbr_az[ri][:, :c].astype(np.float32)
        fs = FarStackSweep(radius_km=self.radius_km)
        far = fs._far_matrix(cube)
        self._scale_np = scale
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
        self._T = {"scale": t(scale), "q": t(q), "xyz": t(xyz), "idx": t(idx, torch.long),
                   "dist": t(dist), "az": t(az), "far": t(far)}

    def _inputs(self, cube: Cube, r: np.ndarray, avail: np.ndarray):
        """Day-indexed tensors for a cube: normalised residuals and log σ,
        availability, day of year, and the normalised far stack."""
        torch = _torch()
        dev = self._dev
        s = self._scale_np[:, None, :]
        x = np.where(avail[..., None], np.nan_to_num(r, nan=0.0) / s, 0.0).astype(np.float32)
        sig = np.where(avail[..., None], np.log(np.maximum(np.nan_to_num(cube.sigma, nan=1.0), 1e-3) / s), 0.0)
        doy = np.asarray([d.timetuple().tm_yday for d in cube.days.astype("datetime64[D]").astype(object)])
        ang = 2 * np.pi * doy / 365.25
        far = self._T["far"]
        a_t = torch.as_tensor(avail, dtype=torch.float32, device=dev)
        r_t = torch.as_tensor(np.where(avail[..., None], np.nan_to_num(r, nan=0.0), 0.0), dtype=torch.float32, device=dev)
        den = far @ a_t                                                             # S×T
        fs = torch.stack([far @ r_t[..., c] for c in range(3)], -1) / den.clamp_min(1)[..., None]
        fs = fs / self._T["scale"][:, None, :]
        return {"x": torch.as_tensor(x, device=dev), "sig": torch.as_tensor(sig.astype(np.float32), device=dev),
                "a": a_t, "doy": torch.as_tensor(np.c_[np.sin(ang), np.cos(ang)].astype(np.float32), device=dev),
                "fs": fs, "fs_ok": (den > 0).float(), "den": den}

    # ---- one batch of samples -------------------------------------------
    def _batch(self, D, targets, t0, rng=None, truth=None):
        """Features for (target, window start) pairs. `rng` (training) turns
        on neighbour dropout and augmentation."""
        torch = _torch()
        T, cfg, dev = self._T, self.cfg, self._dev
        B, W, K = len(targets), cfg.window, cfg.k
        days = t0[:, None] + torch.arange(W, device=dev)[None, :]                     # B×W
        cand = T["idx"][targets]                                                     # B×C
        cvalid = cand >= 0
        cand_c = cand.clamp_min(0)
        a_c = D["a"][cand_c[:, :, None], days[:, None, :]] * cvalid[..., None]         # B×C×W
        ok = a_c.mean(-1) >= cfg.min_window_avail
        rank = torch.arange(cand.shape[1], device=dev)[None, :].float()
        order = torch.argsort(torch.where(ok, rank, rank + 1e4), dim=1)[:, :K]
        nb = torch.gather(cand_c, 1, order)                                          # B×K
        nb_ok = torch.gather(ok, 1, order)
        nb_dist = torch.gather(T["dist"][targets], 1, order)
        nb_az = torch.gather(T["az"][targets], 1, order)
        g = (nb[:, :, None], days[:, None, :])
        x_nb, a_nb, s_nb = D["x"][g], D["a"][g] * nb_ok[..., None], D["sig"][g]
        hidden_nb = torch.zeros_like(a_nb)
        if rng is not None:                                                          # neighbour dropout
            p = torch.as_tensor(rng.uniform(0, cfg.neighbour_dropout, B), device=dev, dtype=torch.float32)
            drop = torch.rand(B, K, device=dev) < p[:, None]
            hidden_nb = (drop[..., None] & (a_nb > 0)).float()
            a_nb = a_nb * (~drop)[..., None]
        fs, fs_ok = D["fs"][targets[:, None], days], D["fs_ok"][targets[:, None], days]   # B×W×3, B×W
        if rng is not None and cfg.p_aug > 0:
            x_nb, fs = self._augment(D, targets, nb, days, x_nb, a_nb, fs, rng)
        x_nb = x_nb * a_nb[..., None]
        s_nb = s_nb * a_nb[..., None]
        # node features: target, K neighbours, far-stack summary
        def incr(x, a):
            dx = torch.zeros_like(x)
            both = a[..., 1:] * a[..., :-1]
            dx[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) * both[..., None]
            return dx
        doy = D["doy"][days]                                                         # B×W×2
        q_t, q_nb = T["q"][targets], T["q"][nb]
        tgt = torch.zeros(B, 1, W, N_CHANNELS, device=dev)
        tgt[..., 7] = 1.0                                                            # hidden flag
        tgt[..., 11:15] = q_t[:, None, None, :]
        nbf = torch.cat([x_nb, s_nb, a_nb[..., None], hidden_nb[..., None], incr(x_nb, a_nb),
                         q_nb[:, :, None, :].expand(B, K, W, 4)], -1)
        nbf = torch.cat([nbf, torch.zeros(B, K, W, 2, device=dev)], -1)
        fsn = torch.zeros(B, 1, W, N_CHANNELS, device=dev)
        fsn[:, 0, :, 0:3] = fs * fs_ok[..., None]
        fsn[:, 0, :, 6] = fs_ok
        fsn[:, 0, :, 8:11] = incr(fs[:, None] * fs_ok[:, None, :, None], fs_ok[:, None])[:, 0]
        feats = torch.cat([tgt, nbf, fsn], 1)
        feats[..., 15:17] = doy[:, None]
        edge = torch.zeros(B, K + 2, N_EDGE, device=dev)
        edge[:, 0, 3] = 1.0
        edge[:, 1:K + 1, 0] = torch.log(nb_dist.clamp_min(1.0)) / 5.0
        edge[:, 1:K + 1, 1] = torch.sin(nb_az)
        edge[:, 1:K + 1, 2] = torch.cos(nb_az)
        edge[:, K + 1, 4] = 1.0
        key_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), a_nb.sum(-1) > 0,
                            (fs_ok.sum(-1) > 0)[:, None]], 1)
        return feats, edge, key_ok

    def _augment(self, D, targets, nb, days, x_nb, a_nb, fs, rng):
        """Add one planted field (Gaussian blob or fault-like dipole) to the
        neighbours' inputs and to the far stack, for a p_aug share of the
        samples; labels are left unchanged."""
        torch = _torch()
        T, cfg, dev = self._T, self.cfg, self._dev
        B, W = days.shape
        use = torch.as_tensor(rng.random(B) < cfg.p_aug, device=dev)
        if not use.any():
            return x_nb, fs
        lo, hi = cfg.aug_width_km
        L = torch.as_tensor(np.exp(rng.uniform(np.log(lo), np.log(hi), B)), device=dev, dtype=torch.float32)
        amp = torch.as_tensor(np.exp(rng.uniform(*np.log(cfg.aug_amp), B)), device=dev, dtype=torch.float32)
        amp_mm = amp * T["scale"][targets, :2].mean(1)
        # centre: a random point within R + 300 km of the target
        reach = (self.radius_km + 300.0) / EARTH_KM
        p0 = T["xyz"][targets]                                                        # B×3
        rnd = torch.randn(B, 3, device=dev)
        tang = rnd - (rnd * p0).sum(1, keepdim=True) * p0
        tang = tang / tang.norm(dim=1, keepdim=True)
        ang = torch.as_tensor(np.sqrt(rng.uniform(0, 1, B)) * reach, device=dev, dtype=torch.float32)
        ctr = p0 * torch.cos(ang)[:, None] + tang * torch.sin(ang)[:, None]
        dist = torch.arccos((T["xyz"] @ ctr.T).clamp(-1, 1)).T * EARTH_KM            # B×S
        shape = torch.exp(-dist**2 / (2 * L[:, None] ** 2))
        dipole = torch.as_tensor(rng.random(B) < 0.5, device=dev)
        north = torch.tensor([0.0, 0.0, 1.0], device=dev)
        e_ax = torch.cross(north.expand(B, 3), ctr, dim=1)
        e_ax = e_ax / e_ax.norm(dim=1, keepdim=True).clamp_min(1e-9)
        n_ax = torch.cross(ctr, e_ax, dim=1)
        th = torch.as_tensor(rng.uniform(0, np.pi, B), device=dev, dtype=torch.float32)
        line_n = e_ax * torch.cos(th)[:, None] + n_ax * torch.sin(th)[:, None]        # normal to the dividing line
        side = torch.sign(T["xyz"] @ line_n.T).T                                     # B×S
        shape = torch.where(dipole[:, None], shape * side, shape)
        dur = torch.as_tensor(rng.integers(cfg.aug_duration[0], cfg.aug_duration[1] + 1, B), device=dev)
        start = torch.as_tensor(rng.integers(-cfg.aug_duration[1], W, B), device=dev)
        t = torch.arange(W, device=dev)[None, :] - start[:, None]                    # B×W
        hold = dur // 2
        up = 0.5 - 0.5 * torch.cos(math.pi * (t / dur[:, None]).clamp(0, 1))
        down = 0.5 - 0.5 * torch.cos(math.pi * ((t - dur[:, None] - hold[:, None]) / dur[:, None]).clamp(0, 1))
        prof = up - down                                                             # rise, hold, fall
        phi = torch.as_tensor(rng.uniform(0, 2 * np.pi, B), device=dev, dtype=torch.float32)
        u = torch.stack([torch.cos(phi), torch.sin(phi)], 1)                         # east, north unit vector
        s_all = (amp_mm * use)[:, None, None, None] * shape[:, :, None, None] * prof[:, None, :, None] \
            * u[:, None, None, :]                                                    # B×S×W×2 (mm)
        s_nb = torch.gather(s_all, 1, nb[:, :, None, None].expand(-1, -1, W, 2))
        x_nb = x_nb.clone()
        x_nb[..., :2] = x_nb[..., :2] + s_nb / T["scale"][nb][:, :, None, :2] * (a_nb[..., None] > 0)
        # the far stack is a mean over available far stations; add the field's mean over them
        wts = T["far"][targets][:, :, None] * D["a"][:, days].permute(1, 0, 2)       # B×S×W
        s_far = (s_all * wts[..., None]).sum(1) / wts.sum(1).clamp_min(1)[..., None]  # B×W×2
        fs = fs.clone()
        fs[..., :2] = fs[..., :2] + s_far / T["scale"][targets][:, None, :2]
        return x_nb, fs

    # ---- training ------------------------------------------------------
    def _checkpoint(self) -> Path:
        return MODEL_DIR / self.name / f"R{self.radius_km:g}_seed{self.seed}.pt"

    def _log(self, line: str) -> None:
        print(line, file=sys.stderr, flush=True)
        path = MODEL_DIR / self.name / "train.log"
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
            meta = json.loads(path.with_suffix(".json").read_text())
            self.history = meta["history"]
            return
        cfg = self.cfg
        cells = cube.avail & ~cube.exclude & (cube.split_day == 0)[None, :] & (cube.split_sta == 0)[:, None]
        if self.event_window is not None:
            cells &= ~self.event_window
        D = self._inputs(cube, quiet_residuals(cube), cells)
        train_days = np.flatnonzero(cube.split_day == 0)
        last_start = train_days[-1] - cfg.window + 1
        counts = np.lib.stride_tricks.sliding_window_view(cells, cfg.window, axis=1).sum(-1)  # S×(T-W+1)
        good = counts[:, : last_start + 1] >= cfg.min_target_days
        pairs = np.argwhere(good)                                                    # (target, t0)
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.AdamW(self._net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 *
                                                  (1 + math.cos(math.pi * min(s, cfg.max_steps) / cfg.max_steps)))
        n_par = sum(p.numel() for p in self._net.parameters())
        self._log(f"{self.name} R={self.radius_km:g} seed={self.seed}: {n_par} parameters, {len(pairs)} "
                  f"(target, window) pairs, p_aug {cfg.p_aug}, device {self._dev}")
        best, best_state, bad, t0 = np.inf, None, 0, time.time()
        run_loss, run_n = 0.0, 0
        amp = self._dev == "cuda"
        for step in range(1, cfg.max_steps + 1):
            self._net.train()
            pick = pairs[rng.integers(0, len(pairs), cfg.batch)]
            tg = torch.as_tensor(pick[:, 0], device=self._dev)
            st = torch.as_tensor(pick[:, 1], device=self._dev)
            feats, edge, key_ok = self._batch(D, tg, st, rng=rng)
            days = st[:, None] + torch.arange(cfg.window, device=self._dev)[None, :]
            y = D["x"][tg[:, None], days]
            m = D["a"][tg[:, None], days]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = self._net(feats, edge, key_ok)
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
                          f"{'  *' if improved else ''}  ({time.time() - t0:.0f} s)")
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
        """Leave-one-out prediction for every station. With `split`, only
        the days of that split (plus a window of context) are predicted, the
        rest left at zero (used for validation during training)."""
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
        sub = crop_days(cube, lo, hi)
        if hi - lo < W:
            raise ValueError(f"cube shorter than the {W}-day window")
        avail = sub.avail & ~hide[:, lo:hi]
        D = self._inputs(sub, sub.r, avail)
        n = hi - lo
        starts = list(range(0, n - W + 1, W // 2))
        if starts[-1] != n - W:
            starts.append(n - W)
        taper = 1.0 - np.abs(np.arange(W) - (W - 1) / 2) / (W / 2) + 1e-3
        acc = torch.zeros(S, n, 3, device=self._dev)
        wsum = torch.zeros(n, device=self._dev)
        tp = torch.as_tensor(taper, dtype=torch.float32, device=self._dev)
        for s0 in starts:
            wsum[s0:s0 + W] += tp
        pairs = [(i, s0) for s0 in starts for i in range(S)]
        self._net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._dev == "cuda"):
            for b in range(0, len(pairs), cfg.infer_batch):
                chunk = np.asarray(pairs[b:b + cfg.infer_batch])
                tg = torch.as_tensor(chunk[:, 0], device=self._dev)
                st = torch.as_tensor(chunk[:, 1], device=self._dev)
                pred = self._net(*self._batch(D, tg, st)).float() * tp[None, :, None]
                days = st[:, None] + torch.arange(W, device=self._dev)[None, :]
                acc.index_put_((tg[:, None].expand(-1, W), days), pred, accumulate=True)
        res = (acc / wsum[None, :, None]) * self._T["scale"][:, None, :]
        out[:, lo:hi] = res.cpu().numpy()
        return out

    def affected_by(self, cube: Cube, inside: np.ndarray) -> np.ndarray:
        """Stations whose prediction a change inside `inside` can reach:
        through a candidate neighbour, or through the far stack."""
        from gnssdl.dataset import distance_azimuth
        idx = cube.nbr_idx[cube.radius_index(self.radius_km)][:, :self.cfg.candidates]
        via_nb = ((idx >= 0) & inside[np.where(idx >= 0, idx, 0)]).any(axis=1)
        d, _ = distance_azimuth(cube.lat, cube.lon)
        far = d >= self.radius_km
        np.fill_diagonal(far, False)
        return inside | via_nb | far[:, inside].any(axis=1)

    def config(self) -> dict:
        best = min(self.history, key=lambda h: h["val_nrmse_fast"]) if self.history else None
        return {**super().config(), "seed": self.seed, "m5": asdict(self.cfg),
                "steps": self.history[-1]["step"] if self.history else 0,
                "best_step": best and best["step"], "best_val_nrmse_fast": best and best["val_nrmse_fast"]}


class M5Aug(M5Attention):
    """M5 trained with planted transients in the neighbours' inputs and the
    far stack, labels unchanged (docs/06, M5-aug)."""
    name = "m5aug"
    p_aug = 0.5
