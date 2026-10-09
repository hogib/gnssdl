"""M3k: learned blind-spot kernel (docs/07-m3k-learned-kernel.md).

    r̂_ic(t) = Σ_j w_ijc a_j(t) r_jc(t) / Σ_j w_ijc a_j(t),   w_ijc = softplus(f_θ(g_ij, q_j))_c

over the neighbours beyond the exclusion radius, with f_θ a small MLP of the
neighbour's position relative to the target (log distance, azimuth) and its
quality (log noise scale, record completeness). The weights never depend on
the day's values, so once trained the model is M1 with a learned kernel and
predicts with the same station × station matrix product. The target never
enters its own sum: one pass is a leave-one-out prediction for every station.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from gnssdl.bench.base import Reconstructor, station_scale
from gnssdl.bench.train import quiet_residuals
from gnssdl.dataset import Cube

MODEL_DIR = Path("data/models")
HIDDEN = 32
BATCH_DAYS = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
PATIENCE = 6


def _torch():
    import torch
    return torch


def static_features(cube: Cube) -> np.ndarray:
    """S×4 neighbour quality: log robust noise scale (E, N, U) and the
    fraction of training days the station has data."""
    scale = station_scale(cube)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nanmedian(scale, axis=0))
    train = cube.split_day == 0
    completeness = cube.avail[:, train].mean(axis=1)
    return np.c_[np.log(scale), completeness].astype(np.float32)


def edge_features(cube: Cube, ri: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """S×K×3 relative geometry (log distance, sin and cos azimuth) and the
    S×K neighbour indices (−1 pads) for radius index `ri`."""
    idx = cube.nbr_idx[ri][:, :k]
    d = np.maximum(cube.nbr_dist[ri][:, :k].astype(np.float64), 1.0)
    az = cube.nbr_az[ri][:, :k].astype(np.float64)
    g = np.stack([np.log(d), np.sin(az), np.cos(az)], axis=-1)
    return np.where((idx >= 0)[..., None], g, 0.0).astype(np.float32), idx


class M3Kernel(Reconstructor):
    """Normalized convolution with a learned, continuous kernel."""

    name = "m3k"
    never_reads_target = True
    n_neighbours = 16

    def __init__(self, radius_km: float = 0.0, seed: int = 0, **kw):
        super().__init__(radius_km, **kw)
        self.seed = int(seed)
        self.event_window: np.ndarray | None = None      # set by the harness
        self.history: list[dict] = []
        self._w: np.ndarray | None = None                # S×K×3 learned weights

    # ---- network -------------------------------------------------------
    def _net(self):
        torch = _torch()
        torch.manual_seed(self.seed)
        nn = torch.nn
        return nn.Sequential(nn.Linear(7, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                             nn.Linear(HIDDEN, 3))

    def _weights(self, net, g, q_nb, valid):
        """softplus kernel weights, zero for padded slots."""
        torch = _torch()
        w = torch.nn.functional.softplus(net(torch.cat([g, q_nb], dim=-1)))
        return w * valid[..., None]

    def _checkpoint(self) -> Path:
        return MODEL_DIR / self.name / f"R{self.radius_km:g}_seed{self.seed}.pt"

    # ---- fit -----------------------------------------------------------
    def fit(self, cube: Cube, val_hide=None, scorer=None) -> None:
        torch = _torch()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ri = cube.radius_index(self.radius_km)
        g_np, idx = edge_features(cube, ri, self.n_neighbours)
        q_np = static_features(cube)
        valid_np = idx >= 0
        nb_np = np.where(valid_np, idx, 0)
        net = self._net().to(dev)
        g = torch.from_numpy(g_np).to(dev)
        q_nb = torch.from_numpy(q_np[nb_np]).to(dev)
        valid = torch.from_numpy(valid_np.astype(np.float32)).to(dev)

        path = self._checkpoint()
        if scorer is None:
            # rebuilding a trained model (e.g. for the signal tests): load it
            if not path.exists():
                raise FileNotFoundError(f"{path}: run `gnssdl bench run {self.name}` first")
            net.load_state_dict(torch.load(path, map_location=dev))
            self.history = json.loads(path.with_suffix(".json").read_text())["history"]
            with torch.no_grad():
                self._w = self._weights(net, g, q_nb, valid).cpu().numpy().astype(np.float64)
            return

        # training data: quiet residuals on training cells of seen stations,
        # used both as inputs and as targets (contract §1.5)
        scale = station_scale(cube)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        cells = cube.avail & ~cube.exclude & (cube.split_day == 0)[None, :] & (cube.split_sta == 0)[:, None]
        if self.event_window is not None:
            cells &= ~self.event_window
        x_np = np.where(cells[..., None], np.nan_to_num(quiet_residuals(cube), nan=0.0), 0.0)
        days = np.flatnonzero(cells.any(axis=0))
        x_all = torch.from_numpy(x_np.astype(np.float32)).to(dev)              # S×T×3
        a_all = torch.from_numpy(cells.astype(np.float32)).to(dev)              # S×T
        inv_s = torch.from_numpy((1.0 / scale).astype(np.float32)).to(dev)      # S×3
        nb = torch.from_numpy(nb_np).to(dev)

        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(len(days) / BATCH_DAYS)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS * steps_per_epoch)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = np.inf, None, 0
        log = self._log_file()
        self._log(log, f"{self.name} R={self.radius_km:g} seed={self.seed}: {len(days)} training days, "
                       f"{steps_per_epoch} steps/epoch, device {dev}")
        for epoch in range(MAX_EPOCHS):
            t0 = time.time()
            net.train()
            total, n = 0.0, 0
            for b in np.array_split(rng.permutation(days), steps_per_epoch):
                bt = torch.from_numpy(b).to(dev)
                x = x_all[:, bt]                                       # S×B×3
                a = a_all[:, bt]                                       # S×B
                w = self._weights(net, g, q_nb, valid)                 # S×K×3
                ww = w[:, :, None, :] * a[nb][..., None]               # S×K×B×3
                num = (ww * x[nb]).sum(1)
                den = ww.sum(1)
                ok = (den > 0) & (a[..., None] > 0)                    # target has truth and a neighbour
                pred = num / den.clamp_min(1e-12)
                err = ((x - pred) * inv_s[:, None, :]) ** 2
                loss = err[ok].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * int(ok.sum())
                n += int(ok.sum())
            net.eval()
            with torch.no_grad():
                self._w = self._weights(net, g, q_nb, valid).cpu().numpy().astype(np.float64)
            score = float(scorer(self.predict(cube, np.zeros(cube.avail.shape, dtype=bool))))
            self.history.append({"epoch": epoch + 1, "train_loss": round(total / max(n, 1), 5),
                                 "val_nrmse_fast": round(score, 5)})
            improved = score < best - 1e-5
            if improved:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            self._log(log, f"  epoch {epoch + 1:3d}  train loss {total / max(n, 1):.4f}  "
                           f"val nrmse_fast {score:.4f}{'  *' if improved else ''}  ({time.time() - t0:.1f} s)")
            if bad >= PATIENCE:
                self._log(log, f"  early stop: best val {best:.4f}")
                break
        net.load_state_dict(best_state)
        with torch.no_grad():
            self._w = self._weights(net, g, q_nb, valid).cpu().numpy().astype(np.float64)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path)
        path.with_suffix(".json").write_text(json.dumps({**self.config(), "history": self.history}, indent=1))

    def _log_file(self) -> Path:
        path = MODEL_DIR / self.name / "train.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _log(path: Path, line: str) -> None:
        print(line, file=sys.stderr, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    # ---- predict -------------------------------------------------------
    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("fit first")
        ri = cube.radius_index(self.radius_km)
        idx = cube.nbr_idx[ri][:, :self.n_neighbours]
        nb = np.where(idx >= 0, idx, 0)
        S, T, _ = cube.r.shape
        rows = np.repeat(np.arange(S), idx.shape[1])
        a = cube.avail.astype(np.float64)
        out = np.zeros((S, T, 3), dtype=np.float32)
        for c in range(3):
            W = np.zeros((S, S))
            np.add.at(W, (rows, nb.ravel()), self._w[:, :, c].ravel())
            x = np.where(cube.avail, np.nan_to_num(cube.r[..., c], nan=0.0), 0.0).astype(np.float64)
            num, den = W @ x, W @ a
            out[..., c] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
            if c == 0:
                self.no_neighbour = den <= 0
        return out

    def config(self) -> dict:
        best = min(self.history, key=lambda h: h["val_nrmse_fast"]) if self.history else None
        return {**super().config(), "seed": self.seed, "n_neighbours": self.n_neighbours,
                "epochs": len(self.history), "best_epoch": best and best["epoch"],
                "best_val_nrmse_fast": best and best["val_nrmse_fast"]}

    def kernel(self) -> np.ndarray:
        """The learned S×K×3 weights (for plotting the kernel)."""
        return self._w


class M3Kernel256(M3Kernel):
    """M3k over the 256 nearest neighbours beyond R (ablation 3, widened
    after the far-stack results)."""
    name = "m3k256"
    n_neighbours = 256
