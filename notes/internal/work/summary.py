"""Validation-year numbers for the internal report, Pacific Northwest cube (read-only).

Run from the repository root on the machine that holds the results:
    uv run python notes/internal/work/summary.py [overnight.log] > notes/internal/work/summary.txt
Reads data/runs/*/scores and data/runs/*/signal/validation only; never the test split.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from gnssdl.bench import signal as bsig  # noqa: E402
from gnssdl.dataset import Cube  # noqa: E402

ROOT = Path("data/runs")
CUBE = Path("data/cube/pnw/pnw.npz")
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 1000)

REF = ["m0", "c1", "c3-k1p1", "c3-k1p2", "c3-k1p3", "c3-k4p1", "c3-k4p2", "c3-k4p3", "c3-k8p1", "c3-k8p2",
       "c3-k8p3", "m2self", "fs", "d1", "m1", "m1k64", "m1k256", "m2", "m3a", "m3b", "m4a", "m4b"]
LEARNED = ["m3k", "m3k256", "m5", "m5aug", "m6r16", "m6r16aug", "m6r64", "m6r64aug", "m6r256", "m6r256aug",
           "m6", "m6aug", "m6r", "m6raug"]


def split_label(lbl):
    m = re.match(r"^(.*)-s(\d+)$", lbl)
    return (m.group(1), int(m.group(2))) if m else (lbl, 0)


def load(kind):
    frames = [f for f in (pd.read_csv(p) for p in sorted(ROOT.glob(f"*/signal/validation/{kind}.csv"))) if len(f)]
    df = pd.concat(frames, ignore_index=True)
    df[["base", "seed"]] = df.model.apply(lambda s: pd.Series(split_label(s)))
    df["R"] = df.radius_km.fillna(-1.0)
    return df


def section(title):
    print(f"\n==== {title}")


# ---------------------------------------------------------------- the cube
cube = Cube.load(CUBE)
section("cube")
days = pd.DatetimeIndex(cube.days)
S, T = cube.avail.shape
val = cube.split_day == 1
print(f"stations {S}, held out {(cube.split_sta == 1).sum()}, days {T} ({days[0].date()} to {days[-1].date()})")
for code, name in [(0, "train"), (1, "validation"), (2, "test")]:
    d = days[cube.split_day == code]
    print(f"  {name}: {len(d)} days, {d[0].date()} to {d[-1].date()}")
print(f"coverage {cube.avail.mean():.3f}; validation coverage {cube.avail[:, val].mean():.3f}")
print(f"stations with data in validation: {(cube.avail[:, val].sum(axis=1) > 0).sum()}")
train = cube.split_day == 0
print(f"training cells barred (exclude & avail / avail, training days): "
      f"{(cube.exclude[:, train] & cube.avail[:, train]).sum() / cube.avail[:, train].sum():.3f}; "
      f"stations touched {(cube.exclude & cube.avail).any(axis=1).sum()}")

# ---------------------------------------------------------------- mask scores
sc = pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob("*/scores/*.csv"))], ignore_index=True)
sc = sc[sc.split == "validation"].copy()
sc[["base", "seed"]] = sc.model.apply(lambda s: pd.Series(split_label(s)))
sc["R"] = sc.radius_km.fillna(-1.0)
section("mask scores: scored stations per pattern (m0)")
print(sc[sc.model == "m0"][["pattern", "stations", "cells"]].to_string(index=False))
fast = sc.pivot_table(index=["base", "seed", "R"], columns="pattern", values="nrmse_fast")
section("mask scores, fast part, every model, seed and radius")
print(fast.round(3).to_string())
for part in ("slow", "total"):
    section(f"mask scores, {part} part, R = 0 or radius-free, seed 0")
    t = sc[(sc.seed == 0) & (sc.R <= 0)].pivot_table(index="base", columns="pattern", values=f"nrmse_{part}")
    print(t.round(3).to_string())
section("mask scores, fast part, per component, scatter, R = 0, seed 0")
t = sc[(sc.seed == 0) & (sc.R <= 0) & (sc.pattern == "scatter")].set_index("base")
print(t[["nrmse_fast_e", "nrmse_fast_n", "nrmse_fast_u"]].round(3).to_string())

# ---------------------------------------------------------------- signal tests
inj, sea, off, eps, noi = (load(k) for k in ("injections", "seasonal", "offsets", "episodes", "noise"))

section("planted transients: count per model and radius (seed 0)")
print(inj[inj.seed == 0].groupby(["base", "R"]).size().unstack().to_string())
z = inj[(inj.model == "fs") & (inj.R == 0)]
print(f"fs R=0: {len(z)} transients, median stations within 2L {z.stations.median():g}, "
      f"median stations that can carry spurious signal {z.spurious_stations.median():g}")
print("median stations within 2L by L:", z.groupby("footprint_km").stations.median().to_dict())


def per_point(df, seeds=(0,)):
    d = df[df.seed.isin(seeds)]
    rows = {}
    for (b, s, r), g in d.groupby(["base", "seed", "R"]):
        row = {"kept": g.rho.median(), "spurious": g.spurious_max.median(),
               "spur_sta": g.spurious_stations.median(), "spur_med": g.spurious_median.median()}
        for L, gg in g.groupby("footprint_km"):
            row[f"L{L:g}"] = gg.rho.median()
            row[f"sp{L:g}"] = gg.spurious_max.median()
        rows[(b, s, r)] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.names = ["base", "seed", "R"]
    return out


pts = per_point(inj, (0, 1, 2))
section("planted transients: kept (median rho) and spurious by width, every model, seed, radius")
print(pts.round(3).to_string())
section("kept by amplitude (mm), R = 0, seed 0")
z = inj[(inj.seed == 0) & (inj.R <= 0)]
print(z.pivot_table(index="base", columns="amplitude_mm", values="rho", aggfunc="median").round(2).to_string())
section("kept by duration (days), R = 0, seed 0")
print(z.pivot_table(index="base", columns="duration_days", values="rho", aggfunc="median").round(2).to_string())
section("kept by width and duration, L = 25 and 100, R = 0, seed 0")
zz = z[z.footprint_km.isin([25, 100])]
print(zz.pivot_table(index="base", columns=["footprint_km", "duration_days"], values="rho",
                     aggfunc="median").round(2).to_string())

section("seasonal kept: median over stations, every model, seed, radius")
t = sea.groupby(["base", "seed", "R"])[["rho_e", "rho_n", "rho_u", "rho_h", "rho_all"]].median()
t["stations"] = sea.groupby(["base", "seed", "R"]).size()
print(t.round(3).to_string())
z = sea[sea.model == "m0"]
print("m0 median seasonal amplitudes (mm):",
      z[[c for c in z if c.endswith("_mm")]].median().round(2).to_dict())
z = sea[(sea.model == "fs") & (sea.R == 0)]
print("fs R=0 seasonal rho_all quartiles:", z.rho_all.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict())

section("real earthquake offsets: every row of seed 0 at R = 0 or radius-free")
print(off[(off.seed == 0) & (off.R <= 0)][["model", "sta", "date", "event", "mag", "offset_mm", "kept"]]
      .to_string(index=False))
section("real earthquake offsets kept by model, seed, radius")
print(off.groupby(["base", "seed", "R"]).kept.median().unstack().round(3).to_string())

section("slow-slip episodes: the validation list (fs, R = 0)")
z = eps[(eps.model == "fs") & (eps.R == 0)]
print(bsig.episode_table(z).merge(z.groupby("episode").obs_mm.median().reset_index(), on="episode")
      .rename(columns={"retained": "kept"}).round(3).to_string(index=False))
section("slow-slip episodes kept: episode_summary per model, seed, radius")
rows = []
for (b, s, r), g in eps.groupby(["base", "seed", "R"]):
    for x in bsig.episode_summary(g).itertuples(index=False):
        rows.append({"base": b, "seed": s, "R": r, "timing": x.timing, "episodes": x.episodes,
                     "stations": x.stations, "kept": x.retained, "weak": x.weak})
print(pd.DataFrame(rows).set_index(["base", "seed", "R", "timing"]).round(3).to_string())
section("slow-slip episodes kept per episode, R = 0, seed 0 (pooled over stations)")
z = eps[(eps.seed == 0) & (eps.R <= 0)]
z = z.groupby(["base", "episode"])[["num", "den"]].sum()
print((z.num / z.den).unstack().round(2).to_string())

section("noise removed, every model, seed, radius")
print(noi.pivot_table(index=["base", "seed"], columns="R", values="noise_removed").round(3).to_string())
print("stations:", noi.stations.unique())

# ---------------------------------------------------------------- seed spread
section("seed spread at R = 0: per seed, mean, and max - min")
cols = {}
for (b, s), g in fast.xs(0.0, level="R").groupby(level=["base", "seed"]):
    cols.setdefault((b, s), {}).update(g.iloc[0].to_dict())
for (b, s, r), row in pts.iterrows():
    if r == 0:
        cols.setdefault((b, s), {}).update({k: row[k] for k in ("kept", "L25", "L100", "spurious")})
for (b, s), g in sea[sea.R == 0].groupby(["base", "seed"]):
    cols[(b, s)]["seasonal"] = g.rho_all.median()
for (b, s), g in eps[eps.R == 0].groupby(["base", "seed"]):
    cols[(b, s)]["episodes"] = bsig.episode_summary(g).retained.iloc[0]
for (b, s), g in noi[noi.R == 0].groupby(["base", "seed"]):
    cols[(b, s)]["noise"] = g.noise_removed.iloc[0]
seeds = pd.DataFrame.from_dict(cols, orient="index")
seeds.index.names = ["base", "seed"]
seeds = seeds.loc[[b for b in LEARNED if b in seeds.index.get_level_values(0)]]
order = ["scatter", "block", "station", "noise", "kept", "L25", "L100", "spurious", "seasonal", "episodes"]
seeds = seeds[order]
print(seeds.round(3).to_string())
g = seeds.groupby(level="base", sort=False)
print("\nmean over seeds")
print(g.mean().round(3).to_string())
print("\nmax - min over seeds")
print((g.max() - g.min()).round(3).to_string())

# ---------------------------------------------------------------- training record
if len(sys.argv) > 1:
    section("training record from the queue log: best step and validation criterion")
    pat = re.compile(r"^end train_(\S+)_s(\d)_R(\d+) exit (\d+) \S+ \S+ done \(best (step|epoch) (\d+)/(\d+), "
                     r"val ([\d.]+), (.*)\)")
    rec = []
    for line in Path(sys.argv[1]).read_text().splitlines():
        m = pat.match(line)
        if m:
            rec.append({"model": m[1], "seed": int(m[2]), "R": int(m[3]), "exit": int(m[4]), "unit": m[5],
                        "best": int(m[6]), "of": int(m[7]), "val": float(m[8]), "checks": m[9]})
    rec = pd.DataFrame(rec)
    print(rec.pivot_table(index=["model", "seed"], columns="R", values="best").to_string())
    print(rec.pivot_table(index=["model", "seed"], columns="R", values="val").round(4).to_string())
    print("leak checks:", rec.checks.value_counts().to_dict(), " exits:", rec.exit.value_counts().to_dict())
    early = rec[rec.best <= 1000][["model", "seed", "R", "unit", "best", "of"]]
    print("stopped at the first checks (best <= 1000 steps or epoch 1):")
    print(early.to_string(index=False))

# ---------------------------------------------------------------- criterion, intervals, neighbours (2026-11-12)
for name, title in (("verdict_validation.csv", "success criterion (bench verdict), sorted by noise removed"),
                    ("neighbours.csv", "neighbours actually read (bench neighbours)")):
    path = ROOT / name
    if path.exists():
        section(title)
        t = pd.read_csv(path)
        if "noise_removed" in t:
            t = t.sort_values("noise_removed", ascending=False)
            print(t.verdict.value_counts().to_dict(), "| best:", t[t.best].model.tolist(),
                  "| same as best:", t[t.same_as_best].model.tolist())
        print(t.round(3).to_string(index=False))
path = ROOT / "intervals_validation.csv"
if path.exists():
    section("95% bootstrap intervals (bench intervals), R = 0 or radius-free, seed 0; kept and spurious at 100 km")
    t = pd.read_csv(path)
    t = t[(t.radius_km.fillna(0) == 0) & ~t.model.str.contains(r"-s\d+$") & (t.footprint_km.isna() | (t.footprint_km == 100))]
    t["cell"] = [f"{v:.3f} [{a:.3f}, {b:.3f}]" for v, a, b in zip(t.value, t.ci_low, t.ci_high)]
    print(t.pivot_table(index="model", columns="measure", values="cell", aggfunc="first")
          [["noise_removed", "error", "kept", "spurious", "seasonal"]].fillna("").to_string())
