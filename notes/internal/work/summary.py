"""Test-year summary numbers for the internal report (read-only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from gnssdl.paper_figures import _signal_tables  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 400)
t = _signal_tables(Path("data/runs"), "test")
inj, ev, st, off, rc, vel, noi = (t[k] for k in ("injections", "events", "event_stations", "offsets",
                                                  "ridgecrest", "velocity", "noise"))

print("== noise removed by model, R")
print(noi.pivot_table(index="model", columns="R", values="noise_removed").round(3))

print("\n== retention by L (median), R = 0 (d1 at 400)")
g = inj.groupby(["model", "R", "footprint_km"]).rho.median().unstack().round(2)
print(g)
print("\n== n injections by model, R")
print(inj.groupby(["model", "R"]).size().unstack())
print("\n== spurious_max median by L")
print(inj.groupby(["model", "R", "footprint_km"]).spurious_max.median().unstack().round(2))
print("\n== retention by amplitude, L = 25 and 100, R = 0")
z = inj[inj.R == 0]
print(z.groupby(["model", "footprint_km", "amplitude_mm"]).rho.median().unstack().round(2))
print("\n== retention by duration, L = 25 and 100, R = 0")
print(z.groupby(["model", "footprint_km", "duration_days"]).rho.median().unstack().round(2))

print("\n== events: count, whole-event rho, spurious")
print(ev.groupby(["model", "R", "template"]).agg(n=("rho", "size"), rho=("rho", "median"),
                                               spur=("spurious_max", "median"),
                                               stations=("stations", "median"),
                                               peak=("peak_mm", "median")).round(2))
print("\n== afterslip rho by distance bin, R = 0")
a = st[(st.template == "afterslip")]
bins = [0, 5, 10, 20, 40, 80, 400]
a = a.assign(bin=pd.cut(a.dist_km, bins))
print(a.groupby(["model", "R", "bin"], observed=False).rho.median().unstack().round(2))
a2 = a.assign(band=pd.cut(a.dist_km, [0, 5, 15, 40, 1000]))
print(a2.groupby(["model", "R", "band"], observed=False).rho.median().unstack().round(2))
print("\n== afterslip near-fault rho by side, within 15 km")
n15 = a[a.dist_km <= 15]
print(n15.groupby(["model", "R", "side"]).rho.median().unstack().round(2))

print("\n== offsets")
o = off[off.R == 0]
print("per model n:", o.groupby("model").size().to_dict())
print(o[o.model == "fs"].groupby(["event", "mag"]).size())
print(off.groupby(["model", "R"]).kept.median().unstack().round(2))
print("offset_mm median", o[o.model == "fs"].offset_mm.median())

print("\n== ridgecrest")
print(rc.groupby(["model", "R"]).retained.median().unstack().round(2))
print("n stations", rc[rc.model == "fs"].shape[0])

print("\n== velocity |dv| median E N U")
v = vel.assign(ae=vel.dv_e.abs(), an=vel.dv_n.abs(), au=vel.dv_u.abs())
print(v.groupby(["model", "R"])[["ae", "an", "au"]].median().round(2))
print("n stations", v.groupby("model").size().to_dict())
