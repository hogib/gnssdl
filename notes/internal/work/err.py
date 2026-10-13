"""Test-year leave-one-out fast error over radii for the far stack and M1 family."""
import sys
from pathlib import Path
sys.path.insert(0, "src")
from gnssdl import ngl
from gnssdl.dataset import Cube
from gnssdl.bench.run import ScoringContext, load_model
from gnssdl.paper_figures import loo_error

cube = Cube.load(Path("data/cube/california.npz"))
ctx = ScoringContext.build(cube, ngl.read_steps(Path("data/meta/steps.txt")))
for m, radii in (("fs", (0, 25, 50, 100, 200, 400, 600)), ("m1", (0, 10, 25, 50, 100, 200, 400)),
                 ("m1k64", (0, 100, 400)), ("m1k256", (0, 100, 400))):
    for R in radii:
        for split, name in ((1, "val"), (2, "test")):
            e = loo_error(load_model(m, float(R), cube, Path("data/runs"), ctx=ctx), cube, ctx, split)
            print(m, R, name, round(e, 4), flush=True)
