"""A filter written outside gnssdl, audited in one call.

    uv run python examples/median_filter.py            # needs the PNW cube (gnssdl build)

The filter: a station's common mode on a day is the median, per component, of
every other station farther than the exclusion radius. It never uses the
target station's own values, so it declares `never_reads_target = True`; the
audit's leak checks verify that before anything is scored. A filter that does
read the target (most stacking and PCA filters in use do) sets
`reference_filter = True` instead and is scored by the signal tests only.
"""

import numpy as np

import gnssdl
from gnssdl.dataset import distance_azimuth


class FarMedian(gnssdl.Filter):
    name = "farmedian"                 # command-line name, results directory, `model` column
    never_reads_target = True

    def predict(self, cube, hide):
        """`cube.r` is stations x days x 3 in mm; `cube.avail` marks the cells
        that may be read (hidden cells are already unavailable). Return the
        predicted common mode in the same shape."""
        values = np.where(cube.avail[..., None], cube.r, np.nan)
        distance, _ = distance_azimuth(cube.lat, cube.lon)
        out = np.zeros(values.shape, dtype=np.float32)
        for i in range(len(cube.sta)):
            far = distance[i] > self.radius_km
            far[i] = False
            if far.any():
                with np.errstate(all="ignore"):
                    out[i] = np.nan_to_num(np.nanmedian(values[far], axis=0))
        return out


if __name__ == "__main__":
    net = gnssdl.Network.load("pnw")
    # centres=5: a tenth of the benchmark's planted transients, for a card in minutes
    card = gnssdl.audit(FarMedian, net, centres=5, log=print)
    print(card)
