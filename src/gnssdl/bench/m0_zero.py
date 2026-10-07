"""M0: predict zero (docs/01-m0-zero.md). The floor."""

from __future__ import annotations

import numpy as np

from gnssdl.bench.base import Reconstructor
from gnssdl.dataset import Cube


class M0Zero(Reconstructor):
    name = "m0"

    def predict(self, cube: Cube, hide: np.ndarray) -> np.ndarray:
        return np.zeros(cube.r.shape, dtype=np.float32)
