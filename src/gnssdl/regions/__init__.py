"""Regions: what is specific to one study area, in one file per region.

A region file (TOML) names the area and, optionally, its slow-slip episode
list; everything else in the benchmark is the same for every network. The
built-in regions live beside this file (`pnw.toml`); your own can live
anywhere:

    name = "mine"
    title = "My region"
    bbox = [S, N, W, E]                 # optional; the box of `gnssdl fetch --bbox`

    [episodes]                          # optional; paths are relative to the region file
    list = "mine/episodes.csv"          # columns: id, source, timing, start, end, lat_min, lat_max, note
    band = "mine/band.csv"              # the slip band's axis: its longitude per latitude bin
    exclude_from_training = true

A region with no `[episodes]` table has no slow-slip episode measure: it is
skipped with a message, and the other measures are unaffected.

A cube belongs to the region recorded in its summary file (`"region"` in the
.json beside it, written by `gnssdl build --region`), or else to the built-in
region with the cube file's name, or else to none.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

BUILTIN_DIR = Path(__file__).parent


class RegionError(ValueError):
    pass


@dataclass(frozen=True)
class Region:
    name: str
    title: str
    source: Path                                   # the region file
    bbox: tuple[float, float, float, float] | None = None     # S, N, W, E
    min_days: int | None = None
    episodes_path: Path | None = None
    band_path: Path | None = None
    catalogue_path: Path | None = None
    exclude_episodes_from_training: bool = False

    @property
    def has_episodes(self) -> bool:
        return self.episodes_path is not None

    def load_episodes(self) -> pd.DataFrame:
        from gnssdl.bench import episodes as eps
        self._need_episodes()
        return eps.load_episodes(self.episodes_path)

    def load_band(self) -> pd.DataFrame:
        from gnssdl.bench import episodes as eps
        self._need_episodes()
        return eps.load_band(self.band_path)

    def training_exclusion(self) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None:
        """f(lat, lon, days) -> S × T mask of cells barred from training
        because of the region's catalogued episodes; None when the region has
        none or does not bar them."""
        if not (self.has_episodes and self.exclude_episodes_from_training):
            return None
        from gnssdl.bench import episodes as eps
        episodes, band = self.load_episodes(), self.load_band()
        return lambda lat, lon, days: eps.training_exclusion(lat, lon, days, episodes, band)

    def _need_episodes(self) -> None:
        if not self.has_episodes:
            raise RegionError(f"region {self.name!r} has no slow-slip episode list ({self.source} has no "
                              "[episodes] table)")


def read(path: Path | str) -> Region:
    """A region from its TOML file."""
    path = Path(path)
    if not path.exists():
        raise RegionError(f"no region file {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    known = {"name", "title", "bbox", "min_days", "episodes"}
    if set(raw) - known:
        raise RegionError(f"{path}: unknown keys {sorted(set(raw) - known)}; known: {sorted(known)}")
    if not isinstance(raw.get("name"), str) or not raw["name"]:
        raise RegionError(f"{path}: `name` is required")
    bbox = raw.get("bbox")
    if bbox is not None and not (len(bbox) == 4 and bbox[0] < bbox[1] and bbox[2] < bbox[3]):
        raise RegionError(f"{path}: bbox is [S, N, W, E] with S < N and W < E")
    ep = raw.get("episodes")
    files: dict[str, Path | None] = {"list": None, "band": None, "catalogue": None}
    exclude = False
    if ep is not None:
        unknown = set(ep) - {"list", "band", "catalogue", "exclude_from_training"}
        if unknown:
            raise RegionError(f"{path}: unknown [episodes] keys {sorted(unknown)}")
        for key in ("list", "band"):
            if key not in ep:
                raise RegionError(f"{path}: [episodes] needs both `list` and `band`")
        for key in files:
            if key in ep:
                files[key] = (path.parent / ep[key]).resolve()
                if not files[key].exists():
                    raise RegionError(f"{path}: [episodes] {key} = {ep[key]!r}: no such file")
        exclude = bool(ep.get("exclude_from_training", False))
    return Region(raw["name"], raw.get("title", raw["name"]), path, None if bbox is None else tuple(map(float, bbox)),
                  raw.get("min_days"), files["list"], files["band"], files["catalogue"], exclude)


def names() -> list[str]:
    """The built-in regions."""
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.toml"))


def get(region: "Region | str | Path") -> Region:
    """A Region as it is, a built-in region by name, or a region file by path."""
    if isinstance(region, Region):
        return region
    text = str(region)
    if text.endswith(".toml"):
        return read(text)
    path = BUILTIN_DIR / f"{text}.toml"
    if not path.exists():
        raise RegionError(f"no built-in region {text!r} (built in: {', '.join(names())}); "
                          "give the path of a region .toml file instead")
    return read(path)


def for_cube(cube_path: Path | str) -> Region | None:
    """The region a cube belongs to (see the module docstring), or None."""
    cube_path = Path(cube_path)
    summary = cube_path.with_suffix(".json")
    if summary.exists():
        try:
            record = json.loads(summary.read_text())
        except json.JSONDecodeError:
            record = {}
        if isinstance(record, dict) and "region" in record:
            return None if record["region"] is None else get(record["region"])
    return get(cube_path.stem) if cube_path.stem in names() else None
