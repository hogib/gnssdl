"""The filters the benchmark knows, built-in and outside ones alike.

A filter is a `Reconstructor` subclass (bench/base.py) plus the few facts the
harness needs to run it: at which exclusion radii, whether it reads the
target station, whose tuning it reuses. Built-in filters are registered at
the bottom of this file; an outside filter registers the same way:

    from gnssdl.bench.base import Reconstructor
    from gnssdl.bench.registry import register

    class MyFilter(Reconstructor):
        name = "mine"
        def predict(self, cube, hide): ...

    register(MyFilter, title="My filter", radii=(0.0, 50.0))

or, to be found by the `gnssdl` command without any import, through an entry
point in the package that ships it (pyproject.toml):

    [project.entry-points."gnssdl.filters"]
    mine = "mypackage.filters:register_all"      # a function that calls register()

The name is the command-line name, the results directory and the `model`
column of every results file, so it never changes once results exist.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib.metadata import entry_points

from gnssdl.bench.base import Reconstructor

ENTRY_POINT_GROUP = "gnssdl.filters"
# what a filter reads, for grouping in tables and figures
FAMILIES = ("baseline", "reads-target", "far-stack", "nearest", "learned", "other")


@dataclass(frozen=True)
class FilterSpec:
    name: str                               # command-line name, results directory, `model` column
    cls: type[Reconstructor]
    title: str                              # display name in tables and figures
    family: str = "other"                   # one of FAMILIES
    radius_free: bool = False               # no exclusion radius: run once, radius column empty
    fixed_radius: float | None = None       # defined at one radius only
    radii: tuple[float, ...] | None = None  # its own radii (needs no neighbour lists); None: every cube radius
    reads_target: bool = False              # reference filter: scored by the signal tests, never on the masks
    tuned_from: str | None = None           # reads_target only: the filter whose tuned settings it reuses
    sweep: tuple[dict, ...] = ()            # settings swept rather than tuned; one labelled point each

    @property
    def has_seed(self) -> bool:
        """Trained filters take a training seed."""
        return "seed" in inspect.signature(self.cls.__init__).parameters

    @property
    def tuning_source(self) -> str | None:
        """Name of the results file its hyper-parameters are read from; None when it has none to read."""
        return self.tuned_from if self.reads_target else self.name


_FILTERS: dict[str, FilterSpec] = {}
_plugins_loaded = False


def register(cls: type[Reconstructor], *, name: str | None = None, title: str | None = None,
             family: str = "other", radius_free: bool = False, fixed_radius: float | None = None,
             radii: tuple[float, ...] | None = None, tuned_from: str | None = None,
             sweep: tuple[dict, ...] | list[dict] = (), replace: bool = False) -> FilterSpec:
    """Make a filter known to the benchmark. `name` defaults to the class's
    `name`; whether it reads the target comes from its `reference_filter`."""
    name = name or cls.name
    if not (isinstance(cls, type) and issubclass(cls, Reconstructor)):
        raise TypeError(f"{name}: a filter is a subclass of gnssdl.bench.base.Reconstructor")
    if not name or name == Reconstructor.name or not name.replace("_", "").isalnum() or name != name.lower():
        raise ValueError(f"filter name {name!r}: use lower-case letters, digits and underscores "
                         "(it becomes a directory and a results label)")
    if family not in FAMILIES:
        raise ValueError(f"{name}: family must be one of {FAMILIES}")
    if name in _FILTERS and not replace:
        raise ValueError(f"a filter named {name!r} is already registered")
    if radius_free and (fixed_radius is not None or radii):
        raise ValueError(f"{name}: a radius-free filter has no radii")
    spec = FilterSpec(name, cls, title or name, family, radius_free, fixed_radius,
                      tuple(float(r) for r in radii) if radii else None, bool(cls.reference_filter),
                      tuned_from, tuple(sweep))
    _FILTERS[name] = spec
    return spec


def unregister(name: str) -> None:
    """Forget a filter (its saved results stay readable as plain labels)."""
    _FILTERS.pop(name, None)


def _load_plugins() -> None:
    """Call every `gnssdl.filters` entry point once; each registers its filters."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        ep.load()()


def get(name: str) -> FilterSpec:
    _load_plugins()
    if name not in _FILTERS:
        raise KeyError(f"no filter named {name!r}; known: {', '.join(_FILTERS)}")
    return _FILTERS[name]


def find(name: str) -> FilterSpec | None:
    """Like `get`, but None for an unknown name (e.g. a label in an old results file)."""
    _load_plugins()
    return _FILTERS.get(name)


def names(masks: bool | None = None) -> list[str]:
    """Registered names, in registration order. `masks=True`: only filters
    scored on the masks (`bench run`); `masks=False`: only the reference
    filters that read the target."""
    _load_plugins()
    return [n for n, s in _FILTERS.items() if masks is None or s.reads_target != masks]


# --------------------------------------------------------------------------- built-in filters

def _builtin() -> None:
    from gnssdl.bench.audit import C3_COMPONENTS, C3_REGIONS, C1Stack, C3RegionalPCA, D1FarStack, FarStackSweep
    from gnssdl.bench.m0_zero import M0Zero
    from gnssdl.bench.m1_stack import M1K64, M1K256, M1Stack
    from gnssdl.bench.m2_robust import M2Robust, M2Self
    from gnssdl.bench.m3_ridge import M3Pooled, M3Ridge
    from gnssdl.bench.m3k_kernel import M3Kernel, M3Kernel256
    from gnssdl.bench.m4_band import M4Band, M4Pooled
    from gnssdl.bench.m5_attention import M5Attention, M5Aug
    from gnssdl.bench.m6_network import VARIANTS as M6_VARIANTS, M6Aug, M6Network

    register(M0Zero, family="baseline", radius_free=True)
    register(M1Stack, title="M1", family="nearest")
    register(M1K64, title="M1, 64", family="nearest")
    register(M1K256, title="M1, 256", family="nearest")
    register(M2Robust, title="M2", family="nearest")
    register(D1FarStack, title="D1", family="far-stack", fixed_radius=400.0)
    # needs no neighbour lists, so it runs at radii outside the cube (10 km added 2026-11-12)
    register(FarStackSweep, title="far stack", family="far-stack",
             radii=(0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 600.0))
    register(M3Kernel, title="M3k", family="learned")
    register(M3Kernel256, title="M3k, 256", family="learned")
    register(M5Attention, title="M5", family="learned")
    register(M5Aug, title="M5-aug", family="learned")
    register(M3Ridge, title="M3a", family="nearest")
    register(M3Pooled, title="M3b", family="nearest")
    register(M4Band, title="M4a", family="nearest")
    register(M4Pooled, title="M4b", family="nearest")
    # m6*: post-freeze experiment. The whole-network ones run one pass per group of targets, and
    # at R >= 200 km nearly every station needs its own pass (R-disc of 226-650 stations).
    whole_network = (0.0, 10.0, 25.0, 50.0, 100.0)
    register(M6Network, family="learned", radii=whole_network)
    register(M6Aug, family="learned", radii=whole_network)
    for cls in M6_VARIANTS.values():            # m6r[16|64|256][aug]: far-stack residual, N nearest
        # the N-nearest variants run one pass per station, so they take every cube radius
        register(cls, family="learned", radii=whole_network if cls.name in ("m6r", "m6raug") else None)
    # reference filters: they read the target on purpose
    register(M2Self, title="M2-self", family="reads-target", fixed_radius=0.0, tuned_from="m2")
    register(C1Stack, title="C1", family="reads-target", radius_free=True)
    register(C3RegionalPCA, family="reads-target", radius_free=True,
             sweep=[{"regions": k, "components": p} for k in C3_REGIONS for p in C3_COMPONENTS])


_builtin()
