"""gnssdl: how much real signal does a GNSS common-mode filter remove?

    import gnssdl

    net = gnssdl.Network.load("pnw")          # a built cube and its results root
    card = gnssdl.audit(MyFilter, net)        # a registered name or a Reconstructor subclass
    print(card)

`Filter` is the base class to subclass and `register` makes a filter known
by name to the `gnssdl` command; see gnssdl/bench/card.py and registry.py.
"""

_PUBLIC = {
    "audit": ("gnssdl.bench.card", "audit"),
    "ReportCard": ("gnssdl.bench.card", "ReportCard"),
    "Network": ("gnssdl.bench.tasks", "Network"),
    "TaskError": ("gnssdl.bench.tasks", "TaskError"),
    "Filter": ("gnssdl.bench.base", "Reconstructor"),
    "register": ("gnssdl.bench.registry", "register"),
}
__all__ = ["main", *_PUBLIC]


def __getattr__(name: str):
    # loaded on first use, so `gnssdl meta` or `gnssdl fetch` do not import the benchmark
    if name in _PUBLIC:
        from importlib import import_module
        module, attr = _PUBLIC[name]
        value = getattr(import_module(module), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'gnssdl' has no attribute {name!r}")


def main() -> None:
    from gnssdl.cli import main as cli_main

    raise SystemExit(cli_main())
