"""Where each model's files live: one directory per model under data/runs.

    data/runs/<model>/
        scores/<label>.csv, <label>.json   mask scores and configs (`bench run`);
                                           label is the model, or <model>-s<seed>
        signal/<period>/<kind>.csv         signal tests (`bench signal`): injections,
                                           noise, ridgecrest, events, event_stations,
                                           offsets, velocity, seasonal
        checkpoints/R<R>_seed<s>.pt, .json trained weights and training history
        logs/train.log                     per-step training log (learned models)
        logs/<time>_<command>.log          the console output of every bench command

`root` defaults to data/runs; `bench --results` overrides it.
"""

from __future__ import annotations

import re
from pathlib import Path

RUNS = Path("data/runs")
SIGNAL_KINDS = ("injections", "noise", "ridgecrest", "events", "event_stations", "offsets", "velocity",
                "seasonal")


def base_name(label: str) -> str:
    """The model a results label belongs to: `m5-s3` -> `m5`."""
    return re.sub(r"-s\d+$", "", label)


def run_dir(model: str, root: Path = RUNS) -> Path:
    return Path(root) / base_name(model)


def scores_path(label: str, root: Path = RUNS, suffix: str = ".csv") -> Path:
    return run_dir(label, root) / "scores" / f"{label}{suffix}"


def score_files(root: Path = RUNS) -> list[Path]:
    return sorted(Path(root).glob("*/scores/*.csv"))


def signal_path(model: str, period: str, kind: str, root: Path = RUNS) -> Path:
    return run_dir(model, root) / "signal" / period / f"{kind}.csv"


def signal_files(period: str, kind: str, root: Path = RUNS) -> list[Path]:
    return sorted(Path(root).glob(f"*/signal/{period}/{kind}.csv"))


def checkpoint_dir(model: str, root: Path = RUNS) -> Path:
    return run_dir(model, root) / "checkpoints"


def log_dir(model: str, root: Path = RUNS) -> Path:
    return run_dir(model, root) / "logs"
