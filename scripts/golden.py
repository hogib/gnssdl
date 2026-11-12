"""Before/after check for a refactor: the same inputs must give the same tables.

    uv run python scripts/golden.py make            # write the reference outputs (before the refactor)
    uv run python scripts/golden.py check           # recompute and compare (after every step)
    uv run python scripts/golden.py make --compute  # also the part that runs filters (minutes, one core)
    uv run python scripts/golden.py check --compute

Two parts, both driven through the command line, the one interface a refactor
must not change:

  saved    reads a snapshot of result tables (data/golden/runs, copied from the
           server) and rebuilds the scoreboard, the success-criterion verdict and
           the bootstrap intervals. No filter is run.
  compute  runs two classical filters on the local cube (noise removed, seasonal,
           offsets, episodes, a small planted-transient test, the paired
           comparison). Needs data/cube/pnw/pnw.npz and data/meta.

Compared: everything printed to stdout and every CSV written. The snapshot is
never written to: each part works in a temporary results directory. Nothing
here reads the test period.
"""

from __future__ import annotations

import argparse
import contextlib
import filecmp
import io
import shutil
import sys
import tempfile
from pathlib import Path

GOLDEN = Path("data/golden")
SNAPSHOT = GOLDEN / "runs"
EXPECTED = GOLDEN / "expected"
CUBE = Path("data/cube/pnw/pnw.npz")

SAVED = [
    ("report", ["bench", "report"]),
    ("verdict", ["bench", "verdict"]),
    ("intervals", ["bench", "intervals"]),
]
COMPUTE = [
    ("neighbours", ["bench", "neighbours"]),
    ("fs_noise", ["bench", "signal", "fs", "--radius", "0", "--noise-only"]),
    ("fs_seasonal", ["bench", "signal", "fs", "--radius", "0", "--seasonal"]),
    ("fs_offsets", ["bench", "signal", "fs", "--radius", "0", "--offsets"]),
    ("fs_episodes", ["bench", "signal", "fs", "--radius", "0", "--episodes"]),
    ("fs_transients", ["bench", "signal", "fs", "--radius", "25", "--centres", "1"]),
    ("m1_transients", ["bench", "signal", "m1", "--radius", "25", "--centres", "1"]),
    ("compare", ["bench", "compare", "fs", "m1", "--radius", "25"]),
    ("report", ["bench", "report"]),
]
COMPUTE_MODELS = ("fs", "m1")


def _run(argv: list[str], results: Path) -> str:
    """One gnssdl command against `results`; returns its stdout with the
    temporary path replaced, so two runs can be compared."""
    from gnssdl.cli import main
    cmd = [argv[0], argv[1], "--results", str(results), "--cube", str(CUBE), *argv[2:]]
    out = io.StringIO()
    old_argv = sys.argv
    sys.argv = ["gnssdl", *cmd]              # the per-command log quotes sys.argv
    try:
        with contextlib.redirect_stdout(out):
            code = main(cmd)
    finally:
        sys.argv = old_argv
    return out.getvalue().replace(str(results), "<results>") + f"[exit {code}]\n"


def _produce(part: str, dest: Path) -> None:
    """Run the part's commands in a temporary results directory and collect
    their stdout and written CSVs under `dest`."""
    commands = SAVED if part == "saved" else COMPUTE
    with tempfile.TemporaryDirectory(prefix="gnssdl-golden-") as tmp:
        results = Path(tmp)
        before: set[Path] = set()
        if part == "saved":
            for d in sorted(p for p in SNAPSHOT.iterdir() if p.is_dir()):
                (results / d.name).symlink_to(d.resolve(), target_is_directory=True)
        else:
            for name in COMPUTE_MODELS:          # tuned settings only; signal results are recomputed
                shutil.copytree(SNAPSHOT / name / "scores", results / name / "scores")
            before = {p.relative_to(results) for p in results.rglob("*.csv")}
        for k, (name, argv) in enumerate(commands):
            print(f"  {part}: gnssdl {' '.join(argv)}", file=sys.stderr, flush=True)
            (dest / f"{k:02d}_{name}.out").write_text(_run(argv, results))
        # files written by the commands: top-level tables, and in the compute part the signal files
        written = [p for p in results.glob("*.csv")]
        if part == "compute":
            written += [p for p in results.rglob("*.csv") if p.relative_to(results) not in before
                        and p.parent != results]
        for p in sorted(written):
            target = dest / "files" / p.relative_to(results)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)


def _files(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


def _compare(expected: Path, got: Path) -> list[str]:
    problems = []
    a, b = _files(expected), _files(got)
    for p in sorted(set(a) - set(b)):
        problems.append(f"missing now: {p}")
    for p in sorted(set(b) - set(a)):
        problems.append(f"new file: {p}")
    for p in sorted(set(a) & set(b)):
        if not filecmp.cmp(expected / p, got / p, shallow=False):
            ea, eb = (expected / p).read_text().splitlines(), (got / p).read_text().splitlines()
            line = next((i for i, (x, y) in enumerate(zip(ea, eb)) if x != y), min(len(ea), len(eb)))
            problems.append(f"differs: {p} (first at line {line + 1}; {len(ea)} lines before, {len(eb)} now)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["make", "check"])
    ap.add_argument("--compute", nargs="?", const=True, default=False, choices=[True, "only"],
                    help="also the part that runs filters on the local cube; `--compute only` skips the saved part")
    args = ap.parse_args()
    if not SNAPSHOT.is_dir():
        print(f"no snapshot at {SNAPSHOT}: copy the result tables there first", file=sys.stderr)
        return 1
    parts = ["compute"] if args.compute == "only" else ["saved"] + (["compute"] if args.compute else [])
    if args.compute and not CUBE.exists():
        print(f"no cube at {CUBE}", file=sys.stderr)
        return 1
    failed = False
    for part in parts:
        expected = EXPECTED / part
        if args.action == "make":
            if expected.exists():
                shutil.rmtree(expected)
            expected.mkdir(parents=True)
            _produce(part, expected)
            print(f"{part}: wrote {len(_files(expected))} reference files under {expected}")
            continue
        if not expected.is_dir():
            print(f"{part}: no reference under {expected}; run `make` on the code before the refactor",
                  file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory(prefix="gnssdl-golden-out-") as tmp:
            _produce(part, Path(tmp))
            problems = _compare(expected, Path(tmp))
            if problems:
                failed = True
                keep = GOLDEN / "last_mismatch" / part
                if keep.exists():
                    shutil.rmtree(keep)
                shutil.copytree(tmp, keep)
                print(f"{part}: {len(problems)} DIFFERENCES (outputs kept under {keep})")
                for line in problems:
                    print(f"  {line}")
            else:
                print(f"{part}: identical ({len(_files(expected))} files)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
