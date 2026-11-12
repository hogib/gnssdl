"""A queue of benchmark jobs described in one file, restartable.

    gnssdl queue plan   experiments/pnw_validation.toml    # list the jobs; runs nothing
    gnssdl queue run    experiments/pnw_validation.toml    # run what is not done yet
    gnssdl queue status experiments/pnw_validation.toml    # done / failed / pending per phase

The manifest (TOML) names a network and a list of phases; a phase is one task
for a set of filters, seeds and radii, run a few jobs at a time. Each job is
one `gnssdl bench ...` command in its own process with its own log:

    network = "pnw"                 # a network name or a cube path
    period = "validation"           # the test period is refused before the freeze
    centres = 20                    # planted transients per grid cell (signal phases)

    [groups]                        # optional: names for lists of filters, used as "@name"
    light = ["m3k", "m5"]

    [[phase]]
    name = "train"
    task = "fit"                    # fit | transients | noise | seasonal | offsets | episodes | tables,
                                    # or a list of signal tasks: ["transients", "seasonal"]
    filters = ["@light"]
    seeds = [1, 2]                  # omit for the default seed and for filters without one
    radii = [0, 25]                 # omit for every radius of each filter
    per_radius = true               # one job per radius (default: one job per filter and seed)
    reuse = true                    # fit only: load a checkpoint where one exists
    jobs = 3                        # how many at a time

    [[phase]]
    name = "tables"
    task = "tables"                 # neighbours, verdict and intervals; always recomputed

State lives in <data-dir>/queue/<manifest name>/ (or `logs = "..."` in the
manifest): queue.log (one line per start and end, flushed, for `tail -f`),
done.txt (jobs that exited 0, skipped on the next run), <job>.log. Running the
manifest again runs only what is not done; a last pass retries the failures
one at a time. The queue only starts processes: detaching it from the
terminal (setsid nohup ... &) is up to whoever starts it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SIGNAL_FLAGS = {"transients": [], "noise": ["--noise-only"], "seasonal": ["--seasonal"],
                "offsets": ["--offsets"], "episodes": ["--episodes"]}
TASKS = ("fit", *SIGNAL_FLAGS, "tables")
TABLES = ("neighbours", "verdict", "intervals")
PHASE_KEYS = {"name", "task", "filters", "seeds", "radii", "per_radius", "reuse", "jobs"}
TOP_KEYS = {"network", "period", "results", "data_dir", "centres", "logs", "nice", "timeout_hours", "env",
            "groups", "phase"}


class QueueError(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    name: str                 # unique in the manifest; the log file and the done-list entry
    phase: str
    argv: tuple[str, ...]     # arguments of the `gnssdl` command
    always: bool = False      # run every time (tables), never recorded as done


@dataclass
class Phase:
    name: str
    width: int
    jobs: list[Job] = field(default_factory=list)


@dataclass
class Queue:
    path: Path
    cube: Path
    period: str
    results: Path
    data_dir: Path
    logs: Path
    nice: int | None
    timeout_s: float | None
    env: dict[str, str]
    phases: list[Phase]

    @property
    def jobs(self) -> list[Job]:
        return [j for p in self.phases for j in p.jobs]

    @property
    def main_log(self) -> Path:
        return self.logs / "queue.log"

    @property
    def done_file(self) -> Path:
        return self.logs / "done.txt"

    def done(self) -> set[str]:
        return set(self.done_file.read_text().split()) if self.done_file.exists() else set()


def _radius_text(r: float) -> str:
    return f"{float(r):g}"


def _phase_jobs(path, raw_phase: dict, task: str, period: str, centres, top: list[str], common: list[str],
                filters_of, radii_of) -> list[Job]:
    """The jobs of one task of one phase."""
    from gnssdl.bench import registry
    name = raw_phase["name"]
    jobs: list[Job] = []
    if task == "tables":
        for t in TABLES:
            argv = [*top, "bench", t, *common] + ([] if t == "neighbours" else ["--period", period])
            jobs.append(Job(f"{name}_{t}", name, tuple(argv), always=True))
    else:
        asked = [float(r) for r in raw_phase["radii"]] if "radii" in raw_phase else None
        per_radius = bool(raw_phase.get("per_radius", False))
        for seed in raw_phase.get("seeds", [None]):
            for flt in filters_of(raw_phase):
                spec = registry.get(flt)
                if seed is not None and not spec.has_seed:
                    raise QueueError(f"{path}: phase {name!r}: {flt} has no training seed")
                if task == "fit" and spec.reads_target:
                    raise QueueError(f"{path}: phase {name!r}: {flt} reads the target, so it is not fitted on "
                                     "the masks; give it signal phases only")
                fixed = spec.radius_free or spec.fixed_radius is not None
                sets = [[r] for r in radii_of(flt, asked)] if per_radius and not fixed else \
                    [None if fixed or asked is None else radii_of(flt, asked)]
                for radii in sets:
                    if radii is not None and not radii:
                        continue                      # none of the asked radii is one of this filter's
                    tag = f"{task}_{flt}" + ("" if seed is None else f"_s{seed}") \
                        + ("" if radii is None or not per_radius else f"_R{_radius_text(radii[0])}")
                    argv = [*top, "bench", "run" if task == "fit" else "signal", flt, *common]
                    if seed is not None:
                        argv += ["--seed", str(seed)]
                    for r in radii or []:
                        argv += ["--radius", _radius_text(r)]
                    if task == "fit":
                        argv += ["--reuse"] if raw_phase.get("reuse", False) else []
                        argv += [] if period == "validation" else ["--period", "all"]   # after the freeze
                    else:
                        argv += ["--period", period, *SIGNAL_FLAGS[task]]
                        if task == "transients" and centres is not None:
                            argv += ["--centres", str(int(centres))]
                    jobs.append(Job(f"{name}_{tag}", name, tuple(argv)))
    return jobs


def load(path: Path | str) -> Queue:
    """Read a manifest and expand it into jobs. Nothing is run."""
    from gnssdl.bench import registry
    from gnssdl.bench.freeze import require
    from gnssdl.bench.paths import RUNS

    path = Path(path)
    if not path.exists():
        raise QueueError(f"no manifest {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if set(raw) - TOP_KEYS:
        raise QueueError(f"{path}: unknown keys {sorted(set(raw) - TOP_KEYS)}")
    if "network" not in raw or not raw.get("phase"):
        raise QueueError(f"{path}: needs `network` and at least one [[phase]]")
    period = raw.get("period", "validation")
    require(period)                                   # NotFrozen for test or prospective before the freeze
    data_dir = Path(raw.get("data_dir", "data"))
    network = str(raw["network"])
    cube = Path(network) if network.endswith(".npz") else data_dir / "cube" / network / f"{network}.npz"
    results = Path(raw.get("results", RUNS))
    logs = Path(raw["logs"]) if "logs" in raw else data_dir / "queue" / path.stem
    groups = raw.get("groups", {})
    common = ["--cube", str(cube), "--results", str(results)]
    top = ["--data-dir", str(data_dir)]
    centres = raw.get("centres")
    _cube_radii: list[float] = []

    def filters_of(phase: dict) -> list[str]:
        out: list[str] = []
        for item in phase.get("filters", []):
            names = groups.get(item[1:]) if item.startswith("@") else [item]
            if names is None:
                raise QueueError(f"{path}: phase {phase['name']!r} uses the group {item!r}, which [groups] lacks")
            out += names
        for name in out:
            if registry.find(name) is None:
                raise QueueError(f"{path}: phase {phase['name']!r}: no filter named {name!r}")
        if not out:
            raise QueueError(f"{path}: phase {phase['name']!r} lists no filters")
        return list(dict.fromkeys(out))

    def radii_of(name: str, asked: list[float] | None) -> list[float]:
        """The radii of one filter for a per-radius phase: those asked that it
        runs at, or all of its own."""
        from gnssdl.bench.run import model_radii
        spec = registry.get(name)
        if spec.radius_free or spec.fixed_radius is not None:
            return model_radii(name, None)
        if spec.radii:
            return [r for r in spec.radii if asked is None or r in asked]
        if asked is not None:
            return asked
        if not _cube_radii:                           # every radius of the cube: read it once
            from gnssdl.dataset import Cube
            if not cube.exists():
                raise QueueError(f"{path}: no cube at {cube}; `radii` omitted with per_radius needs it")
            _cube_radii.extend(float(r) for r in Cube.load(cube).radii)
        return list(_cube_radii)

    phases, seen = [], set()
    for raw_phase in raw["phase"]:
        if set(raw_phase) - PHASE_KEYS:
            raise QueueError(f"{path}: phase {raw_phase.get('name')!r}: unknown keys "
                             f"{sorted(set(raw_phase) - PHASE_KEYS)}")
        name, task = raw_phase.get("name"), raw_phase.get("task")
        tasks = task if isinstance(task, list) else [task]
        if not name or not tasks or any(t not in TASKS for t in tasks):
            raise QueueError(f"{path}: every phase needs a `name` and a `task` from {TASKS}")
        if len(tasks) > 1 and not set(tasks) <= set(SIGNAL_FLAGS):
            raise QueueError(f"{path}: phase {name!r}: only signal tasks can be listed together")
        phase = Phase(name, int(raw_phase.get("jobs", 1)))
        for task in tasks:
            phase.jobs += _phase_jobs(path, raw_phase, task, period, centres, top, common, filters_of, radii_of)
        for job in phase.jobs:
            if job.name in seen:
                raise QueueError(f"{path}: two jobs are named {job.name!r}; give the phases different names")
            seen.add(job.name)
        phases.append(phase)
    timeout = raw.get("timeout_hours")
    return Queue(path, cube, period, results, data_dir, logs, raw.get("nice"),
                 None if timeout is None else float(timeout) * 3600.0,
                 {str(k): str(v) for k, v in raw.get("env", {}).items()}, phases)


def plan(queue: Queue) -> str:
    """The jobs, one line each, with what is already done."""
    done = queue.done()
    lines = [f"{queue.path}: {len(queue.jobs)} jobs in {len(queue.phases)} phases; logs in {queue.logs}"]
    for p in queue.phases:
        lines.append(f"{p.name}: {len(p.jobs)} jobs, {p.width} at a time")
        for j in p.jobs:
            mark = "always" if j.always else "done" if j.name in done else "to run"
            lines.append(f"  [{mark:>6}] {j.name}: gnssdl {' '.join(j.argv)}")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now().strftime("%m-%d %H:%M")


def run(queue: Queue, command: list[str] | None = None) -> int:
    """Run every job that is not done, phase by phase, then retry the
    failures one at a time. Returns the number of jobs still failing.
    `command` is what a job's arguments are appended to (default: this
    Python running the gnssdl command line)."""
    command = command or [sys.executable, "-c", "from gnssdl.cli import main; raise SystemExit(main())"]
    queue.logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    env = {**os.environ, **queue.env}
    prefix = [] if queue.nice is None else ["nice", "-n", str(int(queue.nice))]
    running: set[subprocess.Popen] = set()

    with queue.main_log.open("a", buffering=1) as main:
        def say(text: str) -> None:
            with lock:
                main.write(text + "\n")
                main.flush()

        def one(job: Job) -> bool:
            if not job.always and job.name in queue.done():
                say(f"skip {job.name} (done)")
                return True
            say(f"start {job.name} {_now()}")
            with (queue.logs / f"{job.name}.log").open("w") as log:
                log.write("$ gnssdl " + " ".join(job.argv) + "\n")
                log.flush()
                proc = subprocess.Popen([*prefix, *command, *job.argv], stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, env=env)
                with lock:
                    running.add(proc)
                try:
                    code = proc.wait(timeout=queue.timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    code = "timeout"
                finally:
                    with lock:
                        running.discard(proc)
            if code == 0 and not job.always:
                with lock, queue.done_file.open("a") as f:
                    f.write(job.name + "\n")
            say(f"end {job.name} exit {code} {_now()}")
            return code == 0

        def run_phase(name: str, jobs: list[Job], width: int) -> list[Job]:
            say(f"=== {name}: {len(jobs)} jobs, {width} at a time {_now()}")
            with ThreadPoolExecutor(max_workers=max(1, width)) as pool:
                ok = list(pool.map(one, jobs))
            return [j for j, good in zip(jobs, ok) if not good]

        say(f"=== queue {queue.path.name} started {_now()}")
        failed: list[Job] = []
        try:
            for p in queue.phases:
                failed += run_phase(p.name, p.jobs, p.width)
            if failed:
                failed = run_phase("retry", failed, 1)
        except BaseException:
            with lock:
                for proc in running:                  # do not leave jobs running behind a stopped queue
                    proc.terminate()
            say(f"=== queue stopped {_now()}")
            raise
        say(f"=== all done {_now()}; still failing: {len(failed)}"
            + ("" if not failed else " (" + ", ".join(j.name for j in failed) + ")"))
    return len(failed)


def status(queue: Queue) -> dict:
    """Where the queue stands, from its done list and main log: per phase the
    jobs done, failed (last end was not exit 0), started and not ended, and
    not started; plus the names of the failed and unfinished ones."""
    done = queue.done()
    last: dict[str, str] = {}
    finished = False
    if queue.main_log.exists():
        for line in queue.main_log.read_text().splitlines():
            parts = line.split()
            if line.startswith("=== queue") and "started" in line:
                finished = False
            elif line.startswith("=== all done") or line.startswith("=== queue stopped"):
                finished = True
            elif len(parts) >= 2 and parts[0] == "start":
                last[parts[1]] = "started"
            elif len(parts) >= 4 and parts[0] == "end":
                last[parts[1]] = "ok" if parts[3] == "0" else "failed"
    phases, failed, unfinished = [], [], []
    for p in queue.phases:
        row = {"phase": p.name, "jobs": len(p.jobs), "done": 0, "failed": 0, "started": 0, "pending": 0}
        for j in p.jobs:
            state = "done" if j.name in done or (j.always and last.get(j.name) == "ok") else \
                {"failed": "failed", "started": "started"}.get(last.get(j.name, ""), "pending")
            row[state] += 1
            if state == "failed":
                failed.append(j.name)
            elif state == "started":
                unfinished.append(j.name)
        phases.append(row)
    return {"phases": phases, "failed": failed, "started_not_ended": unfinished,
            "last_run_finished": finished, "log": str(queue.main_log)}
