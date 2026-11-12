"""The manifest queue (gnssdl.queue): expansion into jobs, restart, retry, status."""
import sys
from pathlib import Path

import pytest

from gnssdl import queue as q

# a stand-in for the gnssdl command: fails for any job whose arguments contain the word in FAIL_FILE
STUB = """
import sys, pathlib
fail = pathlib.Path(sys.argv[1])
args = sys.argv[2:]
print("ran", " ".join(args))
bad = fail.read_text().split() if fail.exists() else []
raise SystemExit(1 if any(b in args for b in bad) else 0)
"""


def _manifest(tmp_path, body):
    path = tmp_path / "grid.toml"
    path.write_text(f'network = "toy"\ndata_dir = "{tmp_path}"\nresults = "{tmp_path}/runs"\n' + body)
    return path


GRID = """
centres = 5
[groups]
stacks = ["m1", "fs"]

[[phase]]
name = "fit"
task = "fit"
filters = ["m1", "m5"]
radii = [0, 25]
per_radius = true
reuse = true
jobs = 2

[[phase]]
name = "signal"
task = ["transients", "seasonal"]
filters = ["@stacks", "c1"]
radii = [0, 10]

[[phase]]
name = "seeds"
task = "noise"
filters = ["m5"]
seeds = [1, 2]

[[phase]]
name = "tables"
task = "tables"
"""


def test_a_manifest_expands_into_the_gnssdl_commands(tmp_path):
    queue = q.load(_manifest(tmp_path, GRID))
    jobs = {j.name: " ".join(j.argv) for j in queue.jobs}
    assert [p.name for p in queue.phases] == ["fit", "signal", "seeds", "tables"] and queue.phases[0].width == 2
    assert list(jobs)[:4] == ["fit_fit_m1_R0", "fit_fit_m1_R25", "fit_fit_m5_R0", "fit_fit_m5_R25"]
    common = f"--cube {tmp_path}/cube/toy/toy.npz --results {tmp_path}/runs"
    assert jobs["fit_fit_m5_R25"] == f"--data-dir {tmp_path} bench run m5 {common} --radius 25 --reuse"
    assert jobs["signal_transients_m1"] == (f"--data-dir {tmp_path} bench signal m1 {common} --radius 0 --radius 10 "
                                            "--period validation --centres 5")
    # the far stack runs at its own radii, 10 km among them; C1 has no radius
    assert jobs["signal_seasonal_fs"].endswith("--radius 0 --radius 10 --period validation --seasonal")
    assert jobs["signal_seasonal_c1"].endswith(f"bench signal c1 {common} --period validation --seasonal")
    assert jobs["seeds_noise_m5_s2"].endswith(f"bench signal m5 {common} --seed 2 --period validation --noise-only")
    assert [j.name for j in queue.jobs if j.always] == ["tables_neighbours", "tables_verdict", "tables_intervals"]
    assert jobs["tables_neighbours"].endswith(f"bench neighbours {common}")
    assert jobs["tables_verdict"].endswith("--period validation")
    assert queue.logs == tmp_path / "queue" / "grid" and "to run" in q.plan(queue)


def test_the_packaged_manifest_is_valid_and_covers_every_filter():
    from gnssdl.bench import registry
    queue = q.load("experiments/pnw_validation.toml")
    used = {j.argv[4] for j in queue.jobs if j.argv[3] in ("run", "signal")}
    assert used == set(registry.names())
    assert all("--period" not in j.argv or j.argv[j.argv.index("--period") + 1] == "validation" for j in queue.jobs)


@pytest.mark.parametrize("body, why", [
    ("", "at least one"),
    ('[[phase]]\nname = "a"\ntask = "dance"\n', "task"),
    ('[[phase]]\nname = "a"\ntask = "fit"\nfilters = ["nope"]\n', "no filter named"),
    ('[[phase]]\nname = "a"\ntask = "fit"\nfilters = ["@g"]\n', "group"),
    ('[[phase]]\nname = "a"\ntask = "fit"\nfilters = ["c1"]\n', "reads the target"),
    ('[[phase]]\nname = "a"\ntask = "noise"\nfilters = ["m1"]\nseeds = [1]\n', "no training seed"),
    ('[[phase]]\nname = "a"\ntask = ["fit", "noise"]\nfilters = ["m1"]\n', "only signal tasks"),
    ('[[phase]]\nname = "a"\ntask = "noise"\nfilters = ["m1"]\nspeed = 3\n', "unknown keys"),
    ('[[phase]]\nname = "a"\ntask = "noise"\nfilters = ["m1"]\n[[phase]]\nname = "a"\ntask = "noise"\n'
     'filters = ["m1"]\n', "two jobs"),
    ('colour = 1\n[[phase]]\nname = "a"\ntask = "tables"\n', "unknown keys"),
])
def test_a_wrong_manifest_is_refused_before_anything_runs(tmp_path, body, why):
    with pytest.raises(q.QueueError, match=why):
        q.load(_manifest(tmp_path, body))


def test_the_test_period_is_refused_before_the_freeze(tmp_path, monkeypatch):
    from gnssdl.bench.freeze import NotFrozen
    path = _manifest(tmp_path, 'period = "test"\n[[phase]]\nname = "a"\ntask = "tables"\n')
    monkeypatch.chdir(tmp_path)                                     # a directory with no freeze record
    with pytest.raises(NotFrozen, match="refused"):
        q.load(path)


def test_run_skips_what_is_done_retries_failures_and_reports_status(tmp_path):
    queue = q.load(_manifest(tmp_path, GRID))
    stub, fail = tmp_path / "stub.py", tmp_path / "fail.txt"
    stub.write_text(STUB)
    command = [sys.executable, str(stub), str(fail)]
    assert q.status(queue)["phases"][0] == {"phase": "fit", "jobs": 4, "done": 0, "failed": 0, "started": 0,
                                            "pending": 4}
    fail.write_text("m5")                                           # every m5 job fails
    n_m5 = sum("m5" in j.argv for j in queue.jobs)
    assert q.run(queue, command) == n_m5
    st = q.status(queue)
    assert st["last_run_finished"] and len(st["failed"]) == n_m5 and all("m5" in name for name in st["failed"])
    assert [p["done"] for p in st["phases"]] == [2, 6, 0, 3] and not st["started_not_ended"]
    log = queue.main_log.read_text()
    assert "=== retry: 4 jobs, 1 at a time" in log and f"still failing: {n_m5}" in log
    assert (queue.logs / "fit_fit_m1_R0.log").read_text().splitlines()[0].startswith("$ gnssdl --data-dir")
    assert "done" in q.plan(queue) and "tables_verdict" not in queue.done()      # tables are never marked done

    fail.write_text("")                                             # fixed: the second run does only what is left
    assert q.run(queue, command) == 0
    second = queue.main_log.read_text()[len(log):]
    assert "skip fit_fit_m1_R0 (done)" in second and "start fit_fit_m5_R0" in second
    assert "start tables_verdict" in second and "still failing: 0" in second
    st = q.status(queue)
    assert not st["failed"] and [p["pending"] for p in st["phases"]] == [0, 0, 0, 0]
