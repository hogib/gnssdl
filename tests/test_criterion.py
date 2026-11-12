"""The noise-removed bootstrap and the success criterion (contract §1.6, 2026-11-12)."""
import numpy as np
import pandas as pd
import pytest

from gnssdl.bench import stats


def _blocks(fraction_left, stations=40, blocks=24, seed=0, jitter=0.0):
    """Block sums for a filter that leaves `fraction_left` of the fast rms
    (noise removed = 1 - fraction_left); `jitter` spreads the stations evenly
    around it."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(20.0, 40.0, (stations, blocks, 3))
    left = fraction_left + jitter * np.linspace(-1, 1, stations).reshape(-1, 1, 1)   # median unchanged
    clean = raw * left**2
    rows = [{"sta": f"S{i:02d}", "block": b, "days": 30,
             **{f"clean_{c}": clean[i, b, k] for k, c in enumerate("enu")},
             **{f"raw_{c}": raw[i, b, k] for k, c in enumerate("enu")}}
            for i in range(stations) for b in range(blocks)]
    return pd.DataFrame(rows)


def test_paired_noise_bootstrap_recovers_a_known_difference():
    a, b = _blocks(0.80), _blocks(0.70)
    assert stats.noise_score(a) == pytest.approx(0.20) and stats.noise_score(b) == pytest.approx(0.30)
    res = stats.paired_noise_bootstrap(a, b, n_boot=200)
    assert res["diff"] == pytest.approx(0.10) and res["ci_low"] > 0 and res["stations"] == 40 and res["blocks"] == 24
    same = stats.paired_noise_bootstrap(a, a, n_boot=200)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    with pytest.raises(ValueError):
        stats.paired_noise_bootstrap(a, a.assign(sta="other"))


def _noise(model, radius, fraction_left, **kw):
    return _blocks(fraction_left, **kw).assign(model=model, radius_km=radius)


def _inj(model, radius, kept, spurious, n=30):
    rows = [{"model": model, "radius_km": radius, "id": k, "footprint_km": L, "rho": v, "spurious_max": spurious}
            for L, v in ((25.0, 0.0), (100.0, kept)) for k in range(n)]
    return pd.DataFrame(rows)


def test_criterion_table_gives_each_verdict():
    noise = pd.concat([
        _noise("fs", 0.0, 0.75), _noise("fs", 100.0, 0.76),            # removes less
        _noise("good", 0.0, 0.70), _noise("good", 25.0, 0.68), _noise("tie", 0.0, 0.68, jitter=0.03),
        _noise("greedy", 0.0, 0.60), _noise("ghost", 0.0, 0.60),
        _noise("c1", np.nan, 0.65), _noise("untested", 0.0, 0.60)], ignore_index=True)
    inj = pd.concat([
        _inj("fs", 0.0, 0.80, 0.10), _inj("fs", 100.0, 0.95, 0.12),
        _inj("good", 0.0, 0.78, 0.12), _inj("good", 25.0, 0.76, 0.14), _inj("tie", 0.0, 0.80, 0.10),
        _inj("greedy", 0.0, 0.40, 0.10),                                # loses the signal
        _inj("ghost", 0.0, 0.80, 0.30),                                 # makes spurious signal
        _inj("c1", np.nan, 0.74, 0.16)], ignore_index=True)             # just outside both tolerances
    t = stats.criterion_table(noise, inj, n_boot=200)
    v = {(m, -1.0 if pd.isna(r) else r): x for m, r, x in zip(t.model, t.radius_km, t.verdict)}
    assert v == {("fs", 0.0): "reference", ("fs", 100.0): "not better",
                 ("good", 0.0): "better", ("good", 25.0): "better", ("tie", 0.0): "better",
                 ("greedy", 0.0): "noise gain bought with signal", ("ghost", 0.0): "noise gain bought with signal",
                 ("c1", -1.0): "noise gain bought with signal", ("untested", 0.0): "no signal test"}
    row = lambda m, r: t[(t.model == m) & (t.radius_km == r)].iloc[0]
    assert not row("greedy", 0.0).kept_ok and row("greedy", 0.0).spurious_ok
    assert row("ghost", 0.0).kept_ok and not row("ghost", 0.0).spurious_ok
    assert t.best.sum() == 1
    best = t[t.best].iloc[0]
    assert best.model in ("good", "tie") and best.noise_removed == pytest.approx(0.32, abs=0.01)
    other = "tie" if best.model == "good" else "good"
    assert row(other, 0.0 if other == "tie" else 25.0).same_as_best          # 0.32 vs 0.32 with jitter
    assert not row("good", 0.0).same_as_best                                  # 0.30 is distinguishable


def test_criterion_table_marks_nothing_when_no_filter_passes_and_needs_the_reference():
    noise = pd.concat([_noise("fs", 0.0, 0.75), _noise("greedy", 0.0, 0.60)], ignore_index=True)
    inj = pd.concat([_inj("fs", 0.0, 0.80, 0.10), _inj("greedy", 0.0, 0.40, 0.10)], ignore_index=True)
    t = stats.criterion_table(noise, inj, n_boot=100)
    assert not t.best.any() and not t.same_as_best.any()
    with pytest.raises(ValueError, match="reference"):
        stats.criterion_table(noise[noise.model != "fs"], inj, n_boot=10)
    with pytest.raises(ValueError, match="reference"):
        stats.criterion_table(noise, inj[inj.model != "fs"], n_boot=10)


def test_verdict_command_reads_saved_results_only(tmp_path, capsys):
    from gnssdl.cli import main
    for model, left, kept in (("fs", 0.75, 0.80), ("m1", 0.70, 0.79)):
        d = tmp_path / model / "signal" / "validation"
        d.mkdir(parents=True)
        _noise(model, 0.0, left).to_csv(d / "noise_blocks.csv", index=False)
        _inj(model, 0.0, kept, 0.10).to_csv(d / "injections.csv", index=False)
    assert main(["bench", "verdict", "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "best: m1 at R = 0 km" in out and "reference" in out
    t = pd.read_csv(tmp_path / "verdict_validation.csv")
    assert set(t.verdict) == {"reference", "better"}
    assert main(["bench", "verdict", "--results", str(tmp_path / "empty")]) == 1


# --------------------------------------------------------------------------- #
# Intervals for every measure
# --------------------------------------------------------------------------- #

def test_interval_table_brackets_each_measure():
    noise = _noise("m1", 0.0, 0.70, jitter=0.05)
    noise = noise.assign(err_days=30, err_e=30 * 0.25, err_n=30 * 0.25, err_u=30 * 0.25)   # error 0.5 everywhere
    c1 = _noise("c1", np.nan, 0.60).assign(err_days=0, err_e=0.0, err_n=0.0, err_u=0.0)   # target-inclusive
    rng = np.random.default_rng(0)
    inj = _inj("m1", 0.0, 0.8, 0.1).assign(rho=lambda d: d.rho + rng.normal(0, 0.05, len(d)))
    sea = pd.DataFrame({"model": "m1", "radius_km": 0.0, "sta": [f"S{i}" for i in range(50)],
                        "rho_all": np.linspace(0.0, 0.2, 50)})
    t = stats.interval_table(pd.concat([noise, c1]), inj, sea, n_boot=200)
    assert (t.ci_low <= t.value + 1e-12).all() and (t.value <= t.ci_high + 1e-12).all()
    get = lambda model, measure, L=None: t[(t.model == model) & (t.measure == measure)
                                           & (t.footprint_km.isna() if L is None else t.footprint_km == L)].iloc[0]
    assert get("m1", "noise_removed").value == pytest.approx(0.30, abs=0.01) and get("m1", "noise_removed").n == 40
    assert get("m1", "noise_removed").ci_high > get("m1", "noise_removed").ci_low          # the jitter gives a width
    assert get("m1", "error").value == pytest.approx(0.5) and get("m1", "error").ci_low == pytest.approx(0.5)
    assert get("m1", "kept", 100.0).value == pytest.approx(0.8, abs=0.03) and get("m1", "kept", 100.0).n == 30
    assert get("m1", "spurious", 100.0).value == pytest.approx(0.1)
    assert get("m1", "seasonal").value == pytest.approx(0.1, abs=0.01)
    assert set(t[t.model == "c1"].measure) == {"noise_removed"}                            # no error, no signal files
    assert pd.isna(get("c1", "noise_removed").radius_km)


def test_intervals_command_reads_saved_results_only(tmp_path, capsys):
    from gnssdl.cli import main
    d = tmp_path / "m1" / "signal" / "validation"
    d.mkdir(parents=True)
    _noise("m1", 0.0, 0.7).to_csv(d / "noise_blocks.csv", index=False)       # blocks saved without the error columns
    _inj("m1", 0.0, 0.8, 0.1).to_csv(d / "injections.csv", index=False)
    assert main(["bench", "intervals", "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "noise_removed" in out and "kept" in out and "0.300 [" in out
    t = pd.read_csv(tmp_path / "intervals_validation.csv")
    assert set(t.measure) == {"noise_removed", "kept", "spurious"}
    assert main(["bench", "intervals", "--results", str(tmp_path / "empty")]) == 1


# --------------------------------------------------------------------------- #
# The freeze record
# --------------------------------------------------------------------------- #

def _git(root, *args):
    import subprocess
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org",
                    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args], check=True, capture_output=True)


def test_freeze_record_is_needed_for_any_period_but_validation(tmp_path):
    from gnssdl.bench import freeze
    freeze.require("validation", tmp_path)                          # always allowed
    with pytest.raises(freeze.NotFrozen, match="not frozen"):
        freeze.require("test", tmp_path)
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A"); _git(tmp_path, "commit", "-q", "-m", "one")
    (tmp_path / "FREEZE").write_text("frozen by the author\n")
    with pytest.raises(freeze.NotFrozen, match="tag:"):
        freeze.require("test", tmp_path)
    (tmp_path / "FREEZE").write_text("tag: freeze-x\n")
    with pytest.raises(freeze.NotFrozen, match="does not exist"):
        freeze.require("prospective", tmp_path)
    _git(tmp_path, "tag", "freeze-x")
    freeze.require("test", tmp_path)                                # frozen: allowed
    assert freeze.frozen_tag(tmp_path) == "freeze-x"
    (tmp_path / "src" / "a.py").write_text("x = 2\n")               # the code changed after the tag
    with pytest.raises(freeze.NotFrozen, match="differs from the frozen tag"):
        freeze.require("test", tmp_path)


@pytest.mark.parametrize("argv", [
    ["bench", "signal", "fs", "--period", "test"],
    ["bench", "signal", "fs", "--period", "prospective"],
    ["bench", "run", "m1", "--period", "all"],
    ["bench", "compare", "fs", "m1", "--period", "test"],
    ["bench", "report", "--split", "test"],
    ["bench", "verdict", "--period", "test"],
    ["bench", "intervals", "--period", "test"],
    ["figures", "--period", "test"],
])
def test_commands_refuse_the_test_period_before_the_freeze(argv, tmp_path, monkeypatch, capsys):
    from gnssdl.cli import main
    monkeypatch.chdir(tmp_path)                                     # a directory with no freeze record
    assert main(argv) == 1
    assert "refused" in capsys.readouterr().err


def test_tradeoff_figure_is_drawn_from_saved_results_of_any_network(tmp_path, capsys):
    from gnssdl.cli import main
    for model, radii in (("fs", (0.0, 100.0)), ("m6r16aug", (0.0,)), ("c1", (np.nan,))):
        d = tmp_path / "runs" / model / "signal" / "validation"
        d.mkdir(parents=True)
        pd.DataFrame({"model": model, "radius_km": radii, "noise_removed": 0.25, "stations": 40}).to_csv(
            d / "noise.csv", index=False)
        pd.concat([_inj(model, r, 0.8, 0.1) for r in radii]).to_csv(d / "injections.csv", index=False)
    out = tmp_path / "fig"
    assert main(["figures", "--only", "tradeoff", "--results", str(tmp_path / "runs"), "-o", str(out)]) == 0
    assert (out / "tradeoff_widths.pdf").stat().st_size > 2000
    assert main(["figures", "--only", "tradeoff", "--results", str(tmp_path / "none"), "-o", str(out)]) == 1



# --------------------------------------------------------------------------- #
# Pairing of planted transients across filters
# --------------------------------------------------------------------------- #

def test_paired_rho_bootstrap_checks_that_ids_name_the_same_transient():
    base = pd.DataFrame({"id": range(40), "footprint_km": 100.0, "amplitude_mm": 5.0, "duration_days": 30,
                         "lat": np.linspace(44, 48, 40), "lon": -122.0, "rho": 0.8})
    other = base.assign(rho=0.6)
    t = stats.paired_rho_bootstrap(base, other, n_boot=100)
    assert t.pairing.item() == "checked" and t["diff"].item() == pytest.approx(-0.2)
    shuffled = other.assign(lat=other.lat.to_numpy()[::-1])                 # same ids, different transients
    with pytest.raises(ValueError, match="different transients"):
        stats.paired_rho_bootstrap(base, shuffled, n_boot=100)
    old = stats.paired_rho_bootstrap(base.drop(columns=["lat", "lon"]), other, n_boot=100)
    assert old.pairing.item() == "unverified"                               # rows without a centre cannot be checked


# --------------------------------------------------------------------------- #
# the same tasks called from Python (gnssdl.bench.tasks), without the command line

def _saved(tmp_path):
    for model, left, kept in (("fs", 0.75, 0.80), ("m1", 0.70, 0.79)):
        d = tmp_path / model / "signal" / "validation"
        d.mkdir(parents=True)
        _noise(model, 0.0, left).to_csv(d / "noise_blocks.csv", index=False)
        _inj(model, 0.0, kept, 0.10).assign(amplitude_mm=2.0).to_csv(d / "injections.csv", index=False)


def test_verdict_and_intervals_are_callable_from_python(tmp_path):
    from gnssdl.bench import tasks
    _saved(tmp_path)
    t, path = tasks.verdict(tmp_path, write=False)
    assert set(t.verdict) == {"reference", "better"} and not path.exists()
    t, path = tasks.verdict(tmp_path)
    assert path == tmp_path / "verdict_validation.csv" and len(pd.read_csv(path)) == len(t)
    notes = []
    iv, path = tasks.intervals(tmp_path, log=notes.append)
    assert path.exists() and {"noise_removed", "kept", "spurious"} <= set(iv.measure)
    assert notes == ["(no seasonal.csv: its measures are left out)"]
    tables = tasks.signal_tables(tmp_path)
    assert set(tables) == {"kept", "spurious", "kept_by_amplitude"}
    assert list(tables["kept"].index.names) == ["model", "R_km"]


def test_tasks_say_what_is_missing_and_refuse_the_test_period(tmp_path, monkeypatch):
    from gnssdl.bench import tasks
    from gnssdl.bench.freeze import NotFrozen
    with pytest.raises(tasks.TaskError, match="noise_blocks.csv"):
        tasks.verdict(tmp_path / "empty")
    with pytest.raises(tasks.TaskError, match="no validation signal results"):
        tasks.intervals(tmp_path / "empty")
    with pytest.raises(tasks.TaskError, match="no training seed"):
        tasks.check_seed("m1", 1)
    monkeypatch.chdir(tmp_path)                                     # a directory with no freeze record
    for call in (lambda: tasks.verdict(tmp_path, "test"), lambda: tasks.intervals(tmp_path, "prospective"),
                 lambda: tasks.signal_tables(tmp_path, "test")):
        with pytest.raises(NotFrozen, match="refused"):
            call()
