"""`gnssdl.audit`: an outside filter through the whole benchmark in one call,
on a small synthetic network (no downloads, validation period only)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gnssdl
from gnssdl import dataset
from gnssdl.bench import registry

N = 36


def _network(tmp_path):
    """Stations within ~200 km sharing a 3 mm common mode, 1 mm own noise."""
    rng = np.random.default_rng(5)
    days = pd.date_range("2008-01-01", "2025-12-31", freq="D")
    t = days.year + (days.dayofyear - 0.5) / np.where(days.is_leap_year, 366, 365)
    common = rng.normal(0, 0.003, (len(days), 3))
    st = pd.DataFrame({"sta": [f"A{i:03d}" for i in range(N)],
                       "lat": 36.0 + rng.uniform(0, 1.5, N), "lon": -120.0 + rng.uniform(0, 1.5, N)})

    def load(sta):
        x = common + np.random.default_rng(int(sta[1:])).normal(0, 0.001, (len(days), 3))
        return pd.DataFrame({"decyear": t, "e": x[:, 0], "n": x[:, 1], "u": x[:, 2],
                             "sig_e": 0.001, "sig_n": 0.001, "sig_u": 0.003}, index=days)

    steps = pd.DataFrame(columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    steps["date"] = pd.to_datetime(steps["date"])
    cube, _ = dataset.build_cube(st, load, steps)
    path = tmp_path / "cube" / "toy" / "toy.npz"
    path.parent.mkdir(parents=True)
    cube.save(path)
    net = gnssdl.Network.load("toy", data_dir=tmp_path, results=tmp_path / "runs")
    net.steps = steps                      # our own step catalogue: nothing is downloaded
    return net


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("median_filter", Path("examples/median_filter.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module.FarMedian
    registry.unregister("farmedian")


@pytest.fixture(scope="module")
def audited(example, tmp_path_factory):
    net = _network(tmp_path_factory.mktemp("card"))
    notes = []
    card = gnssdl.audit(example, net, centres=1, log=notes.append)
    return net, card, notes


def test_audit_runs_an_outside_filter_end_to_end(audited):
    net, card, notes = audited
    assert net.masks_path.exists() and any("generating them" in n for n in notes)
    assert any("the reference" in n and "running it" in n for n in notes)
    assert card.table().seasonal.notna().all()                                       # the reference gets every measure too
    assert (net.results / "farmedian" / "scores" / "farmedian.json").exists()       # leak checks ran and passed
    t = card.table()
    assert list(t.model) == ["farmedian", "fs"] and list(t.radius_km) == [0.0, 0.0]
    assert {"noise_removed", "error", "kept_100", "spurious_100", "seasonal", "verdict"} <= set(t.columns)
    mine, ref = t.iloc[0], t.iloc[1]
    # 36 stations sharing one common mode: a far median and the far mean both remove most of it
    assert 0.5 < mine.noise_removed < 1 and 0.5 < ref.noise_removed < 1
    assert ref.verdict == "reference" and card.verdict() == mine.verdict
    assert mine.verdict in ("better", "not better", "noise gain bought with signal")
    m = card.measures
    assert (m.ci_low <= m.value + 1e-9).all() and (m.value <= m.ci_high + 1e-9).all()
    assert card.offsets is None and card.episodes is None                           # none in this network
    assert net.region is None and any("no slow-slip episode list" in n for n in notes)
    assert not (net.results / "farmedian" / "signal" / "validation" / "episodes.csv").exists()
    from gnssdl.bench import tasks
    with pytest.raises(gnssdl.TaskError, match="episode measure is skipped"):
        tasks.run_signal(net, "fs", measure="episodes", radii=[0.0])
    text = str(card)
    assert "farmedian, validation period" in text and "success criterion" in text


def test_a_second_audit_reuses_saved_results(audited, example):
    net, card, _ = audited
    notes = []
    again = gnssdl.audit(example, net, centres=1, log=notes.append)
    assert not any("injections" in n or "leak checks" in n or "noise removed" in n for n in notes)
    pd.testing.assert_frame_equal(again.table(), card.table())
    assert gnssdl.audit("farmedian", net, centres=1).verdict() == card.verdict()    # by name, once registered


def test_audit_refuses_what_it_cannot_score(audited, monkeypatch):
    net, _, _ = audited

    class Undeclared(gnssdl.Filter):
        name = "undeclared"

        def predict(self, cube, hide):
            return np.zeros(cube.r.shape, dtype=np.float32)

    with pytest.raises(gnssdl.TaskError, match="never_reads_target"):
        gnssdl.audit(Undeclared, net)
    registry.unregister("undeclared")

    class Peeks(gnssdl.Filter):
        """Claims not to read the target, but returns its values."""
        name = "peeks"
        never_reads_target = True

        def predict(self, cube, hide):
            return np.nan_to_num(np.where(cube.avail[..., None], cube.r, 0.0)).astype(np.float32)

    with pytest.raises(RuntimeError, match="failed leak checks"):
        gnssdl.audit(Peeks, net, centres=1)
    registry.unregister("peeks")

    class Other(gnssdl.Filter):
        name = "fs"
        never_reads_target = True

    with pytest.raises(gnssdl.TaskError, match="already registered"):
        gnssdl.audit(Other, net)
    with pytest.raises(gnssdl.TaskError, match="no checkpoint"):
        gnssdl.audit("m3k", net)                                                    # never trains by itself
    with pytest.raises(ValueError, match="unknown measures"):
        gnssdl.audit("fs", net, measures=("speed",))
    with pytest.raises(gnssdl.TaskError, match="no cube"):
        gnssdl.audit("fs", "nowhere")
    monkeypatch.chdir(net.results.parent)                                           # no freeze record here
    from gnssdl.bench.freeze import NotFrozen
    with pytest.raises(NotFrozen, match="refused"):
        gnssdl.audit("fs", net, period="test")


def test_status_lists_what_exists_from_the_files_alone(audited, capsys):
    from gnssdl.bench import tasks
    from gnssdl.cli import main
    net, _, _ = audited
    t = tasks.status(net).set_index(["model", "radius_km"])
    fs0, mine = t.loc[("fs", 0.0)], t.loc[("farmedian", 0.0)]
    for row in (fs0, mine):
        assert row.fitted and row.noise and row.transients and row.seasonal
        assert not row.offsets and row.episodes is None            # no earthquakes; no region, so no episode list
    assert not t.loc[("fs", 25.0)].noise                           # the audit ran at R = 0 only
    assert not t.loc[("m3k", 0.0)].fitted and not t.loc[("m1", 0.0)].fitted      # no checkpoint, no tuning
    assert {"c3-k1p1", "c3-k8p3"} <= set(t.index.get_level_values("model"))
    assert main(["bench", "status", "--cube", str(net.cube_path), "--results", str(net.results), "--missing"]) == 0
    out = capsys.readouterr().out
    assert "rows (filter, seed, radius)" in out and "NO" in out and "m3k" in out
