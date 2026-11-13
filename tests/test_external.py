"""File mode (gnssdl.bench.external): a filter run outside Python, through
files, gets the scores the same filter gets when run inside."""
import json

import numpy as np
import pandas as pd
import pytest

import gnssdl
from gnssdl import dataset
from gnssdl.bench import external, registry

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
    net.steps = steps
    return net


def _network_mean_csv(work):
    """The `external` filter: per day and component, subtract the mean of all
    stations (what C1 does), reading and writing the csv layout."""
    for d in sorted((work / "inputs").iterdir()):
        (work / "outputs" / d.name).mkdir(parents=True)
        for c in "enu":
            x = pd.read_csv(d / f"{c}.csv", index_col=0)
            x.sub(x.mean(axis=1), axis=0).to_csv(work / "outputs" / d.name / f"{c}.csv", float_format="%.4f")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("external")
    net = _network(tmp)
    work = tmp / "work"
    notes = []
    manifest = external.export(net, "outside", work, measures=("transients", "seasonal"), centres=1,
                               log=notes.append)
    return net, work, manifest, notes


def test_export_writes_every_dataset_and_nothing_else(exported):
    net, work, manifest, notes = exported
    assert json.loads((work / "manifest.json").read_text()) == manifest
    keys = [d["key"] for d in manifest["datasets"]]
    assert len(keys) == len(set(keys)) > 10 and sorted(p.name for p in (work / "inputs").iterdir()) == sorted(keys)
    assert {d["for"] for d in manifest["datasets"]} == {"transients"}          # seasonal reuses the plain series
    assert manifest["name"] == "outside" and manifest["centres"] == 1 and manifest["output"] == "filtered"
    assert notes[:2] == [f"transients: {len(keys)} new datasets", "seasonal: 0 new datasets"]
    assert f"{len(keys)} datasets written" in notes[-1]
    assert not (net.results / "outside").exists() and registry.find("outside") is None     # nothing scored yet
    table = pd.read_csv(work / "inputs" / keys[0] / "e.csv", index_col=0)
    assert list(table.columns) == list(net.cube.sta) and len(table) == manifest["datasets"][0]["days"]
    assert list(pd.read_csv(work / "stations.csv").columns) == ["sta", "lat", "lon"]
    assert "outputs/" in (work / "README.txt").read_text()
    status = external.check(work)
    assert len(status) == len(keys) and not status.output.any()
    with pytest.raises(gnssdl.TaskError, match="outputs are missing"):
        external.score(work, net)
    with pytest.raises(gnssdl.TaskError, match="already holds an export"):
        external.export(net, "outside", work)
    with pytest.raises(gnssdl.TaskError, match="built-in or registered"):
        external.export(net, "fs", work.parent / "other")


def test_a_filter_run_through_files_scores_like_the_same_filter_run_inside(exported):
    net, work, manifest, _ = exported
    _network_mean_csv(work)
    assert external.check(work).output.all()
    card = external.score(work, net)
    inside = gnssdl.audit("c1", net, measures=("transients", "seasonal"), centres=1)
    assert registry.find("outside") is None                         # the stand-in is gone again
    mine = card.measures[card.measures.model == "outside"].reset_index(drop=True)
    c1 = inside.measures[inside.measures.model == "c1"].reset_index(drop=True)
    assert list(mine.measure) == list(c1.measure) and len(mine) >= 9 and "error" not in set(mine.measure)
    assert np.allclose(mine.value, c1.value, atol=2e-3)             # the files round to 0.0001 mm
    assert (mine.n == c1.n).all()
    t = card.table().set_index("model")
    assert t.loc["outside", "radius_km"] == 0.0 and t.loc["fs", "verdict"] == "reference"
    assert card.verdict() in ("better", "not better", "noise gain bought with signal")
    assert (net.results / "outside" / "signal" / "validation" / "injections.csv").exists()


def test_score_names_what_is_wrong_with_an_output(exported):
    net, work, manifest, _ = exported
    key = manifest["datasets"][0]["key"]
    path = work / "outputs" / key / "n.csv"
    good = path.read_text()
    try:
        frame = pd.read_csv(path, index_col=0)
        frame.iloc[5:8, 3] = np.nan
        frame.to_csv(path)
        with pytest.raises(gnssdl.TaskError, match="have none in the\\s+output|have none in the output"):
            external.score(work, net)
        frame.drop(columns=frame.columns[0]).to_csv(path)
        with pytest.raises(gnssdl.TaskError, match="stations of the input are not columns"):
            external.score(work, net)
        frame.iloc[:-3].to_csv(path)
        with pytest.raises(gnssdl.TaskError, match="rows must be the input"):
            external.score(work, net)
    finally:
        path.write_text(good)


def test_npz_format_and_common_mode_output(exported, tmp_path):
    net, _, _, _ = exported
    work = tmp_path / "npz"
    m = external.export(net, "outside_npz", work, measures=("noise",), fmt="npz", output="common_mode")
    assert len(m["datasets"]) == 1                                  # the noise measure cleans one series
    for path in (work / "inputs").glob("*.npz"):
        with np.load(path) as z:
            r = z["r"]
            assert r.shape[0] == len(z["sta"]) and r.shape[1] == len(z["days"])
        (work / "outputs").mkdir(exist_ok=True)
        np.savez(work / "outputs" / path.name, r=np.broadcast_to(np.nanmean(r, axis=0), r.shape))
    card = external.score(work, results=tmp_path / "runs")           # the network is read from the manifest
    inside = gnssdl.audit("c1", net, measures=("noise",))
    value = lambda c, name: float(c.measures[(c.measures.model == name) & (c.measures.measure == "noise_removed")].value.iloc[0])
    assert value(card, "outside_npz") == pytest.approx(value(inside, "c1"), abs=1e-4)
    assert card.verdicts is None and card.verdict() == "no signal test"
