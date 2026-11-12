"""Region files: what is specific to one study area (gnssdl.regions)."""
import json

import pytest

from gnssdl import regions


def test_the_pnw_region_carries_the_cascadia_episode_list():
    r = regions.get("pnw")
    assert regions.names() == ["pnw"] and r.name == "pnw" and r.bbox == (42.5, 49.5, -125.5, -116.0)
    assert r.has_episodes and r.exclude_episodes_from_training and r.training_exclusion() is not None
    episodes, band = r.load_episodes(), r.load_band()
    assert set(episodes.timing) == {"gnss", "tremor"} and list(band.columns) == ['lat', 'lon']
    assert r.catalogue_path.name == "cascadia_gnss_catalogue.csv"
    from gnssdl.bench import episodes as eps
    assert eps.EPISODES == r.episodes_path and eps.BAND == r.band_path     # the builder writes the region's files


def test_a_region_file_of_your_own(tmp_path):
    (tmp_path / "ep.csv").write_text("id,source,timing,start,end,lat_min,lat_max,note\n")
    (tmp_path / "band.csv").write_text("lat,lon\n")
    plain = tmp_path / "plain.toml"
    plain.write_text('name = "plain"\n')
    r = regions.get(plain)
    assert r.title == "plain" and not r.has_episodes and r.training_exclusion() is None
    with pytest.raises(regions.RegionError, match="no slow-slip episode list"):
        r.load_episodes()
    full = tmp_path / "full.toml"
    full.write_text('name = "full"\ntitle = "Full"\nbbox = [1, 2, 3, 4]\n'
                    '[episodes]\nlist = "ep.csv"\nband = "band.csv"\n')
    r = regions.get(str(full))
    assert r.has_episodes and r.bbox == (1.0, 2.0, 3.0, 4.0) and not r.exclude_episodes_from_training
    assert r.training_exclusion() is None and len(r.load_episodes()) == 0
    for text, why in (('title = "x"\n', "name"), ('name = "x"\nbox = 1\n', "unknown keys"),
                      ('name = "x"\nbbox = [2, 1, 3, 4]\n', "bbox"),
                      ('name = "x"\n[episodes]\nlist = "ep.csv"\n', "both"),
                      ('name = "x"\n[episodes]\nlist = "nope.csv"\nband = "band.csv"\n', "no such file"),
                      ('name = "x"\n[episodes]\nlist = "ep.csv"\nband = "band.csv"\nfar = 3\n', "unknown")):
        bad = tmp_path / "bad.toml"
        bad.write_text(text)
        with pytest.raises(regions.RegionError, match=why):
            regions.get(bad)
    with pytest.raises(regions.RegionError, match="no built-in region"):
        regions.get("atlantis")
    with pytest.raises(regions.RegionError, match="no region file"):
        regions.get(tmp_path / "missing.toml")


def test_a_cube_finds_its_region(tmp_path):
    assert regions.for_cube(tmp_path / "pnw.npz").name == "pnw"            # by the cube file's name
    assert regions.for_cube(tmp_path / "toy.npz") is None
    (tmp_path / "toy.json").write_text(json.dumps({"region": "pnw"}))      # as recorded by `gnssdl build`
    assert regions.for_cube(tmp_path / "toy.npz").name == "pnw"
    (tmp_path / "pnw.json").write_text(json.dumps({"region": None}))       # built with --region none
    assert regions.for_cube(tmp_path / "pnw.npz") is None
    (tmp_path / "old.json").write_text(json.dumps({"stations_in": 3}))     # a summary from before regions
    assert regions.for_cube(tmp_path / "old.npz") is None
