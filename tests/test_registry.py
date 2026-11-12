"""The filter registry: built-in and outside filters are registered the same way."""
import numpy as np
import pytest

from gnssdl.bench import registry
from gnssdl.bench.base import Reconstructor
from gnssdl.bench.run import model_radii, variant_label


class _Cube:
    radii = np.array([0.0, 25.0, 50.0])


class Outside(Reconstructor):
    name = "outside"

    def predict(self, cube, hide):
        return np.zeros(cube.r.shape, dtype=np.float32)


@pytest.fixture
def outside():
    yield
    registry.unregister("outside")
    registry.unregister("outside2")


def test_builtin_filters_keep_their_names_radii_and_roles():
    assert registry.names(masks=False) == ["m2self", "c1", "c3"]
    assert {"m0", "m1", "fs", "d1", "m3k", "m5aug", "m6r16aug"} <= set(registry.names(masks=True))
    assert set(registry.names()) == set(registry.names(masks=True)) | set(registry.names(masks=False))
    assert model_radii("m0", _Cube) == [0.0] and model_radii("c1", _Cube) == [0.0]
    assert model_radii("d1", _Cube) == [400.0] and model_radii("m2self", _Cube, [50.0]) == [0.0]
    assert model_radii("fs", _Cube) == [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 600.0]
    assert model_radii("m6r", _Cube) == [0.0, 10.0, 25.0, 50.0, 100.0]
    assert model_radii("m1", _Cube) == [0.0, 25.0, 50.0] and model_radii("m6r16", _Cube, [25.0]) == [25.0]
    assert registry.get("m2self").tuning_source == "m2" and registry.get("c1").tuning_source is None
    assert registry.get("m2").tuning_source == "m2"
    assert registry.get("m5").has_seed and registry.get("m6r16aug").has_seed and not registry.get("m1").has_seed
    assert len(registry.get("c3").sweep) == 9 and variant_label("c3", registry.get("c3").sweep[4]) == "c3-k4p2"
    assert registry.get("fs").title == "far stack" and registry.get("m0").title == "m0"


def test_an_outside_filter_registers_like_a_builtin_one(outside):
    spec = registry.register(Outside, title="Outside filter", radii=(0, 50))
    assert registry.get("outside") is spec and "outside" in registry.names(masks=True)
    assert spec.radii == (0.0, 50.0) and not spec.reads_target and not spec.has_seed
    assert model_radii("outside", _Cube) == [0.0, 50.0] and variant_label("outside", {}) == "outside"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Outside)
    registry.register(Outside, replace=True, radius_free=True)
    assert model_radii("outside", _Cube) == [0.0]
    swept = registry.register(Outside, name="outside2", sweep=[{"width": 1}, {"width": 2.5}])
    assert [variant_label("outside2", kw) for kw in swept.sweep] == ["outside2-width1", "outside2-width2.5"]


def test_register_refuses_what_would_break_results_files(outside):
    for bad in ("My-Filter", "a b", "base", "m5-s1"):
        with pytest.raises(ValueError, match="filter name"):
            registry.register(Outside, name=bad)
    with pytest.raises(TypeError):
        registry.register(dict, name="outside")
    with pytest.raises(ValueError, match="family"):
        registry.register(Outside, family="deep")
    with pytest.raises(ValueError, match="radius-free"):
        registry.register(Outside, radius_free=True, radii=(0.0,))
    with pytest.raises(KeyError, match="no filter named"):
        registry.get("nope")
    assert registry.find("nope") is None


def test_entry_points_register_filters_once(outside, monkeypatch):
    calls = []

    class EP:
        def load(self):
            return lambda: calls.append(registry.register(Outside))

    monkeypatch.setattr(registry, "entry_points", lambda group: [EP()] if group == "gnssdl.filters" else [])
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    assert "outside" in registry.names() and registry.names() and len(calls) == 1
