"""PyTorch is optional: only the learned filters import it, and they say how to get it."""
import subprocess
import sys

import pytest


def test_the_benchmark_imports_and_registers_every_filter_without_importing_torch():
    code = ("import sys; import gnssdl; from gnssdl.bench import card, registry, tasks; "
            "assert 'm6r16aug' in registry.names() and gnssdl.audit is card.audit; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_a_learned_filter_without_torch_says_which_extra_to_install(monkeypatch):
    from gnssdl.bench.base import torch_module
    from gnssdl.bench import m3k_kernel
    monkeypatch.setitem(sys.modules, "torch", None)            # makes `import torch` fail
    for call in (torch_module, m3k_kernel._torch):
        with pytest.raises(ImportError, match="learned"):
            call()
