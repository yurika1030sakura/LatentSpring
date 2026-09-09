"""Independent-evaluator parsing must not turn coordinates into large forces."""
import importlib.util
import math
from pathlib import Path
import pytest


def test_xtb_gradient_excludes_coordinates_and_converts_atomic_units(monkeypatch):
    directory=Path(__file__).resolve().parents[1]/'scripts/research'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('eval_position_xtb',directory/'eval_position_xtb.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    text='$grad\ncycle = 1\n100 200 300\n400 500 600\n0.01D+0 0.02D+0 0.02D+0\n0 0 0.001\n$end'
    assert module.gradient_norm(text,2)==pytest.approx(.03*module.HARTREE_EV/module.BOHR_A)
    with pytest.raises(ValueError,match='Incomplete'):
        module.gradient_norm('$grad\n0 0 0\n$end',2)
