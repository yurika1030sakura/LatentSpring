import importlib.util
from pathlib import Path
import random


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/research/audit_official_validation.py'
    spec=importlib.util.spec_from_file_location('validation_audit',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_candidate_reservoir_keeps_best_distinct_conditions_independent_of_order():
    cls=module().HashReservoir
    records=[(str(i%11),100-i,{'i':i}) for i in range(90)]
    expected={}
    for key,rank,row in records:
        if key not in expected or rank<expected[key][0]:expected[key]=(rank,row)
    expected=dict(sorted(expected.items(),key=lambda item:item[1][0])[:5])
    for seed in range(5):
        shuffled=list(records);random.Random(seed).shuffle(shuffled);reservoir=cls(5)
        for record in shuffled:reservoir.add(*record)
        assert reservoir.rows==expected
        assert len(reservoir.heap)<=15
