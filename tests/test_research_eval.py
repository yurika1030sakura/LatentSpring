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


def test_xtb_uses_provided_triplet_instead_of_minimum_parity(monkeypatch,tmp_path):
    directory=Path(__file__).resolve().parents[1]/'scripts/research'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('eval_position_xtb',directory/'eval_position_xtb.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    commands=[]
    def invoke(command,*args):
        commands.append(command);return {'ok':False,'failure':'test_stop'}
    monkeypatch.setattr(module,'invoke',invoke)
    task={'symbols':['O','O'],'positions':[[0,0,0],[0,0,1.2]],'charge_recorded':0,
          'spin':3,'arm':'test','validation_index':0,'sample_id':0}
    result=module.evaluate(task,'xtb',tmp_path,20)
    assert commands[0][commands[0].index('--uhf')+1]=='2'
    assert result['spin_metadata_available']


def test_work_panel_selection_and_failure_accounting(monkeypatch):
    directory=Path(__file__).resolve().parents[1]/'scripts/research'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('assess_work_panel',directory/'assess_work_panel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selected=module.select_indices(64,32,9059)
    assert len(set(selected))==32 and selected==module.select_indices(64,32,9059)
    with pytest.raises(ValueError):module.select_indices(16,32,9059)
    left=[{'sample_id':i,'success':i%2==0,'strain_eV':3.} for i in range(4)]
    right=[{'sample_id':i,'success':i<2,'strain_eV':2.} for i in range(4)]
    result=module.paired_outcomes(left,right)
    assert (result['better'],result['worse'],result['both_failed'],result['tie'])==(2,1,1,0)
    assert result['attempted_pairs']==4
    with pytest.raises(ValueError):module.paired_outcomes(left,right[:-1])


def test_fm_assessment_does_not_fabricate_importance_weights(monkeypatch,tmp_path):
    import json
    import sys
    import torch
    directory=Path(__file__).resolve().parents[1]/'scripts/research'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('assess_work_panel',directory/'assess_work_panel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'results.json').write_text(json.dumps({'complete':True}))
    sample=tmp_path/'samples.pt'
    torch.save({'condition':{'numbers':[1,1],'charge':0,'spin_multiplicity':1},
        'positions':torch.tensor([[[0.,0,0],[0,0,.7]],[[0.,0,0],[0,0,1.2]]])},sample)
    second=tmp_path/'second.pt';data=torch.load(sample,weights_only=False)
    data['condition']['reference_geometry_loaded']=False;torch.save(data,second)
    binary=tmp_path/'xtb';binary.write_text('test binary identity')
    monkeypatch.setattr(module.shutil,'which',lambda name:str(binary))
    def failed(task,*args):return {**task,'success':False,'failure':'test_failure'}
    monkeypatch.setattr(module,'evaluate',failed)
    monkeypatch.setattr(sys,'argv',['assess_work_panel','--samples','fm',str(sample),'--samples','other',str(second),
        '--out',str(tmp_path/'out'),'--xtb-count','2'])
    module.main();report=json.loads((tmp_path/'out/assessment.json').read_text())
    assert report['complete'] and report['geometry'][0]['weights'] is None
    assert report['geometry'][0]['weighted_profile_variance_A2'] is None
    assert report['xtb_summaries'][0]['attempted']==2 and report['xtb_summaries'][0]['converged']==0
    assert report['sources'][1]['condition']['reference_geometry_loaded'] is False
