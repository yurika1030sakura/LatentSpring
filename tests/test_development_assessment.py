import importlib.util
import json
from pathlib import Path
import sys

import torch


def test_generation_failure_is_not_counted_as_an_attempted_xtb_evaluation(tmp_path,monkeypatch):
    directory=Path(__file__).resolve().parents[1]/'scripts/research';monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('assess_development_panel',directory/'assess_development_panel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    baseline=tmp_path/'baseline';sampling=baseline/'sampling';sampling.mkdir(parents=True)
    condition={'atomic_numbers':[8,1,1],'charge':0,'spin_multiplicity':1}
    references={'complete':True,'source_manifest_sha256':'manifest','rows':[
        {'panel_index':i,'condition':condition,'positions':[[0,0,0],[1,0,0],[0,1,0]],'symbols':['O','H','H']} for i in range(2)]}
    (baseline/'references.json').write_text(json.dumps(references))
    target=sampling/'condition_01';target.mkdir();files=[]
    for steps in [16,64]:
        path=target/f'fm_midpoint_{steps}_samples.pt';torch.save({'positions':torch.ones(2,3,3)},path)
        files.append({'file':path.name,'sha256':module.sha(path)})
    detail=target/'results.json';detail.write_text(json.dumps({'samples':files}))
    panel={'complete':True,'manifest_sha256':'manifest','samples_per_condition':2,'successful_conditions':1,
        'rows':[{'panel_index':0,'condition':condition,'success':False,'error':'generation_nonfinite'},
                {'panel_index':1,'condition':condition,'success':True,'results_sha256':module.sha(detail)}]}
    (sampling/'panel.json').write_text(json.dumps(panel))
    binary=tmp_path/'xtb';binary.write_text('test binary')
    monkeypatch.setattr(module.shutil,'which',lambda name:str(binary))
    monkeypatch.setattr(module,'evaluate',lambda task,*args:{**task,'success':False,'failure':'test_failure'})
    out=tmp_path/'assessment';monkeypatch.setattr(sys,'argv',['assess_development_panel','--baseline',str(baseline),'--out',str(out)])
    module.main();report=json.loads((out/'assessment.json').read_text())
    assert report['complete'] and report['requested_xtb_tasks']==6 and len(report['rows'])==6
    assert report['sampling_failures'][0]['requested_samples_per_resolution']==2
    assert report['summaries'][0]['arms'][0]['attempted']==1
    assert report['summaries'][0]['arms'][1]['attempted']==0
    assert report['summaries'][1]['arms'][1]['attempted']==2
