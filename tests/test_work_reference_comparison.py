import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from cfm_mol.triatomic_reference import invariant_observables


def test_work_offset_is_removed_with_correct_sign_and_reference_is_checked(tmp_path,monkeypatch):
    directory=Path(__file__).resolve().parents[1]/'scripts/research';monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('compare_work_reference',directory/'compare_work_reference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=tmp_path/'run';reference=tmp_path/'reference';run.mkdir();reference.mkdir()
    condition={'numbers':[8,1,1],'charge':0,'spin_multiplicity':1}
    x=torch.randn(16,3,3,dtype=torch.float64,generator=torch.Generator().manual_seed(992));x-=x.mean(1,keepdim=True)
    (run/'results.json').write_text(json.dumps({'complete':True,'condition':condition,'oracle_sha256':'same_oracle',
        'configuration':{'kT':.25,'restraint':.1},'energy_zero_eV':-100.}))
    # log Z=20 and E_zero/kT=-400 imply constant shifted work +380.
    for stage in ['initial','final']:torch.save({'positions':x,'work':torch.full((16,),380.,dtype=torch.float64)},run/f'{stage}_samples.pt')
    moments={name:{'mean':float(value.mean()),'delta_method_se':.1} for name,value in invariant_observables(x.numpy()).items()}
    ref={'complete':True,'condition':condition,'oracle_sha256':'same_oracle','restraint_eV_A2':.1,
        'convergence_certified':False,'aggregate':[{'kT_eV':.25,'particles_per_scramble':4096,
            'log_of_mean_normalizer_estimate':20.,'relative_se_across_scrambles':.04,'moments':moments}]}
    (reference/'reference.json').write_text(json.dumps(ref));output=tmp_path/'comparison.json'
    argv=['compare_work_reference','--run',str(run),'--reference',str(reference),'--out',str(output)]
    monkeypatch.setattr(sys,'argv',argv);module.main();result=json.loads(output.read_text())
    assert not result['reference_convergence_certified']
    for row in result['rows']:
        assert row['log_normalizer_estimate']==pytest.approx(20.,abs=1e-12)
        assert row['log_normalizer_difference']==pytest.approx(0.,abs=1e-12)
        assert row['ess']==16 and row['empirical_normalizer_relative_se']==0
    unweighted=tmp_path/'unweighted.json'
    monkeypatch.setattr(sys,'argv',argv[:-1]+[str(unweighted),'--unweighted']);module.main()
    for row in json.loads(unweighted.read_text())['rows']:
        assert row['ess'] is None and row['log_normalizer_estimate'] is None
        assert row['log_normalizer_difference'] is None and row['maximum_weight'] is None
    for stage in ['initial','final']:
        path=run/f'{stage}_samples.pt';data=torch.load(path,weights_only=False)
        data['sample_cluster_ids']=torch.arange(4).repeat_interleave(4);torch.save(data,path)
    clustered=tmp_path/'clustered.json'
    monkeypatch.setattr(sys,'argv',argv[:-1]+[str(clustered),'--unweighted']);module.main()
    row=json.loads(clustered.read_text())['rows'][-1]
    assert row['error_scope'].startswith('clustered')
    values=torch.tensor(invariant_observables(x.numpy())['radius_gyration_A']).reshape(4,4).mean(1)
    expected=float(values.std()/2)
    assert row['moments']['radius_gyration_A']['empirical_estimate_se']==pytest.approx(expected)
    ref['oracle_sha256']='different_oracle';(reference/'reference.json').write_text(json.dumps(ref))
    monkeypatch.setattr(sys,'argv',argv[:-1]+[str(tmp_path/'wrong.json')])
    with pytest.raises(ValueError,match='Potential'):module.main()
