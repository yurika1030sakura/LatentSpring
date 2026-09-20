"""Audit numerical completion and update score arrays without replacing coordinates."""
import json,re
from pathlib import Path
import numpy as np
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.train_electronic_fm import sha


def apply_completion(root,arrays,methods,file):
    file=file.resolve();report=json.loads(file.read_text());proto=root/'research/evidence/replication_scc_completion_v1.json';spec=json.loads(proto.read_text())
    assert report['complete'] and report['protocol_sha256']==sha(proto) and spec['frozen'] and report['requested']==len(spec['failures'])==6
    for f,h in spec['sources'].items():assert sha(root/f)==h
    completed={};proof={};attempts=0
    for row in report['rows']:
        key=row['source_detail'];original=spec['failures'][key];assert not original['original']['success'] and not original['graph_valid'] and original['minimum_distance_A']>0
        assert sha(root/key/'input.xyz')==original['source_xyz_sha256'];records=row['attempts'];attempts+=len(records)
        for k,r in enumerate(records):
            folder=root/r['details'];assert r['stage']==k
            for name,h in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256'),('solver.inp','solver_sha256')]:assert sha(folder/name)==r[h]
            assert r['input_xyz_sha256']==original['source_xyz_sha256'] and r['original_charge']==original['original']['original_charge'] and r['original_spin_multiplicity']==original['original']['original_spin_multiplicity']
            assert r['command'][-1]=='--grad' and '--opt' not in r['command'] and r['command'][r['command'].index('--acc')+1]=='.1'
            control=(folder/'solver.inp').read_text();assert 'maxiterations=1000' in control and 'temp=300' in control
            if k==0:assert 'broydamp' not in control
            else:assert 'broydamp=0.2' in control and not records[k-1]['success']
            stdout=(folder/'stdout.txt').read_text();assert re.search(r'max\. iterations\s+1000',stdout)
            if r['returncode'] is not None:
                gradient=(folder/'gradient').read_text() if (folder/'gradient').exists() else '';parsed=parse_singlepoint(stdout,(folder/'stderr.txt').read_text(),r['returncode'],gradient,original['natoms']);assert parsed['success']==r['success']
                if r['success']:
                    assert sha(folder/'gradient')==r['gradient_sha256'];assert parsed['energy_eV']==r['energy_eV'];np.testing.assert_array_equal(parsed['force_eV_A'],r['force_eV_A'])
            if r['success']:assert k==len(records)-1
        assert row['successful']==records[-1]['success']
        if row['successful']:completed[key]=records[-1]
        proof[key]=dict(attempts=len(records),success=row['successful'],source_xyz_sha256=original['source_xyz_sha256'],minimum_distance_A=original['minimum_distance_A'])
    assert attempts==report['new_gfn2_attempts'];before_joint=arrays['graph']&arrays['success']&(arrays['force']<=5);updates=[]
    for si in range(5):
        folder=root/f'runs/seed_replication_v1/evaluation/s{si}';parents=folder/'parent_xtb/results.json';readouts=folder/'readouts/physical.json'
        for f in [parents,readouts]:
            values=json.loads(f.read_text())['rows']
            for row in values:
                if 'physical' in row:
                    old=row['physical']['result'];detail=row['physical']['details'];family,method=row['method'].split('_',1);label=family+'_'+{'base':'physical','radial':'radial','molecule_start0':'hydrogen'}[method]
                else:
                    old=row;detail=str((f.parent/'details'/old['task_id']).relative_to(root));family=old['method'].split('_')[0];label=family+('_parent' if old['method'].endswith('a0') else '_physical')
                if old['success'] or detail not in completed:continue
                r=completed[detail];mi=methods.index(label);i,j=row['condition_index'],row['sample_index'];key=(si,mi,i,j)
                assert not arrays['graph'][key];force=float(np.sqrt(np.square(r['force_eV_A']).sum(-1).mean()));energy=r['energy_eV']/len(r['force_eV_A'])
                if arrays['success'][key]:assert arrays['energy'][key]==energy and arrays['force'][key]==force
                else:updates.append(dict(fit=si,method=label,condition_index=i,sample_index=j,source_detail=detail))
                arrays['success'][key]=True;arrays['energy'][key]=energy;arrays['force'][key]=force
    after_joint=arrays['graph']&arrays['success']&(arrays['force']<=5);np.testing.assert_array_equal(before_joint,after_joint)
    return dict(complete=True,protocol_sha256=sha(proto),results_sha256=sha(file),proof=proof,updated_score_cells=updates,new_gfn2_attempts=attempts,
        unchanged_graph_and_joint_arrays=True,successful_unique_geometries=len(completed),scope='Only numerically unconverged scores are completed. Original outcomes remain in the first-pass audit; no coordinates, models, energy criteria or valid-output counts change.')
