#!/usr/bin/env python3
"""Audit diagnostic copies and report censored relaxation burden without filtering."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import intervals
from scripts.research.capped_relaxation import aligned_rmsd
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','run','out']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    assert not a.out.exists()
    pp=a.project/'research/evidence/capped_relaxation_v1.json';spec=json.loads(pp.read_text())
    report=json.loads((a.run/'results.json').read_text());tasks=json.loads((a.run/'tasks.json').read_text())
    assert report['complete'] and report['protocol_sha256']==sha(pp) and report['tasks_sha256']==sha(a.run/'tasks.json')
    lookup={r['id']:r for r in report['rows']};checked=0
    assert len(lookup)==len(tasks)==408
    summaries={m:{} for m in spec['methods']+['reference']}
    arrays={str(cap):{name:np.zeros((2,2,24,4)) for name in ['graph','converged','same_graph_converged']} for cap in [0,5,20]}
    for task in tasks:
        row=lookup[task['id']];c=task['condition'];directory=a.run/'trajectories'/task['id']
        assert json.loads((directory/'result.json').read_text())==row
        initial=np.asarray(task['positions']);cached=task['cached_singlepoint']
        if task['method']=='reference':
            path=a.project/f"runs/fresh_physics_v1/s0/study/physical_eval/reference_c{task['condition_index']}.pt"
            expected=torch.load(path,map_location='cpu',weights_only=False)['positions'].numpy()
        else:
            path=a.project/f"runs/fresh_physics_v1/s{task['seed']}/study/evaluation/{task['method']}_c{task['condition_index']}.pt"
            expected=torch.load(path,map_location='cpu',weights_only=False)['positions'][task['sample_index']].numpy()
        assert sha(path)==task['source_sample_sha256']==row['source_sample_sha256']
        np.testing.assert_array_equal(initial,expected)
        queries=json.loads((directory/'queries.json').read_text()) if (directory/'queries.json').exists() else []
        assert len(queries)==row['new_singlepoint_attempts']
        for query in queries:
            root=directory/'details'/query['task_id']
            for f,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:
                assert sha(root/f)==query[key]
            gradient=(root/'gradient').read_text() if (root/'gradient').exists() else ''
            if query['gradient_sha256'] is not None:assert sha(root/'gradient')==query['gradient_sha256']
            if query['returncode'] is None:
                assert not query['success'] and query['failure']=='single_point_timeout'
            else:
                parsed=parse_singlepoint((root/'stdout.txt').read_text(),(root/'stderr.txt').read_text(),query['returncode'],gradient,len(c['numbers']))
                assert parsed['success']==query['success']
                if parsed['success']:
                    assert parsed['energy_eV']==query['energy_eV']
                    np.testing.assert_allclose(parsed['force_eV_A'],query['force_eV_A'],rtol=0,atol=0)
                else:assert parsed['failure']==query['failure']
            checked+=1
        raw=row['stages']['0'];raw_pass=raw['graph']['graph_supported']==1
        raw_smiles=raw['graph']['records'][0].get('smiles')
        for cap in [0,5,20]:
            stage=row['stages'][str(cap)]
            key=str(cap);bucket=summaries[task['method']].setdefault(key,dict(attempted=0,available=0,graph_supported=0,
                force_converged=0,raw_graph_supported=0,same_graph_converged=0,raw_graph_preserved=0,
                aligned_rmsd_A=[],raw_graph_aligned_rmsd_A=[]))
            bucket['attempted']+=1;bucket['raw_graph_supported']+=int(raw_pass)
            graph_ok=converged=same=False
            if stage is not None:
                assert stage['step']<=cap and (stage['step']==cap or row['optimizer_converged'])
                replay=assess(torch.tensor([stage['positions']],dtype=torch.float64),c,[task['sample_index']])
                assert replay==stage['graph']
                rmsd=aligned_rmsd(np.asarray(stage['positions']),initial)
                assert np.isclose(rmsd,stage['aligned_rmsd_A'],atol=1e-12)
                converged=stage['max_force_eV_A'] is not None and stage['max_force_eV_A']<spec['force_tolerance_eV_A']
                assert converged==stage['converged']
                graph_ok=replay['graph_supported']==1
                same=raw_pass and graph_ok and replay['records'][0]['smiles']==raw_smiles
                bucket['available']+=1;bucket['graph_supported']+=int(graph_ok)
                bucket['force_converged']+=int(converged);bucket['raw_graph_preserved']+=int(same)
                bucket['same_graph_converged']+=int(same and converged)
                bucket['aligned_rmsd_A'].append(rmsd)
                if raw_pass:bucket['raw_graph_aligned_rmsd_A'].append(rmsd)
            if task['method']!='reference':
                index=(task['seed'],spec['methods'].index(task['method']),task['condition_index'],task['sample_index'])
                arrays[key]['graph'][index]=graph_ok
                arrays[key]['converged'][index]=converged
                arrays[key]['same_graph_converged'][index]=same and converged
    for stages in summaries.values():
        for bucket in stages.values():
            for name in ['aligned_rmsd_A','raw_graph_aligned_rmsd_A']:
                values=bucket.pop(name);bucket[name+'_median']=float(np.median(values)) if values else None
                bucket[name+'_mean']=float(np.mean(values)) if values else None
    rng=np.random.default_rng(38791)
    comparisons={cap:{metric:intervals(values[:,1]-values[:,0],[0]*24,rng) for metric,values in group.items()} for cap,group in arrays.items()}
    assert checked==report['new_raw_singlepoint_attempts']
    write(a.out,dict(complete=True,trajectories_replayed=408,raw_singlepoints_reparsed=checked,
        failed_trajectories=[dict(id=r['id'],error=r['error']) for r in report['rows'] if r['error']],
        summaries=summaries,force_update_minus_harmonic=comparisons,report_sha256=sha(a.run/'results.json'),
        tasks_sha256=sha(a.run/'tasks.json'),protocol_sha256=sha(pp),additional_smoke_gfn2_queries=1,
        scientific_submission_ready=False,scope='Separate diagnostic copies only. Missing trajectories count as unsuccessful in all-attempt rates. RMSD summaries condition on available trajectories and are descriptive. Same-graph convergence uses the raw graph as its reference. Five/twenty steps are caps, not stable-minimum certificates or global thermal tests.'))


if __name__=='__main__':
    main()
