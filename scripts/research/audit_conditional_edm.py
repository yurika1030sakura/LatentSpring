#!/usr/bin/env python3
"""Audit the independently adapted EDM and compare matched composition panels."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts.research.run_conditional_edm import build
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import flags
from scripts.research.tree_prior_fm import geometry_counts
from scripts.research.evaluate_chemical_policy import write


def independent_intervals(left,right,rng,repeats=5000):
    s,c,n=left.shape
    def draws(values):
        indices=rng.integers(n,size=(repeats,s,c,n))
        return values[np.arange(s)[None,:,None,None],np.arange(c)[None,None,:,None],indices].mean((1,2,3))
    difference=left.astype(float)-right.astype(float)
    interval=np.quantile(draws(left)-draws(right),[.025,.975]).tolist()
    return dict(mean=float(difference.mean()),independent_draw95=interval,
                by_continuation=difference.mean((1,2)).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','run','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();torch.set_num_threads(2)
    methods=['edm_128','edm_1001'];graph=np.zeros((2,2,24,32),bool);geometry=graph.copy()
    summary={};artifacts=[]
    for seed in [0,1]:
        pp=a.project/f'research/evidence/conditional_edm_s{seed}_v1.json';spec=json.loads(pp.read_text());ph=sha(pp)
        root=a.run/f's{seed}/study';done=json.loads((root/'complete.json').read_text())
        assert done['complete'] and done['protocol_sha256']==ph and done['new_neural_outputs']==1536
        model=build(spec,'cpu');saved=torch.load(root/'last.ckpt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==ph;model.load_state_dict(saved['state_dict'],strict=True)
        assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
        training=json.loads((root/'training.json').read_text());assert training['complete'] and training['checkpoint_sha256']==sha(root/'last.ckpt')
        data=torch.load(a.project/spec['data'],map_location='cpu',weights_only=False)
        records=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
        assert len(records)==spec['training_steps']==len(data['training'])==3000
        assert [r['processed_index'] for r in records]==[r['condition']['processed_index'] for r in data['training']]
        assert all(np.isfinite([r['loss'],r['gradient_norm']]).all() for r in records)
        del model,saved,data
        summary[seed]={};initials={}
        for mi,method in enumerate(methods):
            file=root/'evaluation'/f'{method}_results.json';report=json.loads(file.read_text())
            assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==sha(root/'last.ckpt')
            for i,row in enumerate(report['rows']):
                assert row['condition_index']==i
                source=root/'evaluation'/f'{method}_c{i}.pt';assert sha(source)==row['sample_sha256']
                sample=torch.load(source,map_location='cpu',weights_only=False);x=sample['positions'];c=sample['condition']
                assert x.shape==(32,c['n_atoms'],3) and torch.isfinite(x).all()
                assert sample['primitive_denoiser_calls_per_sample']==int(method.split('_')[1])
                assert sample['batch_seeds']==[spec['evaluation_seed']*1000003+i*100003+begin for begin in [0,16]]
                if mi==0:initials[i]=sample['initial_positions']
                else:torch.testing.assert_close(initials[i],sample['initial_positions'],rtol=0,atol=0)
                replay=assess(x,c,list(range(32)));assert all(row[k]==v for k,v in replay.items())
                assert geometry_counts(x,c['numbers'])==row['final_geometry']
                graph[seed,mi,i]=flags(row,'graph_supported').astype(bool)
                geometry[seed,mi,i]=flags(row,'geometrically_supported').astype(bool)
            summary[seed][method]={k:sum(r[k] for r in report['rows']) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','validator_errors','generation_seconds']}
            summary[seed][method]['training']=training
            artifacts.append(dict(seed=seed,method=method,checkpoint_sha256=sha(root/'last.ckpt'),report_sha256=sha(file)))
            print(json.dumps(dict(seed=seed,method=method,summary=summary[seed][method])),flush=True)
    prior_path=a.project/'research/evidence/fresh_physics_esen_audit_v1.json';prior=json.loads(prior_path.read_text())
    arrays=prior_path.parent/prior['arrays_file'];assert sha(arrays)==prior['arrays_sha256']
    original=np.load(arrays);rng=np.random.default_rng(38891);comparisons={}
    for m in ['harmonic_tree','escort_delta']:
        for mi,baseline in enumerate(methods):
            comparisons[m+' minus '+baseline]={metric:independent_intervals(original[key][:,prior['methods'].index(m)],new[:,mi],rng)
                for metric,key,new in [('graph_supported','graph',graph),('geometrically_supported','geometry',geometry)]}
    np.savez_compressed(a.out.with_suffix('.npz'),graph=graph,geometry=geometry)
    write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,records_replayed=3072,artifacts=artifacts,
        arrays_sha256=sha(a.out.with_suffix('.npz')),arrays_file=a.out.with_suffix('.npz').name,
        scientific_submission_ready=False,scope='Shared continuation dataset and task, different pretrained backbones/histories. Conditional independent-draw intervals hold trained models and compositions fixed; they are not retraining/general-population intervals. No native published joint-generation or matched total-compute superiority claim.'))


if __name__=='__main__':main()
