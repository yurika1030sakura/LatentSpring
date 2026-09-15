#!/usr/bin/env python3
"""Replay raw connectivity-study records and fixed parameter-difference controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from flowmol.model_utils.load import model_from_config, read_config_file
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source, geometry_counts
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import flags, intervals
from scripts.research.evaluate_chemical_policy import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'run', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    torch.set_num_threads(2)
    methods = ['frozen','replay','local','tree','local_delta','tree_delta']
    graph = np.zeros((2,6,12,32), bool)
    geometry = graph.copy()
    summary, artifacts = {}, []
    for seed in [0,1]:
        path = a.project/f'research/evidence/endpoint_connectivity_s{seed}_v1.json'
        spec = json.loads(path.read_text()); ph = sha(path)
        root = a.run/f's{seed}/study'
        done = json.loads((root/'complete.json').read_text())
        assert done['complete'] and done['protocol_sha256'] == ph and done['generated_outputs'] == 2304
        assert spec['methods'] == methods
        warm = torch.load(a.project/spec['warm_checkpoint'], map_location='cpu', weights_only=False)
        assert sha(a.project/spec['warm_checkpoint']) == spec['warm_checkpoint_sha256']
        prior = prior_from_checkpoint(warm)
        cfg = read_config_file(a.project/spec['config']); cfg['mol_fm'].pop('bgfm',None)
        model = model_from_config(cfg); prepare_research_backbone(model, warm['research_protocol'])
        parameters = {name for name,_ in model.named_parameters()}
        saved = {'frozen':warm}
        logs = {}
        for method in methods[1:]:
            saved[method] = torch.load(root/method/'last.ckpt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved[method]['state_dict'],strict=True)
            if method in spec['training_methods']:
                logs[method] = [json.loads(line) for line in (root/method/'metrics.jsonl').read_text().splitlines()]
                assert len(logs[method]) == spec['training_steps']
                for r in logs[method]:
                    assert np.isfinite([r['fm_loss'],r['support_loss'],r['total_loss'],r['gradient_norm']]).all()
                    assert np.isclose(r['total_loss'], r['fm_loss']+spec['support_weight']*r['support_loss'],atol=1e-6)
        for method in ['local','tree']:
            assert [(r['step'],r['processed_index'],r['source_seed'],r['time']) for r in logs[method]] == [
                (r['step'],r['processed_index'],r['source_seed'],r['time']) for r in logs['replay']]
            for name,value in warm['state_dict'].items():
                expected = value + (saved[method]['state_dict'][name]-saved['replay']['state_dict'][name]) if name in parameters else value
                torch.testing.assert_close(expected,saved[method+'_delta']['state_dict'][name],rtol=0,atol=0)
        del model
        summary[seed] = {}
        for mi,method in enumerate(methods):
            file = root/'evaluation'/f'{method}_results.json'
            report = json.loads(file.read_text())
            checkpoint = a.project/spec['warm_checkpoint'] if method=='frozen' else root/method/'last.ckpt'
            assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==sha(checkpoint)
            for i,row in enumerate(report['rows']):
                assert row['condition_index']==i
                samplefile = root/'evaluation'/f'{method}_c{i}.pt'
                assert sha(samplefile)==row['sample_sha256']
                sample = torch.load(samplefile,map_location='cpu',weights_only=False);c=sample['condition']
                assert len(sample['positions'])==32
                for j,draw_seed in enumerate(sample['seeds']):
                    assert draw_seed == spec['evaluation_seed']*1000003+i*100003+j
                    expected,tree=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed)
                    torch.testing.assert_close(expected,sample['initial_positions'][j],rtol=0,atol=0)
                    assert tree==sample['auxiliary_tree_edges'][j]
                replay=assess(sample['positions'],c,list(range(32)))
                assert all(row[k]==v for k,v in replay.items())
                assert geometry_counts(sample['positions'],c['numbers'])==row['final_geometry']
                graph[seed,mi,i]=flags(row,'graph_supported').astype(bool)
                geometry[seed,mi,i]=flags(row,'geometrically_supported').astype(bool)
            summary[seed][method]={k:sum(r[k] for r in report['rows']) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','validator_errors','generation_seconds']}
            artifacts.append(dict(seed=seed,method=method,report_sha256=sha(file),checkpoint_sha256=sha(checkpoint)))
            print(json.dumps(dict(seed=seed,method=method,summary=summary[seed][method])),flush=True)
    comparisons={};rng=np.random.default_rng(38691)
    for left,right in [('tree_delta','frozen'),('tree_delta','local_delta'),('local_delta','frozen'),('tree','replay'),('tree','local'),('replay','frozen')]:
        li,ri=methods.index(left),methods.index(right)
        comparisons[left+' minus '+right]={metric:intervals(values[:,li].astype(float)-values[:,ri].astype(float),[0]*12,rng)
            for metric,values in [('graph_supported',graph),('geometrically_supported',geometry)]}
    gate=all(comparisons['tree_delta minus '+m]['graph_supported']['paired_draw95'][0]>0 and
        all(summary[s]['tree_delta']['graph_supported']>summary[s][m]['graph_supported'] for s in [0,1]) for m in ['frozen','local_delta'])
    diversity=sum(summary[s]['tree_delta']['distinct_connectivity']-summary[s]['frozen']['distinct_connectivity'] for s in [0,1])/768>=-.02
    write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,primary_gate=bool(gate and diversity),
        records_replayed=4608,artifacts=artifacts,new_physical_queries=0,scientific_submission_ready=False,
        scope='Reused12-composition development panel; same fitted initial models. No quantum, thermal-law or guaranteed-validity claim. No further tuning of this frozen recipe on these outcomes.'))


if __name__=='__main__':
    main()
