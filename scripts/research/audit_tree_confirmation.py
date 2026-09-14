#!/usr/bin/env python3
"""Independent-seed confirmation audit of the fixed tree versus Gaussian source."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from scripts.research.tree_prior_fm import sample_source,geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());assert spec['frozen'];torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    records={};selection=None;indices=None;checked=0;hashes={}
    for task,method in enumerate(['gaussian','fixed']):
        root=a.run/f's{task}/study';directory=root/method
        selected=json.loads((root/'selection.json').read_text())
        if selection is None:selection=selected
        else:assert selected==selection
        validation=json.loads((directory/'validation.json').read_text());recipe=json.loads((directory/'protocol.json').read_text())
        assert validation['complete'] and validation['global_step']==spec['fm_steps']
        assert recipe['tree_protocol_sha256']==sha(a.protocol) and recipe['warm_sha256']==spec['warm_checkpoint_sha256']
        assert sha(directory/'last.ckpt')==validation['checkpoint_sha256']
        checkpoint=torch.load(directory/'last.ckpt',map_location='cpu',weights_only=False)
        assert checkpoint['research_protocol']==recipe and recipe['source_prior_kind']==method
        if method=='gaussian':assert checkpoint['source_prior'] is None
        else:assert checkpoint['source_prior']['configuration']['mode']=='fixed'
        del checkpoint
        metrics=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
        actual=[r['processed_index'] for r in metrics]
        assert len(actual)==spec['fm_steps'] and actual==selection['selected'][:spec['fm_steps']]
        assert len(set(actual))==len(actual)
        assert [r['step'] for r in metrics]==list(range(1,spec['fm_steps']+1))
        assert all(np.isfinite(r['fm_loss']) and np.isfinite(r['gradient_norm']) for r in metrics)
        if indices is None:indices=actual
        else:assert actual==indices
        rp=root/f'evaluation/{method}_results.json';r=json.loads(rp.read_text())
        assert r['complete'] and r['protocol_sha256']==sha(a.protocol) and r['checkpoint_sha256']==validation['checkpoint_sha256']
        assert sorted(row['condition_index'] for row in r['rows'])==sorted(spec['conditions'])
        prior=None if method=='gaussian' else TreeMixturePrior('fixed',width=spec['edge_log_width']).double()
        for row in r['rows']:
            index=row['condition_index'];path=root/f'evaluation/{method}_c{index}.pt'
            assert sha(path)==row['sample_sha256']
            d=torch.load(path,map_location='cpu',weights_only=False);c=d['condition'];x=d['positions'];x0=d['initial_positions']
            assert c==row['condition'] and len(x)==spec['samples_per_condition']
            assert len(d['seeds'])==len(x)==len(d['auxiliary_tree_edges'])
            for j,seed in enumerate(d['seeds']):
                assert seed==spec['evaluation_seed']*1000003+index*100003+j
                z,t=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],seed)
                torch.testing.assert_close(z,x0[j],atol=1e-10,rtol=1e-10);assert t==d['auxiliary_tree_edges'][j]
                checked+=1
            for k,v in assess(x,c,list(range(len(x)))).items():assert row[k]==v
            assert geometry_counts(x,c['numbers'])==row['final_geometry'] and geometry_counts(x0,c['numbers'])==row['initial_geometry']
        records[method]=r['rows'];hashes[method]=dict(results_sha256=sha(rp),checkpoint_sha256=validation['checkpoint_sha256'],metrics_sha256=sha(directory/'metrics.jsonl'))
    differences=[]
    for index in spec['conditions']:
        l=next(r for r in records['fixed'] if r['condition_index']==index);r=next(r for r in records['gaussian'] if r['condition_index']==index)
        assert l['condition']==r['condition']
        differences.append([int(x['graph_supported'])-int(y['graph_supported']) for x,y in zip(l['records'],r['records'])])
    d=np.array(differences);g=np.random.default_rng(88601);ids=g.integers(d.shape[1],size=(10000,*d.shape))
    methods={m:dict(attempted=sum(r['attempted'] for r in rows),graph_supported=sum(r['graph_supported'] for r in rows),
        geometrically_supported=sum(r['geometrically_supported'] for r in rows),disconnected=sum(r['final_geometry']['disconnected'] for r in rows),
        overlap=sum(r['final_geometry']['overlap'] for r in rows),validator_errors=sum(r['validator_errors'] for r in rows)) for m,rows in records.items()}
    comparison=dict(graph_support_difference=float(d.mean()),conditional_paired_bootstrap95=np.quantile(np.take_along_axis(d[None],ids,axis=2).mean((1,2)),[.025,.975]).tolist(),per_condition_counts=d.sum(1).tolist())
    assert checked==2*len(spec['conditions'])*spec['samples_per_condition']
    write(a.out,dict(complete=True,protocol_sha256=sha(a.protocol),sources=hashes,methods=methods,comparison=comparison,
        initial_sources_and_final_structures_replayed=checked,matched_training_rows=len(indices),
        rows={m:[{k:v for k,v in r.items() if k!='records'} for r in rows] for m,rows in records.items()},
        new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='Independent training/data-order and generation seeds, same8 development compositions. Prior sampling and structural outcomes replayed; optimizer and final neural generation not independently replayed.'))
    print(json.dumps(dict(methods=methods,comparison=comparison)),flush=True)


if __name__=='__main__':main()
