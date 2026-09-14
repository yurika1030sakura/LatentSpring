#!/usr/bin/env python3
"""Replay all molecular support outcomes and matched FM training provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_chemical_policy import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','run','training','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(2)
    protocol=json.loads(args.protocol.read_text())
    path=args.run/'results.json';report=json.loads(path.read_text())
    assert report['complete'] and report['protocol_sha256']==sha(args.protocol)
    training={};all_indices=[]
    for method in protocol['methods']:
        directory=args.training/method
        recipe=json.loads((directory/'protocol.json').read_text())
        validation=json.loads((directory/'validation.json').read_text())
        metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        assert validation['complete'] and validation['global_step']==protocol['training_steps']==len(metrics)
        assert recipe['pairing_protocol_sha256']==sha(args.protocol) and recipe['pairing']==method
        assert sha(directory/'last.ckpt')==report['sources'][method]['sha256']
        indices=[];changed=0
        for step,row in enumerate(metrics,1):
            assert row['step']==step and np.isfinite(row['fm_loss']) and np.isfinite(row['gradient_norm'])
            assert len(row['processed_indices'])==protocol['batch_size']==len(row['pairing'])
            indices.extend(row['processed_indices'])
            for pair in row['pairing']:
                assert pair['mode']==method
                if method=='steric':
                    assert pair['selected_cost']<=pair['standard_cost']+1e-5+1e-6*abs(pair['standard_cost'])
                    assert pair['selected_candidate'] in range(13)
                    changed+=pair['selected_candidate']!=0
                elif method=='rotation':assert pair['selected_candidate']==0
                else:assert pair['selected_candidate']==-1
        assert len(set(indices))==len(indices)
        all_indices.append(indices)
        training[method]=dict(steps=len(metrics),selected_nonstandard_rotation=changed,
            mean_selected_overlap=float(np.mean([p['selected_overlap'] for r in metrics for p in r['pairing']])),
            mean_standard_overlap=float(np.mean([p['standard_overlap'] for r in metrics for p in r['pairing']])),
            mean_displacement_per_atom_A2=float(np.mean([p['displacement_per_atom'] for r in metrics for p in r['pairing']])),
            seconds=validation['seconds'],metrics_sha256=sha(directory/'metrics.jsonl'),protocol_sha256=sha(directory/'protocol.json'))
    assert all(indices==all_indices[0] for indices in all_indices)
    assert len(report['rows'])==4*len(protocol['conditions'])
    keyed={};replayed=0;files={}
    for row in report['rows']:
        method,index=row['method'],row['condition_index']
        assert method in ['warm']+protocol['methods'] and index in protocol['conditions']
        key=(method,index);assert key not in keyed;keyed[key]=row
        source=args.run/f'{method}_c{index}.pt'
        assert sha(source)==row['sample_sha256']
        saved=torch.load(source,map_location='cpu',weights_only=False)
        assert saved['condition']==row['condition'] and saved['seeds']==row['seeds']
        x=saved['positions'];assert x.shape==(protocol['samples_per_condition'],row['condition']['n_atoms'],3)
        assert torch.isfinite(x).all() and x.mean(1).abs().max()<1e-8
        result=assess(x,row['condition'],list(range(len(x))))
        for k,v in result.items():assert v==row[k],(key,k)
        assert abs(float(x.square().sum(-1).mean(-1).sqrt().mean())-row['radius_of_gyration_mean_A'])<1e-12
        expected=[protocol['evaluation_seed']*1000003+index*100003+b for b in range(len(x)//protocol['evaluation_batch'])]
        assert row['seeds']==expected
        assert row['neural_field_evaluations']==len(x)*2*protocol['midpoint_steps']
        files[str(source)]=sha(source);replayed+=len(x)
    comparisons={}
    for control in ['warm','independent','rotation']:
        difference=[];by_condition=[]
        for index in protocol['conditions']:
            neural=keyed['steric',index];base=keyed[control,index]
            assert neural['condition']==base['condition'] and neural['seeds']==base['seeds']
            d=np.array([int(a['graph_supported'])-int(b['graph_supported']) for a,b in zip(neural['records'],base['records'])])
            difference.append(d)
            by_condition.append(dict(condition=index,supported_difference=int(d.sum()),mean_difference=float(d.mean())))
        matrix=np.stack(difference)
        rng=np.random.default_rng(88411)
        indices=rng.integers(matrix.shape[1],size=(10000,*matrix.shape))
        boot=np.take_along_axis(matrix[None],indices,axis=2).mean((1,2))
        comparisons['steric minus '+control]=dict(graph_support_fraction_difference=float(matrix.mean()),
            conditional_paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),per_condition=by_condition)
    totals={m:sum(keyed[m,i]['graph_supported'] for i in protocol['conditions']) for m in ['warm']+protocol['methods']}
    assert totals==report['graph_supported_totals']
    gate=totals['steric']>totals['independent'] and totals['steric']>totals['rotation']
    assert gate==report['proceed_to_frozen_energy_check']
    write(args.out,dict(complete=True,protocol_sha256=sha(args.protocol),source_results_sha256=sha(path),
        source_sample_hashes=files,all_structural_outcomes_replayed=replayed,training=training,
        matched_training_indices=True,comparisons=comparisons,graph_supported_totals=totals,
        proceed_to_frozen_energy_check=gate,new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='All saved structural outcomes and training metadata/recorded indices checked. Molecular generation and full optimizer training are not independently replayed by this audit. Intervals condition on fixed models/compositions, not independent training replications.'))
    print(json.dumps(dict(totals=totals,gate=gate,comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
