#!/usr/bin/env python3
"""Replay every new source-curve output and report all five fixed continuations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','out']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    budgets=[1000,3000,6000];draws={1000:16,3000:32,6000:16};methods=['gaussian','harmonic_tree']
    arrays={step:np.zeros((3,2,64,draws[step]),bool) for step in budgets};records=[];new_count=0
    for slot,seed in enumerate([2,3,4]):
        path=args.project/f'research/evidence/source_replication_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path)
        root=args.run/f's{seed}';completion=json.loads((root/'complete.json').read_text())
        assert completion['complete'] and completion['protocol_sha256']==ph
        assert sha(args.project/spec['data'])==spec['data_sha256']
        data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training']
        panel=json.loads((args.project/spec['condition_manifest']).read_text())['rows']
        for mi,method in enumerate(methods):
            directory=root/method;metrics=[json.loads(l) for l in (directory/'metrics.jsonl').read_text().splitlines()]
            assert len(metrics)==6000
            for step,row in enumerate(metrics,1):
                index=(step-1)%3000
                assert row['step']==step and row['training_row']==index and row['processed_index']==data[index]['condition']['processed_index']
                assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
            milestones=json.loads((directory/'milestones.json').read_text());assert milestones['complete'] and milestones['protocol_sha256']==ph
            for step,milestone in zip(budgets,milestones['rows']):
                checkpoint=directory/f'step{step}.ckpt';assert milestone['step']==step and sha(checkpoint)==milestone['checkpoint_sha256']
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert state['global_step']==step and state['research_protocol']['source_replication_protocol_sha256']==ph
                assert state['research_protocol']['source_prior_kind']==method
                del state
                out=directory/f'evaluation_step{step}';report_path=out/f'{method}_results.json';report=json.loads(report_path.read_text())
                assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==milestone['checkpoint_sha256']
                assert len(report['rows'])==64
                for index,row in enumerate(report['rows']):
                    assert row['condition_index']==index
                    file=out/f'{method}_c{index}.pt';assert sha(file)==row['sample_sha256']
                    saved=torch.load(file,map_location='cpu',weights_only=False);x=saved['positions'];c=saved['condition']
                    assert len(x)==draws[step] and c['composition_hex']==panel[index]['composition_hex']
                    result=assess(x,c,list(range(len(x))))
                    for key in result:assert result[key]==row[key]
                    arrays[step][slot,mi,index]=[r['graph_supported'] for r in result['records']];new_count+=len(x)
                records.append(dict(seed=seed,data_block=spec['data_block'],method=method,step=step,checkpoint_sha256=milestone['checkpoint_sha256'],report_sha256=sha(report_path),
                    graph_count=int(arrays[step][slot,mi].sum()),attempted=int(arrays[step][slot,mi].size),training_seconds=milestone['training_seconds']))
        print(json.dumps(dict(seed=seed,audited=True,new_outputs_checked=new_count)),flush=True)
    previous=args.project/'research/evidence/wide_generalization_audit_v1.json';prior=json.loads(previous.read_text());arrayfile=previous.with_suffix('.npz')
    assert prior['complete'] and sha(arrayfile)==prior['arrays_sha256'];old=np.load(arrayfile)['harmonic_tree__minus__gaussian__graph_supported']
    new=arrays[3000][:,1].astype(float)-arrays[3000][:,0].astype(float);five=np.concatenate([old,new])
    rng=np.random.default_rng(45233);sampled=rng.integers(0,64,size=(20000,64))
    def interval(difference):
        cell=difference.mean(-1);boot=cell.mean(0)[sampled].mean(1)
        return dict(mean=float(cell.mean()),by_seed=cell.mean(1).tolist(),composition_ci95=np.quantile(boot,[.025,.975]).tolist())
    result=interval(five);effect=np.asarray(result['by_seed']);result.update(continuation_sample_sd=float(effect.std(ddof=1)),positive_continuations=int((effect>0).sum()),data_blocks=[0,1,0,1,0])
    curves={str(step):interval(value[:,1].astype(float)-value[:,0].astype(float)) for step,value in arrays.items()}
    outarray=args.out.with_suffix('.npz');np.savez_compressed(outarray,**{f'graph_step{step}':value for step,value in arrays.items()},five_seed_difference=five)
    write(args.out,dict(complete=True,new_outputs_replayed=new_count,new_training_steps=36000,new_physical_queries=0,
        fixed3000_five_continuations=result,new_three_seed_budget_curves=curves,records=records,previous_audit_sha256=sha(previous),
        arrays_file=outarray.name,arrays_sha256=sha(outarray),selection_used=False,
        scope='Five stochastic continuations of one shared pretrained model on two fixed3000-row TRAIN blocks. Curve points use three new continuations. Every seed and budget is reported; composition intervals condition on fitted models, with continuation variation shown separately.'))
    assert new_count==24576
    print(json.dumps(dict(complete=True,five_continuations=result,curves=curves)),flush=True)


if __name__=='__main__':main()
