#!/usr/bin/env python3
"""Audit validation selection, then evaluate the untouched confirmation panel."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cfm_mol import matched_egnn as base
from cfm_mol import connectivity_feedback as feedback
from scripts.research.audit_generator_output_support import assess
from scripts.research.run_gaga_feedback import evaluate
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def audit_report(path,expected_rows,count):
    report=json.loads(path.read_text());assert report['complete'] and report['raw_unoptimized']
    assert report['settings']['calls']==128 and report['settings']['count']==count
    assert len(report['rows'])==len(expected_rows)
    for index,(row,condition) in enumerate(zip(report['rows'],expected_rows)):
        assert row['condition']==dict(condition,numbers=condition['atomic_numbers']) and row['condition_index']==index
        file=path.parent/(path.name.removesuffix('_results.json')+f'_c{index}.pt')
        assert sha(file)==row['sample_sha256']
        saved=torch.load(file,map_location='cpu',weights_only=False)
        assert saved['settings']==report['settings'] and saved['model_state_sha256']==report['model_state_sha256']
        assert saved['condition']==row['condition']
        repeated=assess(saved['positions'],row['condition'],list(range(count)))
        for key in ['attempted','geometrically_supported','graph_supported','validator_errors','distinct_connectivity','connectivity_counts','records']:
            assert repeated[key]==row[key],(path,index,key)
    assert report['attempted']==sum(r['attempted'] for r in report['rows'])
    assert report['graph_supported']==sum(r['graph_supported'] for r in report['rows'])
    return report


def select(project,run,campaign,panel):
    methods=['distance','tree','harmonic_fm','gaga_350','gaga_500','gaga_650']
    counts={m:[] for m in methods};hashes={};models={}
    for seed in campaign['seeds']:
        for method in methods:
            relative=(Path('training')/f's{seed}'/method/'validation'/f'{method}_results.json'
                      if method in ['distance','tree'] else Path('training')/f's{seed}'/'baselines'/f'{method}_results.json')
            report=audit_report(run/relative,panel['validation_rows'],campaign['validation_samples'])
            assert report['settings']['seed']==43001+seed
            counts[method].append(report['graph_supported'])
            hashes[str(relative)]=sha(run/relative);models[f's{seed}/{method}']=report['model_state_sha256']
    selected=max(campaign['challengers'],key=lambda m:(sum(counts[m]),m=='distance'))
    maximum=max(campaign['gaga_sampling_max_t'],key=lambda t:(sum(counts[f'gaga_{t}']),t))
    result=dict(complete=True,role='Validation-only choices frozen before confirmation generation',
        selected_challenger=selected,selected_gaga_max_t=maximum,graph_counts_by_seed=counts,
        attempts_per_model_seed=512,validation_report_sha256=hashes,validation_model_state_sha256=models,
        fresh_outputs_used_for_selection=False,new_validation_outputs=6144,new_physical_queries=0)
    path=run/'selection.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write(path,result)
    print(json.dumps(dict(selection=selected,gaga_max_t=maximum,validation_counts=counts)),flush=True)
    return result


def bootstrap(array,rng,repetitions):
    # Array[model seed, composition]; resample compositions, retaining both seeds.
    samples=rng.integers(0,array.shape[1],size=(repetitions,array.shape[1]))
    values=array.mean(0)[samples].mean(1)
    return dict(mean=float(array.mean()),by_seed=array.mean(1).tolist(),ci95=np.quantile(values,[.025,.975]).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','campaign']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--stage',choices=['select','confirm','audit'],required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    campaign=json.loads(args.campaign.read_text());assert campaign['frozen']
    assert sha(args.project/campaign['panel'])==campaign['panel_sha256']
    panel=json.loads((args.project/campaign['panel']).read_text())
    if args.stage=='select':
        select(args.project,args.run,campaign,panel);return
    selection=json.loads((args.run/'selection.json').read_text());assert selection['complete']
    for name,digest in selection['validation_report_sha256'].items():assert sha(args.run/name)==digest
    methods=['distance','tree','harmonic_fm',f'gaga_{selection["selected_gaga_max_t"]}']
    if 'gaga_650' not in methods:methods.append('gaga_650')
    if args.stage=='confirm':
        for seed in campaign['seeds']:
            common=json.loads((args.project/f'research/evidence/gaga_feedback_distance_s{seed}_v1.json').read_text())
            source=base.HarmonicSource(common['edge_log_width'])
            for method in methods:
                if method in campaign['challengers']:
                    checkpoint=args.run/'training'/f's{seed}'/method/'last.ckpt'
                    spec_path=args.project/f'research/evidence/gaga_feedback_{method}_s{seed}_v1.json'
                    spec=json.loads(spec_path.read_text())
                    saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
                    assert saved['global_step']==15000 and saved['protocol_sha256']==sha(spec_path)
                    model=feedback.install(base.initialize(spec,'cuda'))
                    model.load_state_dict(saved['ema_state_dict'],strict=True)
                    context=feedback.GeometryContext(source,method,spec['tree_regularization'])
                else:
                    kind='gaga' if method.startswith('gaga') else 'harmonic_fm'
                    spec=json.loads((args.project/f'research/evidence/matched_generators_{kind}_s{seed}_v1.json').read_text())
                    checkpoint=args.project/common['baselines'][kind]['path']
                    assert sha(checkpoint)==common['baselines'][kind]['sha256']
                    saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
                    assert saved['global_step']==30000
                    model=base.initialize(spec,'cuda');model.load_state_dict(saved['ema_state_dict'],strict=True)
                    context=None
                    if kind=='gaga':spec['gaga_max_t']=int(method.split('_')[1])
                del saved
                assert base.state_hash(model)==selection['validation_model_state_sha256'][f's{seed}/{method}']
                spec['evaluation_batch']=campaign['evaluation_batch']
                evaluate(model,source,spec,context,panel['test_rows'],43101+seed,campaign['test_samples'],args.run/'confirmation'/f's{seed}',method)
                del model,context
                torch.cuda.empty_cache()
        write(args.run/'confirmation_complete.json',dict(complete=True,selection_sha256=sha(args.run/'selection.json'),methods=methods,
            new_outputs=len(methods)*2*32*32,new_physical_queries=0))
        return
    completion=json.loads((args.run/'confirmation_complete.json').read_text())
    assert completion['complete'] and completion['selection_sha256']==sha(args.run/'selection.json') and completion['methods']==methods
    matrices={m:[] for m in methods};counts={m:[] for m in methods};hashes={}
    for seed in campaign['seeds']:
        for method in methods:
            relative=Path('confirmation')/f's{seed}'/(method+'_results.json')
            report=audit_report(args.run/relative,panel['test_rows'],campaign['test_samples'])
            assert report['settings']['seed']==43101+seed
            assert report['model_state_sha256']==selection['validation_model_state_sha256'][f's{seed}/{method}']
            matrices[method].append([r['graph_supported']/r['attempted'] for r in report['rows']])
            counts[method].append(report['graph_supported']);hashes[str(relative)]=sha(args.run/relative)
    matrices={k:np.asarray(v) for k,v in matrices.items()}
    baseline=f'gaga_{selection["selected_gaga_max_t"]}'
    rng=np.random.default_rng(campaign['statistics']['bootstrap_seed']);repetitions=campaign['statistics']['bootstrap_repetitions']
    contrasts={}
    for method in ['distance','tree','harmonic_fm']:
        contrasts[method+'_minus_selected_gaga']=bootstrap(matrices[method]-matrices[baseline],rng,repetitions)
    contrasts['tree_minus_distance']=bootstrap(matrices['tree']-matrices['distance'],rng,repetitions)
    primary=contrasts[selection['selected_challenger']+'_minus_selected_gaga']
    result=dict(complete=True,primary_gate=bool(min(primary['by_seed'])>0 and primary['ci95'][0]>0),
        selected_challenger=selection['selected_challenger'],selected_gaga_max_t=selection['selected_gaga_max_t'],
        graph_counts_by_seed=counts,attempts_per_method_seed=1024,contrasts=contrasts,
        selection_sha256=sha(args.run/'selection.json'),report_sha256=hashes,
        all_saved_graph_assays_replayed=True,new_validation_outputs=6144,new_confirmation_outputs=completion['new_outputs'],
        new_physical_queries=0,reserved_outcomes_queried=False,
        inference_scope='All attempts, raw unoptimized coordinates,128 backbone calls.',
        training_scope=campaign['training_budget'],statistics=campaign['statistics'])
    write(args.run/'audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
