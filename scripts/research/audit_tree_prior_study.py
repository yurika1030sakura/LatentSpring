#!/usr/bin/env python3
"""Audit source laws, prior likelihoods, training rows and all generated outcomes."""
import argparse
import json
import math
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
    for key in ['protocol','prior','fixed','learned','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);spec=json.loads(a.protocol.read_text());digest=sha(a.protocol)
    prior_report=json.loads((a.prior/'results.json').read_text())
    assert prior_report['complete'] and prior_report['protocol_sha256']==digest
    assert sha(a.prior/'data.pt')==prior_report['data_sha256']
    data=torch.load(a.prior/'data.pt',map_location='cpu',weights_only=False)
    assert len(data['training'])==spec['prior_steps'] and len(data['validation'])==spec['prior_validation_rows']
    models={};prior_checks={}
    for mode in ['fixed','node','pair']:
        if mode=='fixed':model=TreeMixturePrior(mode,width=spec['edge_log_width']).double()
        else:
            path=a.prior/f'{mode}.pt';assert sha(path)==prior_report['models'][mode]['checkpoint_sha256']
            state=torch.load(path,map_location='cpu',weights_only=False);model=TreeMixturePrior(**state['configuration']).double()
            model.load_state_dict(state['state_dict'],strict=True)
            assert state['data_sha256']==prior_report['data_sha256'] and state['protocol_sha256']==digest
            metrics=json.loads((a.prior/f'{mode}_metrics.json').read_text())
            assert len(metrics)==spec['prior_steps'] and sha(a.prior/f'{mode}_metrics.json')==prior_report['models'][mode]['metrics_sha256']
            assert [r['processed_index'] for r in metrics]==[r['processed_index'] for r in data['training']]
        model.eval();model.requires_grad_(False);models[mode]=model
        values=[float(-model.log_prob(r['positions'],r['numbers'],r['charge'],r['spin'])/max(1,3*(len(r['numbers'])-1))) for r in data['validation']]
        expected=prior_report['fixed_validation_nll_per_dof'] if mode=='fixed' else prior_report['models'][mode]['validation_nll_per_dof']
        np.testing.assert_allclose(values,expected,atol=1e-9,rtol=1e-9)
        prior_checks[mode]=dict(mean_validation_nll_per_dof=float(np.mean(values)),values_replayed=len(values))
    if spec.get('target_domain'):
        selection=json.loads((a.prior/'selection.json').read_text());assert sha(a.prior/'selection.json')==prior_report['selection_sha256']
        for group in [a.fixed,a.learned]:assert json.loads((group/'selection.json').read_text())==selection
        for row in data['training']+data['validation']:
            assert geometry_counts(row['positions'][None],row['numbers'])==dict(disconnected=0,overlap=0)
    rows=[];sources={};training={};indices=[];source_checks=0;density_checks=0
    for method in ['warm']+spec['methods']:
        group=a.fixed if method in ['warm','gaussian','fixed'] else a.learned
        report_path=group/f'evaluation/{method}_results.json';report=json.loads(report_path.read_text())
        assert report['complete'] and report['protocol_sha256']==digest and len(report['rows'])==len(spec['conditions'])
        prior=None if method in ['warm','gaussian'] else models[method]
        if method!='warm':
            directory=group/method;validation=json.loads((directory/'validation.json').read_text())
            assert validation['complete'] and validation['global_step']==spec['fm_steps']
            ckpt=directory/'last.ckpt';assert sha(ckpt)==validation['checkpoint_sha256']==report['checkpoint_sha256']
            state=torch.load(ckpt,map_location='cpu',weights_only=False);recipe=state['research_protocol']
            assert recipe['tree_protocol_sha256']==digest and recipe['source_prior_kind']==method
            assert recipe['data_order_sha256']==prior_report['data_order_sha256'] and recipe['warm_sha256']==spec['warm_checkpoint_sha256']
            if prior is None:assert state['source_prior'] is None
            else:
                assert state['source_prior']['configuration']==prior.configuration
                for name,value in prior.state_dict().items():
                    # Fixed-head random parameters are unused by the fixed source.
                    if method!='fixed' or name in ['radii','base_log_propensity']:
                        torch.testing.assert_close(value,state['source_prior']['state_dict'][name],atol=0,rtol=0)
            del state
            metrics=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
            assert len(metrics)==spec['fm_steps'] and [r['step'] for r in metrics]==list(range(1,spec['fm_steps']+1))
            assert all(np.isfinite(r['fm_loss']) and np.isfinite(r['gradient_norm']) for r in metrics)
            actual=[r['processed_index'] for r in metrics];assert len(set(actual))==len(actual)
            assert actual[:spec['prior_steps']]==[r['processed_index'] for r in data['training']]
            assert set(actual).isdisjoint(r['processed_index'] for r in data['validation']);indices.append(actual)
            training[method]=dict(seconds=validation['seconds'],metrics_sha256=sha(directory/'metrics.jsonl'),
                mean_fm_loss=float(np.mean([r['fm_loss'] for r in metrics])),initial_disconnected=sum(r['initial_geometry']['disconnected'] for r in metrics),
                mean_paired_displacement_A2=float(np.mean([r['pairing'][0]['displacement_per_atom'] for r in metrics])))
        else:assert report['checkpoint_sha256']==spec['warm_checkpoint_sha256']
        sources[method]=dict(results_sha256=sha(report_path),checkpoint_sha256=report['checkpoint_sha256'])
        for row in report['rows']:
            index=row['condition_index'];path=group/f'evaluation/{method}_c{index}.pt'
            assert sha(path)==row['sample_sha256'];stored=torch.load(path,map_location='cpu',weights_only=False)
            assert stored['condition']==row['condition']
            condition=row['condition'];numbers=condition['numbers'];x=stored['positions'];x0=stored['initial_positions']
            assert x.shape==x0.shape==(spec['samples_per_condition'],len(numbers),3)
            for value in [x,x0]:assert torch.isfinite(value).all() and value.mean(1).abs().max()<1e-8
            for j,seed in enumerate(stored['seeds']):
                assert seed==spec['evaluation_seed']*1000003+index*100003+j
                replay,tree=sample_source(prior,numbers,condition['charge'],condition['spin_multiplicity'],seed)
                torch.testing.assert_close(replay,x0[j],atol=1e-10,rtol=1e-10)
                assert tree==stored['auxiliary_tree_edges'][j]
                if prior is not None:
                    assert torch.isfinite(prior.log_prob(x0[j],numbers,condition['charge'],condition['spin_multiplicity']))
                    density_checks+=1
                source_checks+=1
            for k,v in assess(x,condition,list(range(len(x)))).items():assert row[k]==v,(method,index,k)
            assert geometry_counts(x0,numbers)==row['initial_geometry'] and geometry_counts(x,numbers)==row['final_geometry']
            rows.append(row)
    assert all(i==indices[0] for i in indices)
    metrics={}
    for method in ['warm']+spec['methods']:
        selected=[r for r in rows if r['method']==method]
        metrics[method]=dict(attempted=sum(r['attempted'] for r in selected),graph_supported=sum(r['graph_supported'] for r in selected),
            geometrically_supported=sum(r['geometrically_supported'] for r in selected),validator_errors=sum(r['validator_errors'] for r in selected),
            disconnected=sum(r['final_geometry']['disconnected'] for r in selected),overlap=sum(r['final_geometry']['overlap'] for r in selected),
            initial_disconnected=sum(r['initial_geometry']['disconnected'] for r in selected),initial_overlap=sum(r['initial_geometry']['overlap'] for r in selected),
            distinct_connectivities_sum=sum(r['distinct_connectivity'] for r in selected),generation_seconds=sum(r['generation_seconds'] for r in selected))
    comparisons={}
    for left,right in [('fixed','gaussian'),('node','fixed'),('pair','fixed'),('pair','node'),('node','gaussian'),('pair','gaussian')]:
        differences=[]
        for index in spec['conditions']:
            l=next(r for r in rows if r['method']==left and r['condition_index']==index)
            r=next(r for r in rows if r['method']==right and r['condition_index']==index)
            assert l['condition']==r['condition']
            differences.append([int(u['graph_supported'])-int(v['graph_supported']) for u,v in zip(l['records'],r['records'])])
        d=np.array(differences);rng=np.random.default_rng(88601)
        ids=rng.integers(d.shape[1],size=(10000,*d.shape));boot=np.take_along_axis(d[None],ids,axis=2).mean((1,2))
        comparisons[left+' minus '+right]=dict(graph_support_difference=float(d.mean()),conditional_paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),per_condition_counts=d.sum(1).tolist())
    write(a.out,dict(complete=True,protocol_sha256=digest,sources=sources,prior_report_sha256=sha(a.prior/'results.json'),
        prior_validation=prior_checks,training=training,matched_training_rows=len(indices[0]),initial_sources_replayed=source_checks,
        finite_tree_prior_densities_checked=density_checks,final_structural_outcomes_replayed=source_checks,methods=metrics,comparisons=comparisons,
        rows=[{k:v for k,v in r.items() if k!='records'} for r in rows],new_molecular_oracle_calls=0,scientific_submission_ready=False,
        limits=['One FM/prior training seed; bootstrap conditions on fitted models and fixed compositions.','Prior and FM optimizer trajectories and final neural generation are not independently retrained by this audit; saved prior validation, source sampling, training indices and final readouts are replayed.','Source spatial priors have normalized densities; final midpoint/noise outputs do not acquire qualified densities from that fact.','A connected latent tree is not a chemical bond graph or output connectivity guarantee.']))
    print(json.dumps(dict(methods=metrics,comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
