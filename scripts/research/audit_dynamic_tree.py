#!/usr/bin/env python3
"""Replay the matched dynamic-connection study without changing its frozen gate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from flowmol.model_utils.load import model_from_config,read_config_file
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source,geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import intervals,flags
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    training_audit_path=a.project/'research/evidence/dynamic_tree_training_audit_v1.json'
    training_audit=json.loads(training_audit_path.read_text());assert training_audit['complete']
    summaries={};all_rows={};artifacts=[];costs={};replayed=0
    for seed in [0,1]:
        protocol_path=a.project/f'research/evidence/dynamic_tree_s{seed}_v1.json'
        spec=json.loads(protocol_path.read_text());ph=sha(protocol_path)
        root=a.run/f's{seed}/study';progress=json.loads((root/'progress.json').read_text())
        assert progress['complete'] and progress['protocol_sha256']==ph and progress['completed']==spec['methods']
        selection=json.loads((root/'selection.json').read_text());ids=selection['selected'][:spec['fm_steps']]
        independent=training_audit['rows'][seed]
        assert independent['seed']==seed and independent['overlap']==0 and independent['protocol_sha256']==ph
        assert independent['selection_sha256']==sha(root/'selection.json')
        assert len(ids)==spec['fm_steps'] and selection['excluded_compositions']>0
        panel_path=a.project/spec['condition_manifest'];assert sha(panel_path)==spec['condition_manifest_sha256']
        panel=json.loads(panel_path.read_text());assert len(panel['rows'])==12
        cfg_path=a.project/spec['config'];assert sha(cfg_path)==spec['config_sha256']
        cfg=read_config_file(cfg_path);cfg['mol_fm'].pop('bgfm',None)
        assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
        summaries[seed]={};all_rows[seed]={};costs[seed]={};initials={};orders=set()
        for method in spec['methods']:
            directory=root/method;validation=json.loads((directory/'validation.json').read_text())
            file=directory/'last.ckpt';assert sha(file)==validation['checkpoint_sha256']
            state=torch.load(file,map_location='cpu',weights_only=False);recipe=state['research_protocol']
            assert state['global_step']==spec['fm_steps'] and recipe['tree_protocol_sha256']==ph
            assert recipe['source_prior_kind']=='fixed' and not recipe.get('latent_tree_context')
            orders.add(recipe['data_order_sha256'])
            metrics=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
            assert [r['processed_index'] for r in metrics]==ids
            assert [r['step'] for r in metrics]==list(range(1,spec['fm_steps']+1))
            model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True)
            prior=prior_from_checkpoint(state)
            learning={}
            if method!='fixed':
                assert recipe['dynamic_tree_attention']['mode']==method
                initial=torch.load(directory/'dynamic_initial.pt',map_location='cpu',weights_only=False);initials[method]=initial
                final=model.vector_field.dynamic_tree_attention.state_dict()
                for name in ['affinity.2.weight','messages.2.weight']:
                    learning[name]=float((final[name]-initial[name]).norm())
                assert learning['messages.2.weight']>0
                if method=='tree_fixed':
                    assert all(torch.equal(final[k],initial[k]) for k in initial if k.startswith(('affinity.','node_summary.')))
                else:assert learning['affinity.2.weight']>0
            del model,state
            report_path=root/'evaluation'/f'{method}_results.json'
            report=json.loads(report_path.read_text());assert report['complete'] and report['protocol_sha256']==ph
            assert report['checkpoint_sha256']==validation['checkpoint_sha256'] and len(report['rows'])==12
            for i,row in enumerate(report['rows']):
                file=root/'evaluation'/f'{method}_c{i}.pt';assert sha(file)==row['sample_sha256']
                data=torch.load(file,map_location='cpu',weights_only=False);c=data['condition'];x=data['positions'];x0=data['initial_positions']
                assert c['composition_hex']==panel['rows'][i]['composition_hex'] and len(x)==len(x0)==64
                assert torch.isfinite(x).all() and float(x.mean(1).abs().max())<1e-8
                expected=[spec['evaluation_seed']*1000003+i*100003+j for j in range(64)]
                assert data['seeds']==expected
                for j,draw_seed in enumerate(expected):
                    position,tree=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed)
                    torch.testing.assert_close(position,x0[j],rtol=0,atol=0)
                    assert tree==data['auxiliary_tree_edges'][j];replayed+=1
                assessment=assess(x,c,list(range(64)))
                assert all(row[k]==assessment[k] for k in assessment)
                assert geometry_counts(x,c['numbers'])==row['final_geometry']
                assert geometry_counts(x0,c['numbers'])==row['initial_geometry']
            rows=report['rows'];all_rows[seed][method]=rows
            summaries[seed][method]=dict(attempts=sum(r['attempted'] for r in rows),graph=sum(r['graph_supported'] for r in rows),
                geometry=sum(r['geometrically_supported'] for r in rows),distinct_connectivity=sum(r['distinct_connectivity'] for r in rows),
                validator_errors=sum(r['validator_errors'] for r in rows),learning=learning)
            costs[seed][method]=dict(training_seconds=validation['seconds'],generation_seconds=sum(r['generation_seconds'] for r in rows))
            artifacts.append(dict(seed=seed,method=method,checkpoint_sha256=validation['checkpoint_sha256'],report_sha256=sha(report_path),
                training_metrics_sha256=sha(directory/'metrics.jsonl'),selection_sha256=sha(root/'selection.json')))
            print(json.dumps(dict(seed=seed,method=method,**summaries[seed][method])),flush=True)
        assert len(orders)==1
        reference=initials['tree_learned']
        for values in initials.values():assert all(torch.equal(values[k],reference[k]) for k in reference)
    rng=np.random.default_rng(31891);comparisons={}
    strata=[0 if c['n_atoms']<=16 else 1 if c['n_atoms']<=28 else 2 for c in panel['rows']]
    for comparator in ['fixed','local_learned','tree_fixed']:
        comparisons[comparator]={}
        for metric in ['graph_supported','geometrically_supported']:
            diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(all_rows[s]['tree_learned'],all_rows[s][comparator])]) for s in [0,1]])
            comparisons[comparator][metric]=intervals(diff,strata,rng)
    point=all(summaries[s]['tree_learned']['graph']>summaries[s][m]['graph'] for s in [0,1] for m in ['fixed','local_learned'])
    bounds=all(comparisons[m]['graph_supported']['paired_draw95'][0]>0 for m in ['fixed','local_learned'])
    distinct=sum(summaries[s]['tree_learned']['distinct_connectivity']-summaries[s]['fixed']['distinct_connectivity'] for s in [0,1])/1536
    write(a.out,dict(complete=True,summary=summaries,comparisons=comparisons,costs=costs,artifacts=artifacts,
        source_and_structural_attempts_replayed=replayed,matched_training_indices=True,initial_adapter_states_equal=True,full_checkpoint_restoration=True,
        training_exclusion_audit_sha256=sha(training_audit_path),development_gate_passed=point and bounds and distinct>=-.02,new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='Reused12-composition monomer development panel with fresh draws and two new matched training seeds. No optimizer/decoder integration replay, target density or Boltzmann certification. All selected training count vectors independently checked against the frozen exclusion manifests.'))


if __name__=='__main__':main()
