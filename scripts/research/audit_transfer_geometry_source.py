#!/usr/bin/env python3
"""Audit a frozen geometry-only panel, including zero-support conditions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from cfm_mol.condition_systems import load_condition
from cfm_mol.chemical_moves import infer_chemical_graph, covalent_radii, terminal_exchange_actions
from cfm_mol.geometric_domain import connected_nonoverlapping
from scripts.research.evaluate_chemical_policy import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--manifest',type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    pp=args.protocol or root/'research/evidence/transfer_geometry_source_protocol_v1.json'
    protocol=json.loads(pp.read_text())
    mp=args.manifest or root/'research/evidence/transfer_development_panel_v1.json'
    manifest=json.loads(mp.read_text())
    assert sha(mp)==protocol['manifest_sha256'] and manifest['complete'] and manifest['role']=='new_development'
    candidate_path=root/'research/evidence/official_development_candidates.json'
    assert sha(candidate_path)==manifest['source_candidates_sha256']
    candidates=json.loads(candidate_path.read_text())
    old=json.loads((root/'research/evidence/development_panel_v1.json').read_text())
    old_compositions={r['composition_hex'] for r in old['rows']}
    stream=protocol.get('stream','fresh_development')
    assert stream in {'fresh_training','fresh_development'}
    excluded_compositions=set(old_compositions)
    if stream=='fresh_training':
        assert manifest['intended_use']=='proposal_training'
        ep=root/'research/evidence/transfer_development_panel_v1.json'
        assert sha(ep)==manifest['excluded_evaluation_manifest_sha256']
        excluded_compositions.update(r['composition_hex'] for r in json.loads(ep.read_text())['rows'])
        # Reproduce the frozen metadata-only selection before auditing geometry.
        from scripts.research.freeze_proposal_training_panel import select
        selected,counts=select(candidates['rows'],excluded_compositions,manifest['seed'])
        assert selected==manifest['rows'] and counts==manifest['eligible_compositions_by_size']
    assert len(manifest['rows'])==len(protocol['condition_indices'])
    rows=[];all_seeds=[]
    for index in protocol['condition_indices']:
        directory=args.run/f'condition_{index:02d}'
        report=json.loads((directory/'results.json').read_text())
        assert report['complete'] and report['protocol_sha256']==sha(pp) and report['stream']==stream
        assert report['physical_queries']==0 and not report['energy_labels_computed'] and not report['source_density_available']
        assert report['prefix_replay_max_error_A']==0 and report['global_rng_unchanged'] and report['input_positions_unchanged']
        condition=load_condition(mp,index)
        condition.update(requested_kT_eV=1.,numbers=condition['atomic_numbers'])
        assert condition==report['condition'] and 'energy_eV' not in condition
        original=candidates['rows'][condition['candidate_index']]
        assert original['partition']=='new_development' and original['composition_hex'] not in excluded_compositions
        for key,value in original.items():
            if key!='energy_eV':assert condition[key]==value
        assert sha(directory/'samples.pt')==report['samples_sha256']
        data=torch.load(directory/'samples.pt',map_location='cpu',weights_only=False)
        assert data['condition']==condition and data['stream']==stream
        assert not set(data).intersection(['energy_eV','log_q','importance_weights','work'])
        assert data['sample_ids']==list(range(protocol['count']))
        seeds=[1000000000*protocol['seed_namespace']+100003*condition['candidate_index']+100000003*protocol['stream_number']+i for i in range(protocol['count']//protocol['batch'])]
        assert seeds==data['batch_seeds']==report['batch_seeds']
        all_seeds.extend(seeds)
        for chunk_index,seed in enumerate(seeds):
            name=f'positions_{chunk_index:05d}.pt';path=directory/'chunks'/name
            assert sha(path)==report['chunks'][name]
            chunk=torch.load(path,map_location='cpu',weights_only=False)
            begin=chunk_index*protocol['batch'];end=begin+protocol['batch']
            assert chunk['seed']==seed and chunk['sample_ids']==list(range(begin,end))
            assert chunk['condition']==condition and chunk['stream']==stream
            torch.testing.assert_close(chunk['positions'],data['positions'][begin:end],atol=0,rtol=0)
        x=data['positions'];assert torch.isfinite(x).all() and float(x.mean(1).abs().max())<1e-8
        geometric=connected_nonoverlapping(x,covalent_radii(condition['numbers']))
        records=[];identities=Counter()
        for parent in geometric.nonzero().flatten().tolist():
            try:
                graph=infer_chemical_graph(x[parent],condition['numbers'],condition['charge'])
                actions=[a for a in terminal_exchange_actions(condition['numbers'],graph['bond_orders']) if a[2]!=a[3]]
                row=dict(parent_id=parent,supported=True,smiles=graph['connectivity_smiles'],
                    eligible_exchange_count=len(actions),radical_electrons=sum(graph['radical_electrons']))
                identities[row['smiles']]+=1
            except (ValueError,IndexError,RuntimeError) as exc:
                row=dict(parent_id=parent,supported=False,error_type=type(exc).__name__,reason=str(exc),
                    validator_error=not isinstance(exc,ValueError))
            records.append(row)
        rows.append(dict(index=index,condition=condition,source_results_sha256=sha(directory/'results.json'),
            samples_sha256=sha(directory/'samples.pt'),attempted=len(x),geometrically_supported=int(geometric.sum()),
            chemically_supported=sum(r['supported'] for r in records),
            supported_parent_ids=[r['parent_id'] for r in records if r['supported']],
            with_eligible_exchange=sum(r.get('eligible_exchange_count',0)>0 for r in records),
            connectivity_counts=dict(identities),validator_errors=sum(r.get('validator_error',False) for r in records),
            records=records,geometrically_rejected_parent_ids=(~geometric).nonzero().flatten().tolist(),
            generation_seconds=report['seconds'],neural_molecule_evaluations=report['generation_neural_field_calls'],
            additional_replay_neural_molecule_evaluations=report['additional_replay_neural_field_calls']))
        print(json.dumps({k:rows[-1][k] for k in ['index','attempted','chemically_supported','with_eligible_exchange','validator_errors']}),flush=True)
    assert len(all_seeds)==len(set(all_seeds)) and len({r['condition']['composition_hex'] for r in rows})==len(protocol['condition_indices'])
    result=dict(complete=True,protocol_sha256=sha(pp),manifest_sha256=sha(mp),rows=rows,
        all_conditions_retained=True,all_chunk_rows_verified=True,new_physical_queries=0,stream=stream,
        scientific_submission_ready=False,scope='Frozen development-pool compositions with explicit stream role, selected before geometry/energy outcomes. No learned method outcome is assessed here; preserve every zero-support composition and validator failure.')
    if len(rows)==6:result['all_six_conditions_retained']=True
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
