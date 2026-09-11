#!/usr/bin/env python3
"""Frozen generated-parent-only action table and disjoint development starts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.parity_refinement import randomize_inversion


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle-checkpoint',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    protocol_path=root/'research/evidence/chemical_policy_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text())
    source=load_entropy_source(args.runs_root/'species_breadth_source_full_v2',0,
        protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
        manifest_path=root/'research/evidence/development_panel_v1.json')
    physical_path=root/'research/evidence/parity_training_protocol_v1.json'
    physical=json.loads(physical_path.read_text())
    if sha(physical_path)!=protocol['physical_protocol_sha256'] or sha(args.oracle_checkpoint)!=physical['raw_oracle_sha256']:
        raise ValueError('Physical target changed')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,protocol_sha256=sha(protocol_path),source_results_sha256=source['source_results_sha256'],
        condition=source['condition'],training_source_sha256=source['training_sha256'],
        development_source_sha256=source['evaluation_sha256'],reference_geometry_loaded=False,
        inherited_source_raw_queries=source['source']['oracle_evaluations'],
        source_generation_denominators={'training':4096,'development':512},streams={},
        scientific_submission_ready=False)
    write(output,report);oracle=None;target=None;artifacts={};start=time.perf_counter()
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=source['condition']['numbers'],charge=source['condition']['charge'],
            spin_multiplicity=source['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype')!='torch.float32':
            raise ValueError('Require qualified oracle precision')
        report['oracle_runtime']=oracle.handshake
        for stream_index,stream in enumerate(['training','development']):
            before=oracle.evaluated
            target=ChemicalTarget(oracle,source['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
            generator=torch.Generator().manual_seed(protocol['table_seed']+stream_index)
            x,signs=randomize_inversion(source[stream]['positions'],generator=generator)
            eligibility=[];states=[];parent_ids=[]
            for i,pos in enumerate(x):
                try:
                    state=target.coordinate_state(pos);state.update(parent_id=i,stream=stream,inversion_sign=int(signs[i]))
                    states.append(state);parent_ids.append(i)
                    eligibility.append(dict(parent_id=i,valid=True,smiles=state['graph']['connectivity_smiles']))
                except ValueError as exc:eligibility.append(dict(parent_id=i,valid=False,reason=str(exc)))
            if len(states)!=protocol['expected_eligible_parents'][stream]:
                raise ValueError(f'Eligibility changed for {stream}: {len(states)}')
            target.evaluate(states,phase='initial');initial_ids=[s['state_id'] for s in states]
            transitions=[];std=protocol['local_scale']*physical['kT_eV']**.5
            for step in range(protocol['warm_steps']):
                states,records=target.transition(states,policy=None,generator=generator,
                    proposal_std=std,phase=f'warm_{step}',local_only=True)
                transitions.extend(records)
            warm_ids=[s['state_id'] for s in states];warm_queries=oracle.evaluated-before
            table=[]
            if stream=='training':
                candidates=[];pending=[]
                for state_index,state in enumerate(states):
                    for action_index in range(1+len(state['actions'])):
                        repeats=protocol['local_quadrature'] if action_index==0 else 1
                        for repeat in range(repeats):
                            candidate,record=target.propose(state,action_index,generator,std)
                            record.update(source_index=state_index,parent_id=parent_ids[state_index],
                                quadrature_weight=1/repeats,repeat=repeat)
                            candidates.append(candidate);pending.append((state,record))
                target.evaluate([s for s in candidates if s is not None],phase='training_action_table')
                for (state,record),candidate in zip(pending,candidates):
                    table.append(target.finish_record(state,candidate,record,std))
            artifact=dict(stream=stream,condition=source['condition'],kT_eV=physical['kT_eV'],
                restraint_eV_A2=physical['restraint_eV_A2'],source_parent_ids=parent_ids,
                inversion_signs=signs,eligibility=eligibility,states=target.states,
                initial_state_ids=initial_ids,warm_state_ids=warm_ids,warm_transitions=transitions,
                table=table,query_trace=target.query_trace,generator_state=generator.get_state(),
                protocol_sha256=sha(protocol_path),source_results_sha256=source['source_results_sha256'])
            path=args.out/f'{stream}.pt';torch.save(artifact,path);artifacts[stream]=sha(path)
            report['streams'][stream]=dict(eligible=len(states),attempted_parents=len(x),
                warm_raw_queries=warm_queries,table_raw_queries=oracle.evaluated-before-warm_queries,
                raw_queries=oracle.evaluated-before,table_proposals=len(table),
                invalid_table_proposals=sum(not r['valid'] for r in table),
                invalid_warm_proposals=sum(not r['valid'] for r in transitions),
                warm_smiles=[s['graph']['connectivity_smiles'] for s in states],artifact_sha256=sha(path))
            write(output,report);print(json.dumps(report['streams'][stream]),flush=True)
        if oracle.evaluated>protocol['maximum_table_and_warm_raw_queries']:
            raise RuntimeError('Frozen query ceiling exceeded')
        report.update(complete=True,new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,
            artifacts=artifacts,seconds=time.perf_counter()-start)
        write(output,report)
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
