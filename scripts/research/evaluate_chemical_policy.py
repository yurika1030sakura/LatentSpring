#!/usr/bin/env python3
"""Frozen learned/uniform chemical kernels on disjoint generated development starts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from cfm_mol.chemical_sampler import ChemicalTarget,policy_log_probabilities
from cfm_mol.chemical_policy import ChemicalMovePolicy
from cfm_mol.energy_oracle import EnergyOracle


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,r):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table',type=Path,required=True);p.add_argument('--policies',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--replica',type=int,choices=[0,1],required=True)
    p.add_argument('--method',choices=['uniform','learned'],required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle-checkpoint',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    protocol_path=root/'research/evidence/chemical_policy_protocol_v1.json';protocol=json.loads(protocol_path.read_text())
    header=json.loads((args.table/'results.json').read_text());data_path=args.table/'development.pt'
    if not header['complete'] or header['protocol_sha256']!=sha(protocol_path) or sha(data_path)!=header['artifacts']['development']:
        raise ValueError('Development-start provenance failed')
    data=torch.load(data_path,map_location='cpu',weights_only=False)
    if data['stream']!='development':raise ValueError('Development stream required')
    physical_path=root/'research/evidence/parity_training_protocol_v1.json';physical=json.loads(physical_path.read_text())
    if sha(physical_path)!=protocol['physical_protocol_sha256'] or sha(args.oracle_checkpoint)!=physical['raw_oracle_sha256']:
        raise ValueError('Physical target changed')
    policy=None;trained=None
    if args.method=='learned':
        directory=args.policies/f'replica_{args.replica}';trained=json.loads((directory/'results.json').read_text())
        if not trained['complete'] or trained['protocol_sha256']!=sha(protocol_path) or sha(directory/'policy.pt')!=trained['checkpoint_sha256']:
            raise ValueError('Policy provenance failed')
        if trained['training_table_sha256']!=header['artifacts']['training']:raise ValueError('Policy used another training table')
        checkpoint=torch.load(directory/'policy.pt',map_location='cpu',weights_only=False)
        policy=ChemicalMovePolicy(**checkpoint['configuration']).double();policy.load_state_dict(checkpoint['state_dict']);policy.eval()
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,method=args.method,replica=args.replica,protocol_sha256=sha(protocol_path),
        development_sha256=sha(data_path),condition=data['condition'],development_parent_ids=data['source_parent_ids'],
        policy_sha256=trained['checkpoint_sha256'] if trained else None,
        inherited_training_raw_queries=trained['inherited_training_raw_queries'] if trained else 0,
        inherited_development_warm_raw_queries=header['streams']['development']['raw_queries'],
        inherited_source_raw_queries=header['inherited_source_raw_queries'],
        source_generation_denominators=header['source_generation_denominators'],
        reference_coordinates_loaded=False,scientific_submission_ready=False,history=[])
    write(output,report);target=None;oracle=None;transitions=[];history_ids=[];start=time.perf_counter()
    generator=torch.Generator().manual_seed(protocol['evaluation_seeds'][args.replica])
    steps=protocol['evaluation_steps'] if policy is not None else protocol['uniform_total_budget_steps']
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=data['condition']['numbers'],charge=data['condition']['charge'],
            spin_multiplicity=data['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype')!='torch.float32':
            raise ValueError('Require qualified oracle precision')
        report['oracle_runtime']=oracle.handshake
        target=ChemicalTarget(oracle,data['condition'],data['kT_eV'],data['restraint_eV_A2'])
        initial=[data['states'][i] for i in data['warm_state_ids']]
        states=target.evaluate([target.coordinate_state(s['positions']) for s in initial],phase='initial')
        for new,old in zip(states,initial):
            torch.testing.assert_close(new['energy_eV'],old['energy_eV'],atol=1e-4,rtol=0)
            torch.testing.assert_close(new['force_eV_A'],old['force_eV_A'],atol=1e-4,rtol=0)
        for step in range(steps+1):
            with torch.no_grad():local=policy_log_probabilities(states,target.numbers,target.kT,policy)[:,0].exp().tolist()
            row=dict(step=step,energy_eV=[float(s['energy_eV']) for s in states],
                potential_eV=[float(s['potential_eV']) for s in states],
                smiles=[s['graph']['connectivity_smiles'] for s in states],local_probability=local,
                new_raw_queries=oracle.evaluated,total_raw_queries=oracle.evaluated+report['inherited_training_raw_queries'])
            report['history'].append(row);history_ids.append([s['state_id'] for s in states])
            if step%32==0 or step==steps:
                print(json.dumps(row),flush=True);write(output,report)
            if step==steps:break
            states,records=target.transition(states,policy=policy,generator=generator,
                proposal_std=protocol['local_scale']*target.kT**.5,phase=f'evaluation_{step}')
            transitions.extend(records)
        artifact=dict(states=target.states,query_trace=target.query_trace,transitions=transitions,
            history_state_ids=history_ids,generator_state=generator.get_state(),
            policy_sha256=report['policy_sha256'],protocol_sha256=sha(protocol_path))
        torch.save(artifact,args.out/'trace.pt')
        maximum=protocol['maximum_raw_queries_per_learned_eval_arm'] if policy is not None else protocol['maximum_raw_queries_per_uniform_total_budget_arm']
        if oracle.evaluated>maximum:raise RuntimeError('Frozen evaluation query bound exceeded')
        report.update(complete=True,new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,
            seconds=time.perf_counter()-start,trace_sha256=sha(args.out/'trace.pt'),
            proposals=len(transitions),invalid_proposals=sum(not r['valid'] for r in transitions),
            accepted_proposals=sum(r['accepted'] for r in transitions),
            limitation='Only four generated development parents for one composition and two seeds; no equilibrium or generalization certificate.')
        write(output,report)
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace,transitions=transitions),args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
