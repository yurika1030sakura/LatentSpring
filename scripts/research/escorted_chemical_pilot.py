#!/usr/bin/env python3
"""Bounded exact-path physical exchange pilot on the old chemical support."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.chemical_moves import exchange_terminal_sites
from cfm_mol.escorted_exchange import escorted_path
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.tempered_smc import DensityValue
from cfm_mol.chemical_path_guide import graph_guide_energy_force,exchanged_bond_graph


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,r):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--table',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--replica',type=int,choices=[0,1],required=True)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle-checkpoint',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    protocol_path=args.protocol or root/'research/evidence/escorted_chemical_protocol_v1.json';protocol=json.loads(protocol_path.read_text())
    guide=protocol.get('guide')
    header=json.loads((args.table/'results.json').read_text());data_path=args.table/'development.pt'
    if (not header['complete'] or header['protocol_sha256']!=protocol['source_table_protocol_sha256']
            or sha(data_path)!=header['artifacts']['development']):raise ValueError('Changed development starts')
    data=torch.load(data_path,map_location='cpu',weights_only=False)
    physical_path=root/'research/evidence/parity_training_protocol_v1.json';physical=json.loads(physical_path.read_text())
    if sha(physical_path)!=protocol['physical_protocol_sha256'] or sha(args.oracle_checkpoint)!=physical['raw_oracle_sha256']:
        raise ValueError('Changed physical target')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,scope=__doc__,replica=args.replica,condition=data['condition'],
        protocol_sha256=sha(protocol_path),development_sha256=sha(data_path),
        inherited_development_warm_raw_queries=header['streams']['development']['raw_queries'],
        inherited_source_raw_queries=512,inherited_training_queries=0,
        source_attempts=512,eligible_source_parents=len(data['source_parent_ids']),
        parent_ids=data['source_parent_ids'],reference_coordinates_loaded=False,
        scientific_submission_ready=False,history=[])
    write(output,report);oracle=None;target=None;local_rows=[];paths=[];history_ids=[];scale_choices=[]
    generator=torch.Generator().manual_seed(protocol['transition_seeds'][args.replica])
    action_rng=torch.Generator().manual_seed(protocol['action_seed']+args.replica)
    scale_rng=torch.Generator().manual_seed(protocol['scale_seed']+args.replica)
    start=time.perf_counter()
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=data['condition']['numbers'],charge=data['condition']['charge'],
            spin_multiplicity=data['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype')!='torch.float32':
            raise ValueError('Require qualified oracle precision')
        report['oracle_runtime']=oracle.handshake
        target=ChemicalTarget(oracle,data['condition'],data['kT_eV'],data['restraint_eV_A2'])
        basis=target.basis;n=len(target.numbers)
        initial=[data['states'][i] for i in data['warm_state_ids']]
        states=target.evaluate([target.coordinate_state(s['positions']) for s in initial],phase='initial')
        for new,old in zip(states,initial):
            torch.testing.assert_close(new['energy_eV'],old['energy_eV'],atol=1e-4,rtol=0)
            torch.testing.assert_close(new['force_eV_A'],old['force_eV_A'],atol=1e-4,rtol=0)
        def record(cycle):
            row=dict(cycle=cycle,energy_eV=[float(s['energy_eV']) for s in states],
                smiles=[s['graph']['connectivity_smiles'] for s in states],new_raw_queries=oracle.evaluated,
                total_raw_queries=oracle.evaluated+report['inherited_source_raw_queries']+report['inherited_development_warm_raw_queries'])
            report['history'].append(row);history_ids.append([s['state_id'] for s in states]);write(output,report)
            print(json.dumps(row),flush=True)
        def local(cycle,side):
            nonlocal states
            selected=torch.randint(len(protocol['local_scales']),(len(states),),generator=scale_rng)
            scales=torch.tensor(protocol['local_scales'],dtype=torch.float64)[selected]*target.kT**.5
            scale_choices.append(dict(cycle=cycle,side=side,indices=selected.tolist()))
            states,rows=target.transition(states,policy=None,generator=generator,proposal_std=scales,
                phase=f'cycle_{cycle}/local_{side}',local_only=True)
            local_rows.extend(rows)
        record(0)
        for cycle in range(protocol['cycles']):
            local(cycle,'pre');old=list(states);actions=[];inverses=[];forward_counts=[];choice_indices=[]
            for s in states:
                if not s['actions']:raise ValueError('Pilot requires eligible terminal exchanges')
                index=int(torch.randint(len(s['actions']),(1,),generator=action_rng))
                a=s['actions'][index];actions.append(a);inverses.append((a[0],a[1],a[3],a[2]))
                forward_counts.append(len(s['actions']));choice_indices.append(index)
            calls=[];last=[]
            old_bonds=torch.stack([s['graph']['bond_orders'] for s in old])
            new_bonds=torch.stack([exchanged_bond_graph(b,a) for b,a in zip(old_bonds,actions)]) if guide else old_bonds
            active_bonds=old_bonds;guide_energies=[];guide_forces=[];guide_graphs=[]
            def guidance(positions,bonds):
                if guide is None:return torch.zeros(len(positions),dtype=torch.float64),torch.zeros_like(positions)
                return graph_guide_energy_force(positions,bonds,target.radii,**guide)
            initial_guide_energy,initial_guide_force=guidance(torch.stack([s['positions'] for s in old]),old_bonds)
            guide_energies.append(initial_guide_energy);guide_forces.append(initial_guide_force);guide_graphs.append(old_bonds)
            def smooth(z,*,phase):
                nonlocal last
                positions=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
                # Intermediate states intentionally have no graph support claim.
                last=[dict(positions=x,graph=None,actions=[],charge=data['condition']['charge'],
                    spin_multiplicity=data['condition']['spin_multiplicity'],role='smooth_path_vertex') for x in positions]
                target.evaluate(last,phase=phase);calls.append([s['state_id'] for s in last])
                ge,gf=guidance(positions,active_bonds);guide_energies.append(ge);guide_forces.append(gf);guide_graphs.append(active_bonds)
                return DensityValue(-(torch.stack([s['potential_eV'] for s in last])+ge)/target.kT,
                    torch.stack([s['score'] for s in last])+torch.einsum('nk,bnd->bkd',basis,gf).reshape(len(z),-1)/target.kT)
            def transform(z):
                nonlocal active_bonds
                positions=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3));mapped=[];volumes=[]
                for x,a,inverse in zip(positions,actions,inverses):
                    y,volume,inv=exchange_terminal_sites(x,target.radii,a);assert inv==inverse
                    recovered,rv,_=exchange_terminal_sites(y,target.radii,inverse)
                    torch.testing.assert_close(recovered,x,atol=1e-10,rtol=1e-10)
                    torch.testing.assert_close(volume+rv,torch.zeros_like(volume),atol=1e-10,rtol=0)
                    mapped.append((basis.T@y).flatten());volumes.append(volume)
                active_bonds=new_bonds
                return torch.stack(mapped),torch.stack(volumes)
            z=torch.stack([(basis.T@s['positions']).flatten() for s in old])
            cached=DensityValue(-(torch.stack([s['potential_eV'] for s in old])+initial_guide_energy)/target.kT,
                torch.stack([s['score'] for s in old])+
                    torch.einsum('nk,bnd->bkd',basis,initial_guide_force).reshape(len(old),-1)/target.kT)
            y,_,path=escorted_path(z,smooth,transform,steps_per_side=protocol['steps_per_side'],
                std=protocol['path_scale']*target.kT**.5,max_score_norm=100/target.kT,
                generator=generator,initial_value=cached,phase=f'cycle_{cycle}/escort')
            endpoint_correction=(guide_energies[-1]-guide_energies[0])/target.kT
            target_ratio=path['smooth_log_acceptance_ratio']+endpoint_correction
            physical_ratio=-(torch.stack([s['potential_eV'] for s in last])-
                torch.stack([s['potential_eV'] for s in old]))/target.kT+path['path_log_ratio']+path['log_volume']
            torch.testing.assert_close(target_ratio,physical_ratio,atol=1e-7,rtol=1e-9)
            decisions=[];logu=torch.rand(len(states),dtype=torch.float64,generator=generator).log()
            for i,(candidate,inverse) in enumerate(zip(last,inverses)):
                decision=dict(chain=i,valid=False,accepted=False,old_state_id=old[i]['state_id'],
                    proposed_state_id=candidate['state_id'],action=actions[i],inverse_action=inverse,
                    choice_index=choice_indices[i],forward_count=forward_counts[i],log_uniform=float(logu[i]))
                try:
                    graph_fields=target.coordinate_state(candidate['positions']);candidate.update(graph_fields)
                    candidate['role']='supported_path_endpoint'
                    if guide and not torch.equal(candidate['graph']['bond_orders'],new_bonds[i]):
                        raise ValueError('Endpoint graph differs from the paired guide graph')
                    if inverse not in candidate['actions']:raise ValueError('Inverse action absent at endpoint')
                    reverse_count=len(candidate['actions']);correction=math.log(forward_counts[i]/reverse_count)
                    ratio=float(target_ratio[i])+correction
                    take=float(logu[i])<min(0.,ratio)
                    decision.update(valid=True,accepted=take,reverse_count=reverse_count,
                        action_log_ratio=correction,log_acceptance_ratio=ratio)
                    if take:states[i]=candidate
                except ValueError as exc:decision['rejection_reason']=str(exc)
                decisions.append(decision)
            path.update(cycle=cycle,vertex_state_ids=[[s['state_id'] for s in old]]+calls,decisions=decisions,
                guide=guide,guide_energies_eV=torch.stack(guide_energies),guide_forces_eV_A=torch.stack(guide_forces),
                guide_bond_orders=torch.stack(guide_graphs),physical_endpoint_log_correction=endpoint_correction,
                target_log_acceptance_ratio=target_ratio)
            paths.append(path);local(cycle,'post');record(cycle+1)
        artifact=dict(states=target.states,query_trace=target.query_trace,local_transitions=local_rows,
            paths=paths,history_state_ids=history_ids,scale_choices=scale_choices,
            generator_state=generator.get_state(),action_generator_state=action_rng.get_state(),
            scale_generator_state=scale_rng.get_state(),protocol_sha256=sha(protocol_path))
        torch.save(artifact,args.out/'trace.pt')
        maximum=protocol['maximum_new_raw_queries_per_replica']
        if oracle.evaluated>maximum:raise RuntimeError('Frozen query bound exceeded')
        report.update(complete=True,new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,
            seconds=time.perf_counter()-start,trace_sha256=sha(args.out/'trace.pt'),
            escorted_attempts=sum(len(p['decisions']) for p in paths),
            escorted_valid=sum(d['valid'] for p in paths for d in p['decisions']),
            escorted_accepted=sum(d['accepted'] for p in paths for d in p['decisions']))
        write(output,report)
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace,
            local_transitions=local_rows,paths=paths),args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
