#!/usr/bin/env python3
"""Controlled terminal-site exchange versus local MALA from identical warm starts.

This is an additional physical-transition diagnostic. Initialization cost and
reference-origin chains are retained; no new generator or equilibrium claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii,terminal_exchange_actions,exchange_terminal_sites
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.parity_refinement import evaluate_even_potential
from cfm_mol.supported_mcmc import supported_mala_population
from cfm_mol.tempered_smc import DensityValue


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,r):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--method',choices=['mala','exchange'],required=True);p.add_argument('--replica',type=int,choices=[0,1],required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle-checkpoint',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    protocol_path=root/'research/evidence/terminal_exchange_protocol_v1.json';protocol=json.loads(protocol_path.read_text())
    parent_dir=args.runs_root/protocol['parent_run'];parent=json.loads((parent_dir/'results.json').read_text())
    if not parent['complete'] or sha(parent_dir/'results.json')!=protocol['parent_results_sha256']:
        raise ValueError('Unqualified warm-start source')
    trace_path=parent_dir/'reference_trace.pt'
    if sha(trace_path)!=parent['artifact_sha256']:raise ValueError('Changed warm-start trace')
    trace=torch.load(trace_path,map_location='cpu',weights_only=False)
    condition=parent['condition'];numbers=condition['numbers'];n=len(numbers);radii=covalent_radii(numbers)
    physical_path=root/'research/evidence/parity_training_protocol_v1.json';physical=json.loads(physical_path.read_text())
    if sha(physical_path)!=protocol['physical_protocol_sha256'] or sha(args.oracle_checkpoint)!=physical['raw_oracle_sha256']:
        raise ValueError('Changed physical target or oracle')
    kT=physical['kT_eV'];restraint=physical['restraint_eV_A2'];basis=centered_orthonormal_basis(n)
    x=trace['snapshots'][-1]['cold_positions'].clone();initial=x.clone();chains=len(x)
    z=torch.einsum('nk,bnd->bkd',basis,x).reshape(chains,-1)
    zero=float(trace['snapshots'][-1]['energies_eV'][:,0].mean())
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,scope=__doc__,method=args.method,replica=args.replica,condition=condition,
        initialization_families=parent['initialization_families'],parent_results_sha256=sha(parent_dir/'results.json'),
        inherited_reference_raw_queries=parent['raw_queries'],protocol_sha256=sha(protocol_path),
        history=[],exchange_attempts=[],scientific_submission_ready=False,
        limitations=['Uniform chemistry-informed exchange is a baseline, not AI novelty.',
            'Warm starts already consumed reference sampling; reported new queries are additional.',
            'RDKit graph validity is not a quantum or equilibrium certificate.'])
    write(output,report);start=time.perf_counter();oracle=None;snapshots=[];last_graphs=[]
    generator=torch.Generator().manual_seed(protocol['transition_seed']+args.replica)
    choice_generator=torch.Generator().manual_seed(protocol['choice_seed']+args.replica)
    def coordinates(z):return torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
        report['oracle_runtime']=oracle.handshake
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype')!='torch.float32':
            raise ValueError('Require qualified physical-oracle precision')
        def scored(positions):
            energy,force,_=evaluate_even_potential(oracle,positions)
            return DensityValue(-(energy-zero+restraint/2*positions.square().sum((1,2)))/kT,
                torch.einsum('nk,bnd->bkd',basis,(force-restraint*positions)/kT).reshape(len(positions),-1))
        def target(z):
            nonlocal last_graphs
            positions=coordinates(z);last_graphs=[]
            for pos in positions:
                try:last_graphs.append(infer_chemical_graph(pos,numbers,condition['charge']))
                except ValueError:last_graphs.append(None)
            valid=torch.tensor([g is not None for g in last_graphs])
            value=DensityValue(torch.full((len(z),),-torch.inf,dtype=z.dtype),torch.zeros_like(z))
            if valid.any():
                part=scored(positions[valid]);value.log_value[valid]=part.log_value;value.score[valid]=part.score
            return value
        value=target(z).validate(z,True);graphs=last_graphs
        initial_energy=zero-kT*value.log_value-restraint/2*coordinates(z).square().sum((1,2))
        torch.testing.assert_close(initial_energy,trace['snapshots'][-1]['energies_eV'][:,0],atol=1e-4,rtol=0)
        def record(cycle):
            positions=coordinates(z)
            energy=zero-kT*value.log_value-restraint/2*positions.square().sum((1,2))
            row=dict(cycle=cycle,energy_eV=energy.tolist(),smiles=[g['connectivity_smiles'] for g in graphs],raw_queries=oracle.evaluated)
            report['history'].append(row);snapshots.append(dict(positions=positions.clone(),energy_eV=energy.clone(),smiles=row['smiles']))
            write(output,report);print(json.dumps(row),flush=True)
        record(0)
        for cycle in range(protocol['cycles']):
            for stage in range(3):
                if args.method=='mala' or stage!=1:
                    z,value,stats=supported_mala_population(z,target,steps=1,
                        proposal_std=.1*math.sqrt(kT),max_score_norm=100/kT,generator=generator,initial_value=value)
                    graphs=[last_graphs[i] if take else graphs[i] for i,take in enumerate(stats['accepted_per_chain'])]
                    continue
                positions=coordinates(z);proposals=positions.clone();candidates=[]
                for i in range(chains):
                    actions=terminal_exchange_actions(numbers,graphs[i]['bond_orders'])
                    row=dict(cycle=cycle+1,chain=i,accepted=False,actions=len(actions))
                    candidate=dict(row=row,valid=False)
                    if actions:
                        action=actions[int(torch.randint(len(actions),(1,),generator=choice_generator))]
                        y,volume,inverse_action=exchange_terminal_sites(positions[i],radii,action)
                        row.update(action=list(action),reverse_action=list(inverse_action),log_volume=float(volume))
                        try:
                            graph=infer_chemical_graph(y,numbers,condition['charge'])
                            reverse_actions=terminal_exchange_actions(numbers,graph['bond_orders'])
                            if inverse_action not in reverse_actions:raise ValueError('Reverse action not eligible')
                            recovered,reverse_volume,_=exchange_terminal_sites(y,radii,inverse_action)
                            torch.testing.assert_close(recovered,positions[i],atol=1e-10,rtol=1e-10)
                            torch.testing.assert_close(volume+reverse_volume,torch.zeros_like(volume),atol=1e-10,rtol=0)
                            row.update(reverse_actions=len(reverse_actions),new_smiles=graph['connectivity_smiles'])
                            candidate.update(valid=True,graph=graph,correction=float(volume)+math.log(len(actions)/len(reverse_actions)))
                            proposals[i]=y
                        except ValueError as exc:row['rejection_reason']=str(exc)
                    else:row['rejection_reason']='No eligible unlike terminal sites'
                    candidates.append(candidate)
                indices=torch.tensor([i for i,c in enumerate(candidates) if c['valid']],dtype=torch.long)
                if len(indices):
                    proposed=scored(proposals[indices])
                    ratio=proposed.log_value-value.log_value[indices]+torch.tensor([candidates[i]['correction'] for i in indices.tolist()],dtype=z.dtype)
                    take=torch.rand(len(indices),dtype=z.dtype,generator=generator).log()<ratio.clamp_max(0)
                    for at,i in enumerate(indices.tolist()):
                        candidates[i]['row'].update(log_acceptance_ratio=float(ratio[at]),accepted=bool(take[at]))
                        if take[at]:
                            z[i]=(basis.T@proposals[i]).flatten();value.log_value[i]=proposed.log_value[at];value.score[i]=proposed.score[at]
                            graphs[i]=candidates[i]['graph']
                report['exchange_attempts'].extend(c['row'] for c in candidates)
            record(cycle+1)
        torch.save(dict(initial_positions=initial,snapshots=snapshots),args.out/'samples.pt')
        maximum_queries=2*chains*(1+3*protocol['cycles'])
        if oracle.evaluated>maximum_queries:raise RuntimeError('Diagnostic query bound exceeded')
        report.update(complete=True,new_raw_queries=oracle.evaluated,seconds=time.perf_counter()-start,
            maximum_new_raw_queries=maximum_queries,
            artifact_sha256=sha(args.out/'samples.pt'))
        write(output,report)
    except Exception as exc:
        torch.save(dict(initial_positions=initial,snapshots=snapshots),args.out/'failed_samples.pt')
        report.update(complete=False,failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else None)
        write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
