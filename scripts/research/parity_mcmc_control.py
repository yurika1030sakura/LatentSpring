#!/usr/bin/env python3
"""Fixed-query physical MALA/HMC control on the actual FM inversion-mixture source.

All transition acceptance values use both raw orientations. The finite-time
endpoint law, absolute density and importance ESS remain unknown.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.fixed_target_mcmc import budgeted_population
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.parity_refinement import evaluate_even_potential
from cfm_mol.tempered_smc import DensityValue


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, report):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--kernel',choices=['mala','hmc'],required=True)
    parser.add_argument('--replica',type=int,choices=[0,1],default=0)
    parser.add_argument('--oracle-python',type=Path,required=True)
    parser.add_argument('--oracle-checkpoint',type=Path,required=True)
    parser.add_argument('--engineering-smoke',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    protocol_path=root/'research/evidence/parity_mcmc_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text())
    physical_path=root/'research/evidence/parity_training_protocol_v1.json'
    physical=json.loads(physical_path.read_text())
    parent_path=args.samples.parent/'results.json';parent=json.loads(parent_path.read_text())
    if (not protocol['frozen'] or sha(physical_path)!=protocol['physical_protocol_sha256']
            or sha(args.samples)!=protocol['base_samples_sha256']
            or sha(parent_path)!=protocol['parent_results_sha256']
            or sha(args.oracle_checkpoint)!=physical['raw_oracle_sha256']):
        raise ValueError('Frozen source, physical target or oracle changed')
    if not parent['complete'] or parent['artifacts'][args.samples.name]!=sha(args.samples):
        raise ValueError('Require an intact qualified source-parent artifact')
    source=torch.load(args.samples,map_location='cpu',weights_only=False)
    condition=source['condition'];n=len(condition['numbers'])
    scope=protocol['engineering'] if args.engineering_smoke else protocol['production']
    count,updates,tail_count=scope['parents'],scope['force_updates'],scope['tail_parents']
    x0=source['positions'][:count].double()
    if x0.shape!=(count,n,3) or float(x0.mean(1).abs().max())>1e-10:
        raise ValueError('Invalid prescribed COM source parents')
    basis=centered_orthonormal_basis(n)
    z0=torch.einsum('nk,bnd->bkd',basis,x0).reshape(count,-1)
    kT=physical['kT_eV'];restraint=physical['restraint_eV_A2']
    zero=float(source['energy_eV'].mean())
    recipe=protocol['kernels'][args.kernel]
    step_size=recipe['step_size_sqrt_kT']*math.sqrt(kT)
    score_cap=protocol['physical_force_norm_cap_eV_A']/kT
    seed=recipe['seed_base']+args.replica
    tail_indices=torch.randperm(count,generator=torch.Generator().manual_seed(protocol['tail_selection_seed']))[:tail_count]
    expected=2*(count*(1+updates)+tail_count)
    if expected!=scope['raw_oracle_evaluations']:raise ValueError('Protocol budget arithmetic differs')
    source_keys=['source_results_sha256','source_checkpoint_sha256','training_sha256','evaluation_sha256',
                 'source_protocol_sha256','refinement_protocol_sha256','source_kind','target_kind']
    report={key:parent[key] for key in source_keys}
    report.update(complete=False,scope=__doc__,kind='physical_'+args.kernel,condition=condition,
        engineering_only=args.engineering_smoke,replica=args.replica,
        protocol_sha256=sha(protocol_path),base_samples_sha256=sha(args.samples),
        initialization='exact source parents; no trained adapter or reference geometry',
        kernel=args.kernel,seed=seed,parents=count,force_updates_common=updates,
        tail_indices=tail_indices.tolist(),hmc_length=protocol['hmc_length'],
        step_size_A=step_size,max_score_norm_per_A=score_cap,kT_eV=kT,restraint_eV_A2=restraint,
        energy_zero_eV=zero,expected_oracle_evaluations=expected,history=[],
        source_sha256={str(p.relative_to(root)):sha(p) for p in [Path(__file__).resolve(),
            root/'cfm_mol/fixed_target_mcmc.py',root/'cfm_mol/parity_refinement.py',
            root/'cfm_mol/energy_oracle.py',root/'scripts/research/oracle_worker.py']},
        scientific_submission_ready=False,limitations=[
            'Target-invariant transitions do not certify finite-time stationarity or mode coverage.',
            'No endpoint density, KL change, normalizer or importance ESS is assigned.',
            'This budget measures fresh512-parent refinement; learned-method amortization is a separate comparison.',
            'Step sizes are inherited fixed choices, not tuned to the new target outcomes.'])
    args.out.mkdir(parents=True,exist_ok=True);write(output,report)
    start=time.perf_counter();trace=[];snapshots=[];oracle=None
    def coordinates(z):
        return torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],
            device='cuda',batch_size=32)
        report['oracle_runtime']=oracle.handshake
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype')!='torch.float32':
            raise ValueError('Require the qualified full-precision GPU oracle')
        def target(z):
            entry=dict(latent=z.clone(),complete=False);trace.append(entry)
            x=coordinates(z)
            energy,force,components=evaluate_even_potential(oracle,x)
            value=DensityValue(-(energy-zero+restraint/2*x.square().sum((1,2)))/kT,
                torch.einsum('nk,bnd->bkd',basis,(force-restraint*x)/kT).reshape_as(z))
            entry.update(complete=True,log_value=value.log_value.clone(),score=value.score.clone(),
                         energy_eV=energy,force_eV_A=force,**components)
            if len(trace)==1:
                error=float((energy-source['energy_eV'][:count]).abs().max())
                report['initial_energy_replay_max_error_eV']=error
                if error>1e-4:raise ValueError('Source energy replay exceeds qualified GPU tolerance')
            return value
        def record(phase,iteration,z,value,accepted,evaluations):
            x=coordinates(z);energy=zero-kT*value.log_value-restraint/2*x.square().sum((1,2))
            row=dict(phase=phase,iteration=iteration,parents=len(z),mean_energy_eV=float(energy.mean()),
                     accepted_per_chain=accepted.cpu().tolist(),phase_target_evaluations=evaluations,
                     acknowledged_raw_queries=oracle.evaluated,seconds=time.perf_counter()-start)
            report['history'].append(row)
            snapshots.append(dict(phase=phase,iteration=iteration,positions=x.clone(),energy_eV=energy.clone()))
            write(output,report)
            print(json.dumps({k:v for k,v in row.items() if k!='accepted_per_chain'}),flush=True)
        final,value,stats=budgeted_population(z0,target,kernel=args.kernel,force_updates=updates,
            tail_indices=tail_indices,step_size=step_size,max_score_norm=score_cap,seed=seed,
            hmc_length=protocol['hmc_length'],callback=record)
        if oracle.evaluated!=expected or oracle.evaluated!=2*stats['target_evaluations']:
            raise RuntimeError('Actual physical-query budget differs')
        # Reproduce every proposal and acceptance from recorded target values.
        iterator=iter(trace)
        def replay_target(z):
            entry=next(iterator)
            torch.testing.assert_close(z,entry['latent'],atol=1e-10,rtol=1e-10)
            return DensityValue(entry['log_value'],entry['score'])
        replay,replay_value,replay_stats=budgeted_population(z0,replay_target,kernel=args.kernel,
            force_updates=updates,tail_indices=tail_indices,step_size=step_size,
            max_score_norm=score_cap,seed=seed,hmc_length=protocol['hmc_length'])
        if list(iterator) or replay_stats!=stats:raise RuntimeError('Stochastic replay trace differs')
        torch.testing.assert_close(replay,final,atol=0,rtol=0)
        torch.testing.assert_close(replay_value.log_value,value.log_value,atol=0,rtol=0)
        if args.engineering_smoke and not sum(stats['common']['accepted_per_chain']):
            raise RuntimeError('Engineering smoke has no accepted common transition')
        positions=coordinates(final)
        final_energy=zero-kT*value.log_value-restraint/2*positions.square().sum((1,2))
        if float(positions.mean(1).abs().max())>1e-10:raise RuntimeError('COM constraint failed')
        torch.save(dict(positions=positions,energy_eV=final_energy,condition=condition,
            sample_ids=source['sample_ids'][:count],inversion_signs=source['inversion_signs'][:count],
            density_scope='unknown finite-time MCMC endpoint density'),args.out/'final_samples.pt')
        torch.save(dict(initial_positions=x0,initial_latent=z0,tail_indices=tail_indices,
            trace=trace,snapshots=snapshots),args.out/'oracle_trace.pt')
        report.update(complete=True,summary=stats,oracle_evaluations=oracle.evaluated,
            requested_oracle_evaluations=oracle.requested_evaluations,proposal_replay_passed=True,
            seconds=time.perf_counter()-start,mean_final_energy_eV=float(final_energy.mean()),
            artifacts={p.name:sha(p) for p in [args.out/'final_samples.pt',args.out/'oracle_trace.pt']})
        write(output,report)
    except Exception as exc:
        torch.save(dict(initial_positions=x0,initial_latent=z0,tail_indices=tail_indices,
            trace=trace,snapshots=snapshots),args.out/'failed_trace.pt')
        report.update(complete=False,failure=f'{type(exc).__name__}: {exc}',
            oracle_evaluations=oracle.evaluated if oracle is not None else None,
            requested_oracle_evaluations=oracle.requested_evaluations if oracle is not None else None,
            seconds=time.perf_counter()-start)
        write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':
    main()
