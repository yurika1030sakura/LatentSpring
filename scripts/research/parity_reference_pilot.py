#!/usr/bin/env python3
"""Multiple-initialization replica-exchange diagnostic, not an equilibrium certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.fixed_target_mcmc import hmc_population
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.parity_refinement import evaluate_even_potential
from cfm_mol.temperature_exchange import exchange_scaled_states
from cfm_mol.tempered_smc import DensityValue


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle-checkpoint',type=Path,required=True)
    p.add_argument('--engineering-smoke',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[2];protocol_path=root/'research/evidence/parity_reference_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text());physical_path=root/'research/evidence/parity_training_protocol_v1.json'
    physical=json.loads(physical_path.read_text())
    if not protocol['frozen'] or sha(physical_path)!=protocol['physical_protocol_sha256']:
        raise ValueError('Changed physical reference protocol')
    source_path=args.runs_root/protocol['source_file'];ref_path=args.runs_root/protocol['reference_file']
    if sha(source_path)!=protocol['source_sha256'] or sha(ref_path)!=protocol['reference_sha256']:
        raise ValueError('Changed source/reference input')
    if sha(args.oracle_checkpoint)!=physical['raw_oracle_sha256']:raise ValueError('Wrong oracle')
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    refs=json.loads(ref_path.read_text());ref=refs['rows'][protocol['condition_index']]
    condition=source['condition'];numbers=condition['numbers'];n=len(numbers)
    if (not refs['complete'] or refs['source_manifest_sha256']!=condition['manifest_sha256']
            or ref['condition']['atomic_numbers']!=numbers
            or any(ref['condition'][k]!=condition[k] for k in ['raw_index','charge','spin_multiplicity'])):
        raise ValueError('Reference electronic state or identity differs')
    temperatures=torch.tensor(protocol['temperatures_eV'],dtype=torch.float64)
    if float(temperatures[0])!=physical['kT_eV']:raise ValueError('Cold target differs')
    levels=len(temperatures);ladders=4;chains=ladders*levels
    generator=torch.Generator().manual_seed(protocol['initialization_seed'])
    selected=torch.randperm(len(source['positions']),generator=generator)[:2*levels]
    fm=source['positions'][selected].double()
    qc=torch.tensor(ref['positions'],dtype=torch.float64)[None].expand(2*levels,-1,-1).clone()
    jitter=torch.randn(qc.shape,dtype=torch.float64,generator=generator)*protocol['reference_jitter_A']
    qc=qc+jitter-jitter.mean(1,keepdim=True)
    signs=(2*torch.randint(2,(len(qc),),generator=generator)-1)[:,None,None]
    initial=torch.cat([fm,qc*signs])
    basis=centered_orthonormal_basis(n);temp=temperatures.repeat(ladders)
    latent=torch.einsum('nk,bnd->bkd',basis,initial).reshape(chains,-1)
    initial_w=latent/temp.sqrt()[:,None]
    rounds=protocol['engineering_rounds'] if args.engineering_smoke else protocol['pilot_rounds']
    length=protocol['leapfrog_steps'];expected=2*chains*(1+rounds*length)
    restraint=physical['restraint_eV_A2'];zero=float(source['energy_eV'].mean())
    report=dict(complete=False,scope=__doc__,role='independent_reference_diagnostic',
        reference_qualified=False,scientific_submission_ready=False,condition=condition,
        protocol_sha256=sha(protocol_path),source_sha256=sha(source_path),reference_sha256=sha(ref_path),
        initialization_families=['fm','fm','raw_qc_jitter','raw_qc_jitter'],selected_fm_indices=selected.tolist(),
        uses_reference_geometry_for_initialization=True,engineering_only=args.engineering_smoke,
        ladders=ladders,temperatures_eV=temperatures.tolist(),rounds=rounds,leapfrog_steps=length,
        expected_raw_queries=expected,history=[],
        implementation_sha256={str(p.relative_to(root)):sha(p) for p in [Path(__file__).resolve(),
            root/'cfm_mol/temperature_exchange.py',root/'cfm_mol/fixed_target_mcmc.py',
            root/'cfm_mol/parity_refinement.py',root/'scripts/research/oracle_worker.py']},
        limitations=['A raw QC structure is an initialization, not an equilibrium distribution.',
            'Replica exchanges and temperature round trips do not prove configuration-mode mixing.',
            'Cold draws are autocorrelated within four ladders; do not report them as independent samples.'])
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    write(output,report);start=time.perf_counter();trace=[];oracle=None
    def positions(w):
        return torch.einsum('nk,bkd->bnd',basis,(w*temp.sqrt()[:,None]).reshape(chains,n-1,3))
    def run(target,record=False):
        rng=torch.Generator().manual_seed(protocol['sampling_seed'])
        w=initial_w.clone();value=target(w);labels=torch.arange(chains).reshape(ladders,levels)
        snapshots=[];swaps=[];accepts=[]
        def snapshot(iteration):
            x=positions(w).reshape(ladders,levels,n,3)
            energy=zero-temp.reshape(ladders,levels)*value.log_value.reshape(ladders,levels)-restraint/2*x.square().sum((-1,-2))
            snapshots.append(dict(round=iteration,cold_positions=x[:,0].clone(),energies_eV=energy.clone(),walkers=labels.clone()))
            if record and (iteration==0 or iteration%16==0 or iteration==rounds):
                row=dict(round=iteration,cold_energy_by_ladder_eV=energy[:,0].tolist(),
                    raw_queries=oracle.evaluated,seconds=time.perf_counter()-start)
                report['history'].append(row);write(output,report);print(json.dumps(row),flush=True)
        snapshot(0)
        for iteration in range(rounds):
            w,value,stats=hmc_population(w,target,leapfrog_counts=[length],step_size=protocol['scaled_step_size'],
                max_score_norm=protocol['physical_force_cap_eV_A']/float(temperatures[0].sqrt()),
                generator=rng,initial_value=value)
            accepts.append(stats['accepted_per_chain'])
            shaped,value,labels,exchange=exchange_scaled_states(w.reshape(ladders,levels,-1),value,
                temperatures,labels,parity=iteration%2,generator=rng)
            w=shaped.reshape(chains,-1);swaps.append(exchange);snapshot(iteration+1)
        return w,value,snapshots,swaps,accepts
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
        report['oracle_runtime']=oracle.handshake
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype')!='torch.float32':
            raise ValueError('Require qualified oracle precision')
        def target(w):
            entry=dict(w=w.clone(),complete=False);trace.append(entry)
            x=positions(w);energy,force,components=evaluate_even_potential(oracle,x)
            value=DensityValue(-(energy-zero+restraint/2*x.square().sum((1,2)))/temp,
                torch.einsum('nk,bnd->bkd',basis,force-restraint*x).reshape_as(w)/temp.sqrt()[:,None])
            entry.update(complete=True,log_value=value.log_value.clone(),score=value.score.clone(),
                energy_eV=energy,force_eV_A=force,**components)
            return value
        w,value,snapshots,swaps,accepts=run(target,record=True)
        if oracle.evaluated!=expected:raise RuntimeError('Reference query count differs')
        iterator=iter(trace)
        def replay_target(w):
            row=next(iterator);torch.testing.assert_close(w,row['w'],atol=1e-10,rtol=1e-10)
            return DensityValue(row['log_value'],row['score'])
        replay,replay_value,replay_snapshots,replay_swaps,replay_accepts=run(replay_target)
        torch.testing.assert_close(replay,w,atol=0,rtol=0)
        if list(iterator) or replay_accepts!=accepts:raise RuntimeError('Reference replay differs')
        for a,b in zip(swaps,replay_swaps):
            torch.testing.assert_close(a['accepted'],b['accepted'],atol=0,rtol=0)
        torch.save(dict(initial_positions=initial,initial_scaled_latent=initial_w,temperatures=temperatures,
            trace=trace,snapshots=snapshots,swaps=swaps,hmc_acceptances=accepts),args.out/'reference_trace.pt')
        report.update(complete=True,raw_queries=oracle.evaluated,proposal_replay_passed=True,
            seconds=time.perf_counter()-start,artifact_sha256=sha(args.out/'reference_trace.pt'),
            cold_hmc_acceptance_by_ladder=torch.tensor(accepts).reshape(rounds,ladders,levels).double().mean(0)[:,0].tolist())
        write(output,report)
    except Exception as exc:
        torch.save(dict(initial_positions=initial,initial_scaled_latent=initial_w,trace=trace),args.out/'failed_trace.pt')
        report.update(complete=False,failure=f'{type(exc).__name__}: {exc}',
            raw_queries=oracle.evaluated if oracle else None,requested_raw_queries=oracle.requested_evaluations if oracle else None)
        write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
