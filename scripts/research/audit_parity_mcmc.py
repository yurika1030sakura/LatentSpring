#!/usr/bin/env python3
"""Reconstruct physical-target values and every saved MCMC proposal without new queries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from cfm_mol.fixed_target_mcmc import budgeted_population
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.tempered_smc import DensityValue


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(x):
    return dict(mean=float(x.mean()),row_sem=float(x.std()/len(x)**.5),parents=len(x))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    if args.out.exists():raise FileExistsError(args.out)
    p=args.run/'results.json';report=json.loads(p.read_text())
    submission=json.loads((args.run/'submission.json').read_text())
    if not report['complete'] or report['engineering_only'] or not report['proposal_replay_passed']:
        raise ValueError('Require completed production MCMC')
    for name,digest in report['artifacts'].items():
        if sha(args.run/name)!=digest:raise ValueError('Changed MCMC artifact')
    snapshot=Path(submission['snapshot'])
    for name,digest in report['source_sha256'].items():
        if sha(snapshot/name)!=digest:raise ValueError('Changed source snapshot')
    protocol_path=snapshot/'research/evidence/parity_mcmc_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text())
    if sha(protocol_path)!=report['protocol_sha256']:raise ValueError('Protocol hash differs')
    source_path=root/protocol['source_reference']
    if sha(source_path)!=protocol['base_samples_sha256']:raise ValueError('Changed source parent artifact')
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    saved=torch.load(args.run/'oracle_trace.pt',map_location='cpu',weights_only=False)
    samples=torch.load(args.run/'final_samples.pt',map_location='cpu',weights_only=False)
    n=len(report['condition']['numbers']);basis=centered_orthonormal_basis(n)
    def coordinates(z):return torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
    torch.testing.assert_close(saved['initial_positions'],source['positions'],atol=0,rtol=0)
    torch.testing.assert_close(coordinates(saved['initial_latent']),source['positions'],atol=1e-12,rtol=0)
    if samples['sample_ids']!=source['sample_ids'] or samples['condition']!=source['condition']:
        raise ValueError('Sample identity differs')
    kT,restraint=report['kT_eV'],report['restraint_eV_A2'];zero=report['energy_zero_eV']
    evaluated=0;max_com=0.
    for entry in saved['trace']:
        if not entry['complete']:raise ValueError('Incomplete physical target call')
        z=entry['latent'];x=coordinates(z);evaluated+=2*len(z)
        max_com=max(max_com,float(x.mean(1).abs().max()))
        even=.5*(entry['raw_energy_eV']+entry['inverted_energy_eV'])
        torch.testing.assert_close(even,entry['energy_eV'],atol=0,rtol=0)
        log_value=-(even-zero+restraint/2*x.square().sum((1,2)))/kT
        score=torch.einsum('nk,bnd->bkd',basis,(entry['force_eV_A']-restraint*x)/kT).reshape_as(z)
        torch.testing.assert_close(log_value,entry['log_value'],atol=1e-9,rtol=1e-12)
        torch.testing.assert_close(score,entry['score'],atol=1e-9,rtol=1e-12)
    if evaluated!=18048 or evaluated!=report['oracle_evaluations']:
        raise ValueError('Actual raw query budget differs')
    iterator=iter(saved['trace'])
    def replay(z):
        entry=next(iterator);torch.testing.assert_close(z,entry['latent'],atol=1e-10,rtol=1e-10)
        return DensityValue(entry['log_value'],entry['score'])
    z,value,stats=budgeted_population(saved['initial_latent'],replay,kernel=report['kernel'],
        force_updates=report['force_updates_common'],tail_indices=saved['tail_indices'],
        step_size=report['step_size_A'],max_score_norm=report['max_score_norm_per_A'],
        seed=report['seed'],hmc_length=report['hmc_length'])
    if list(iterator) or stats!=report['summary']:raise ValueError('Replay statistics differ')
    final=coordinates(z);energy=zero-kT*value.log_value-restraint/2*final.square().sum((1,2))
    torch.testing.assert_close(final,samples['positions'],atol=0,rtol=0)
    torch.testing.assert_close(energy,samples['energy_eV'],atol=0,rtol=0)
    base_energy=saved['trace'][0]['energy_eV']
    delta_energy=energy-base_energy
    delta_u=delta_energy+restraint/2*(final.square().sum((1,2))-source['positions'].square().sum((1,2)))
    state=subprocess.check_output(['sacct','-j',submission['job_id'],'-X','-n','-P',
        '--format=JobIDRaw,State,ElapsedRaw,NodeList,AllocTRES'],text=True).strip()
    if state.split('|')[1]!='COMPLETED':raise ValueError('Job is not terminal and successful')
    result=dict(complete=True,run=str(args.run),results_sha256=sha(p),artifacts=report['artifacts'],
        kernel=report['kernel'],replica=report['replica'],proposal_batches_replayed=len(saved['trace']),
        parents_replayed=len(z),raw_queries_verified=evaluated,additional_oracle_evaluations=0,
        maximum_COM_error_A=max_com,slurm_accounting=state,
        common_acceptance=stats['common']['acceptance_fraction'],tail_acceptance=stats['tail']['acceptance_fraction'],
        projected_energy_change_eV=summary(delta_energy),restrained_energy_change_eV=summary(delta_u),
        seconds=report['seconds'],scientific_submission_ready=False,
        limitations=['Replay uses stored oracle values, not independent physical re-evaluation.',
                     'Energy changes are not KL changes or probability-calibration evidence.',
                     'Finite-time endpoint density and mode coverage remain unknown.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
