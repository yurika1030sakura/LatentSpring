#!/usr/bin/env python3
"""Check reference traces and quantify retained initialization memory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from rdkit import Chem
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.geometric_domain import connected_nonoverlapping
from cfm_mol.geometry_diagnostics import contact_summary
from cfm_mol.mcmc_diagnostics import diagnose_chains


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    if args.out.exists():raise FileExistsError(args.out)
    report_path=args.run/'results.json';r=json.loads(report_path.read_text())
    submission=json.loads((args.run/'submission.json').read_text())
    if not r['complete'] or r['engineering_only'] or not r['proposal_replay_passed']:
        raise ValueError('Require complete replayed reference pilot')
    path=args.run/'reference_trace.pt'
    if sha(path)!=r['artifact_sha256']:raise ValueError('Changed reference trace')
    data=torch.load(path,map_location='cpu',weights_only=False)
    snapshot=Path(submission['snapshot'])
    for name,digest in r['implementation_sha256'].items():
        if sha(snapshot/name)!=digest:raise ValueError('Changed implementation snapshot')
    name='connected_reference_protocol_v1.json' if r.get('domain')=='connected' else 'parity_reference_protocol_v1.json'
    proto_path=snapshot/'research/evidence'/name;protocol=json.loads(proto_path.read_text())
    if sha(proto_path)!=r['protocol_sha256']:raise ValueError('Changed reference protocol')
    source_path=root/'runs'/protocol['source_file']
    if sha(source_path)!=r['source_sha256']:raise ValueError('Changed source pool')
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    zero=float(source['energy_eV'].mean());numbers=r['condition']['numbers'];n=len(numbers)
    physical=json.loads((snapshot/'research/evidence/parity_training_protocol_v1.json').read_text())
    restraint=physical['restraint_eV_A2'];temperatures=data['temperatures'].repeat(r['ladders'])
    basis=centered_orthonormal_basis(n)
    radii=torch.tensor([Chem.GetPeriodicTable().GetRcovalent(z) for z in numbers],dtype=torch.float64)
    calls=0;maximum_com=0.
    for row in data['trace']:
        if not row['complete']:raise ValueError('Incomplete target trace')
        w=row['w'];x=torch.einsum('nk,bkd->bnd',basis,(w*temperatures.sqrt()[:,None]).reshape(len(w),n-1,3))
        valid=row.get('valid',torch.ones(len(w),dtype=torch.bool))
        if r.get('domain')=='connected' and not torch.equal(valid,connected_nonoverlapping(x,radii)):
            raise ValueError('Support masks differ from actual proposal coordinates')
        logp=torch.full((len(w),),-torch.inf,dtype=w.dtype);score=torch.zeros_like(w)
        if valid.any():
            energy=.5*(row['raw_energy_eV']+row['inverted_energy_eV'])
            torch.testing.assert_close(energy,row['energy_eV'],atol=0,rtol=0)
            logp[valid]=-(energy-zero+restraint/2*x[valid].square().sum((1,2)))/temperatures[valid]
            score[valid]=torch.einsum('nk,bnd->bkd',basis,row['force_eV_A']-restraint*x[valid]).reshape(int(valid.sum()),-1)/temperatures[valid].sqrt()[:,None]
        torch.testing.assert_close(logp,row['log_value'],atol=1e-9,rtol=1e-10)
        torch.testing.assert_close(score,row['score'],atol=1e-9,rtol=1e-10)
        calls+=2*int(valid.sum());maximum_com=max(maximum_com,float(x.mean(1).abs().max()))
    if calls!=r['raw_queries']:raise ValueError('Actual raw query count differs')
    selected=data['snapshots'][r['rounds']//2+1:]
    energy=torch.stack([v['energies_eV'][:,0] for v in selected]).T
    position=torch.stack([v['cold_positions'] for v in selected]).transpose(0,1)
    restrained=energy+restraint/2*position.square().sum((2,3))
    radius=position.square().sum((2,3))/n
    contacts=[[contact_summary(x.numpy(),radii.tolist()) for x in chain] for chain in position]
    summaries=[]
    for i,chain in enumerate(contacts):
        hist={}
        for row in chain:
            k=str(row['contact_components']);hist[k]=hist.get(k,0)+1
        summaries.append(dict(ladder=i,initialization=r['initialization_families'][i],component_histogram=hist,
            last_projected_energy_eV=float(energy[i,-1]),mean_projected_energy_eV=float(energy[i].mean())))
    state=subprocess.check_output(['sacct','-j',submission['job_id'],'-X','-n','-P','--format=JobIDRaw,State,ElapsedRaw'],text=True).strip()
    if state.split('|')[1]!='COMPLETED':raise ValueError('Reference job not terminal')
    result=dict(complete=True,scope=__doc__,domain=r.get('domain','unrestricted'),run=str(args.run),
        results_sha256=sha(report_path),trace_sha256=sha(path),raw_queries_verified=calls,additional_oracle_queries=0,
        maximum_COM_error_A=maximum_com,tail_rounds_per_ladder=len(selected),ladders=summaries,
        projected_energy=diagnose_chains(energy.numpy()),restrained_energy=diagnose_chains(restrained.numpy()),
        radius_of_gyration_squared=diagnose_chains(radius.numpy()),slurm_accounting=state,
        reference_qualified=False,scientific_submission_ready=False,
        limitations=['Source-initialization differences and diagnostic Rhat/ESS require longer independent checks.',
            'Connected/no-overlap support alone is not chemical identity or chemical validity.',
            'Temperature exchange is not proof of physical mode mixing.'])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['domain','raw_queries_verified','projected_energy','restrained_energy','ladders']}),flush=True)


if __name__=='__main__':main()
