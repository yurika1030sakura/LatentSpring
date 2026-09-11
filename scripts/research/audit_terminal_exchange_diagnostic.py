#!/usr/bin/env python3
"""Check saved chemical identities, paired starts, volumes and diagnostic costs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    submission=json.loads((args.run_root/'submission.json').read_text())
    snapshot=Path(submission['snapshot']);proto_path=snapshot/'research/evidence/terminal_exchange_protocol_v1.json'
    protocol=json.loads(proto_path.read_text());common=None;rows=[]
    for method in ['mala','exchange']:
        for replica in [0,1]:
            directory=args.run_root/f'{method}_s{replica}';path=directory/'results.json';r=json.loads(path.read_text())
            if not r['complete'] or r['protocol_sha256']!=sha(proto_path):raise ValueError('Unqualified experiment')
            samples_path=directory/'samples.pt'
            if sha(samples_path)!=r['artifact_sha256']:raise ValueError('Changed diagnostic samples')
            data=torch.load(samples_path,map_location='cpu',weights_only=False)
            if common is None:common=data['initial_positions']
            else:torch.testing.assert_close(common,data['initial_positions'],atol=0,rtol=0)
            if len(data['snapshots'])!=protocol['cycles']+1 or len(r['history'])!=protocol['cycles']+1:
                raise ValueError('Missing trajectory prefix')
            numbers=r['condition']['numbers'];radii=covalent_radii(numbers)
            for saved,logged in zip(data['snapshots'],r['history']):
                identities=[infer_chemical_graph(x,numbers,r['condition']['charge'])['connectivity_smiles']
                            for x in saved['positions']]
                if identities!=saved['smiles'] or identities!=logged['smiles']:
                    raise ValueError('Saved connectivity identities disagree')
                torch.testing.assert_close(saved['energy_eV'],torch.tensor(logged['energy_eV'],dtype=torch.float64),atol=1e-10,rtol=0)
                if float(saved['positions'].mean(1).abs().max())>1e-10:raise ValueError('COM constraint failed')
            expected=2*len(common)*(1+3*protocol['cycles'])
            if r['new_raw_queries']!=expected:raise ValueError('The realized paired query budgets differ')
            for attempt in r['exchange_attempts']:
                if 'action' not in attempt:continue
                i,j,k,l=attempt['action']
                expected_volume=3*math.log(float((radii[i]+radii[l])/(radii[j]+radii[l])*(radii[j]+radii[k])/(radii[i]+radii[k])))
                if abs(attempt['log_volume']-expected_volume)>1e-12 or attempt['reverse_action']!=[i,j,l,k]:
                    raise ValueError('Transformation volume or inverse action differs')
            target=r['history'][0]['smiles'][2]
            first=[]
            for chain in [0,1]:
                hits=[v['cycle'] for v in r['history'] if v['smiles'][chain]==target]
                first.append(min(hits) if hits else None)
            rows.append(dict(method=method,replica=replica,new_raw_queries=r['new_raw_queries'],
                inherited_reference_raw_queries=r['inherited_reference_raw_queries'],
                initial_smiles=r['history'][0]['smiles'],final_smiles=r['history'][-1]['smiles'],
                final_energy_eV=r['history'][-1]['energy_eV'],generated_first_target_cycle=first,
                accepted_exchanges=sum(v['accepted'] for v in r['exchange_attempts']),
                results_sha256=sha(path),samples_sha256=sha(samples_path)))
    state=subprocess.check_output(['sacct','-j',submission['job_id'],'-X','-n','-P','--format=JobIDRaw,State,ElapsedRaw'],text=True).strip()
    if '|COMPLETED|' not in state:raise ValueError('Diagnostic allocation not successfully terminal')
    result=dict(complete=True,scope=__doc__,slurm=state,rows=rows,new_raw_queries=sum(r['new_raw_queries'] for r in rows),
        saved_structures_checked=4*4*(protocol['cycles']+1),additional_oracle_queries=0,
        scientific_submission_ready=False,
        limitations=['Uniform exchange is not a learned policy or established AI novelty.',
            'Two generated initial configurations and two transition seeds, with QC-origin control chains.',
            'Warm-start production costs are inherited and additional.',
            'This v1 saves cycle endpoints and exchange decisions, not a full per-query force trace; full stochastic replay is not certified.'])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
