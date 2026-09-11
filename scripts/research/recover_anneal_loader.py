#!/usr/bin/env python3
"""Recover only the recorded post-training constructor-metadata failures.

Original reports and allocations stay failed. New directories contain copied,
replayed checkpoints and a linked recovery record; no optimization or oracle
query is repeated. This is an artifact qualification, not baseline superiority.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.linear_entropy_adapter import endpoint_kl_change


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--logs-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    jobs={'base_noanneal':'45931183','anneal_kT1':'45931185','anneal_kT1_sw4':'45931191'}
    index={'complete':False,'scope':__doc__,'new_oracle_evaluations':0,'rows':[],'scientific_submission_ready':False}
    for name,job in jobs.items():
        source=args.source_root/name
        original=source/'results.json'
        report=json.loads(original.read_text())
        stderr=args.logs_root/f'bgfm-anneal-{job}.err'
        if "unexpected keyword argument 'minimum_active'" not in stderr.read_text():
            raise ValueError('Failure is outside the narrowly qualified loader recovery')
        if (report['complete'] or report['oracle_evaluations']!=18048 or report['steps']!=1000
                or report['history'][-1]['step']!=1000 or report['kind']!='species_convex'):
            raise ValueError('Require the original completed training and failed loader pattern')
        for file,digest in report['artifacts'].items():
            if sha(source/file)!=digest:raise ValueError('Original artifact changed')
        out=args.out/name;out.mkdir()
        for file in report['artifacts']:
            shutil.copy2(source/file,out/file)
        report['recovery']={'original_results':str(original.resolve()),'original_results_sha256':sha(original),
            'failed_job_id':job,'original_stderr_sha256':sha(stderr),'original_process_success':False,
            'reason':'Constructor options were mixed with derived block metadata.', 'new_oracle_evaluations':0}
        # The public loader requires completed-training provenance. A failed
        # recovery immediately restores false in this new report; originals stay intact.
        report['complete']=True
        path=out/'results.json';path.write_text(json.dumps(report,indent=2)+'\n')
        try:
            model,_,_=load_entropy_adapter(out)
            base=torch.load(out/'base_samples.pt',map_location='cpu',weights_only=False)
            sample=torch.load(out/'adapted_samples.pt',map_location='cpu',weights_only=False)
            maximum=0.
            with torch.no_grad():
                for start in range(0,len(base['positions']),64):
                    y,vol=model(base['positions'][start:start+64])
                    torch.testing.assert_close(y,sample['positions'][start:start+64],atol=1e-9,rtol=1e-9)
                    torch.testing.assert_close(vol,sample['log_volume'][start:start+64],atol=1e-9,rtol=1e-9)
                    maximum=max(maximum,float((y-sample['positions'][start:start+64]).abs().max()))
            change=endpoint_kl_change(base['energy_eV'],sample['energy_eV'],base['positions'],sample['positions'],
                kT=report['kT_eV'],restraint=report['restraint_eV_A2'],log_volume=sample['log_volume'])
            torch.testing.assert_close(change,sample['paired_endpoint_kl_change'],atol=1e-9,rtol=1e-9)
            report['checkpoint_loader_replay_passed']=True
            report['recovery'].update(all_evaluation_parents_replayed=len(change),maximum_position_error_A=maximum)
        except Exception as exc:
            report['complete']=False
            report['recovery']['failure']=f'{type(exc).__name__}: {exc}'
            path.write_text(json.dumps(report,indent=2)+'\n')
            raise
        path.write_text(json.dumps(report,indent=2)+'\n')
        index['rows'].append({'name':name,'report_sha256':sha(path),'run':str(out.resolve()),'recovery':report['recovery']})
        (args.out/'recovery.json').write_text(json.dumps(index,indent=2)+'\n')
        print(json.dumps({'name':name,'replayed':len(change),'max_error':maximum}),flush=True)
    index['complete']=True
    (args.out/'recovery.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':main()
