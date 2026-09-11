#!/usr/bin/env python3
"""Compare full-precision GPU oracle batches with the existing CPU target."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cfm_mol.numpy_energy_oracle import NumpyEnergyOracle


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((args.panel/'manifest.json').read_text())
    if sha(args.checkpoint)!=manifest['protocol']['raw_oracle_sha256']:raise ValueError('Wrong oracle checkpoint')
    worker=Path(__file__).resolve().with_name('oracle_worker.py')
    report={'complete':False,'scope':__doc__,'rows':[],'expected_queries':544,'acknowledged_queries':0,
        'oracle_sha256':sha(args.checkpoint),'worker_sha256':sha(worker),
        'energy_atol_eV':1e-4,'force_atol_eV_A':1e-4,'force_rtol':1e-5,
        'scientific_submission_ready':False,'limitations':['Two prescribed electronic conditions only.',
            'Timing includes synchronous RPC evaluation; excludes startup and four warm-up structures per setting.',
            'GPU/CPU numerical agreement does not establish physical accuracy or calibrated sampling.']}
    write(output,report)
    for item in manifest['conditions']:
        panel=args.panel/item['panel']
        if sha(panel)!=item['panel_sha256']:raise ValueError('Source panel changed')
        data=np.load(panel);base=data['positions'];condition=item['condition']
        rng=np.random.default_rng(9591+item['index'])
        x=base[np.arange(32)%len(base)]+rng.normal(scale=.002,size=(32,*base.shape[1:]))
        x-=x.mean(1,keepdims=True);x=np.concatenate([x,-x])
        np.save(args.out/f'positions_{item["index"]:02d}.npy',x)
        reference=None
        for device,batch in [('cpu',1),('cuda',1),('cuda',8),('cuda',32)]:
            row={'condition_index':item['index'],'condition':condition,'device':device,'batch_size':batch,'complete':False}
            report['rows'].append(row)
            oracle=None
            try:
                oracle=NumpyEnergyOracle(args.oracle_python,worker,args.checkpoint,numbers=condition['numbers'],
                    charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device=device,
                    batch_size=batch,stderr_path=args.out/f'oracle_{item["index"]}_{device}_{batch}.log')
                row['runtime']=oracle.handshake
                if oracle.handshake['tf32'] or oracle.handshake['base_precision_dtype']!='torch.float32':
                    raise ValueError('Qualification requires the declared full-precision inference settings')
                oracle.evaluate_chunked(x[:4])
                start=time.perf_counter();energy,force=oracle.evaluate_chunked(x,max_request=32)
                seconds=time.perf_counter()-start
                np.savez(args.out/f'values_{item["index"]}_{device}_{batch}.npz',energy=energy,force=force)
                if reference is None:reference=(energy,force)
                error_energy=float(np.max(np.abs(energy-reference[0])))
                error_force=float(np.max(np.abs(force-reference[1])))
                np.testing.assert_allclose(energy,reference[0],atol=report['energy_atol_eV'],rtol=0)
                np.testing.assert_allclose(force,reference[1],atol=report['force_atol_eV_A'],rtol=report['force_rtol'])
                row.update(complete=True,seconds=seconds,structures_per_second=64/seconds,
                    maximum_energy_error_eV=error_energy,maximum_force_error_eV_A=error_force)
            except Exception as exc:
                row['failure']=f'{type(exc).__name__}: {exc}'
                raise
            finally:
                row['acknowledged_queries']=oracle.evaluated if oracle is not None else 0
                row['requested_queries']=oracle.requested_evaluations if oracle is not None else 0
                if oracle is not None:oracle.close()
                report['acknowledged_queries']=sum(r.get('acknowledged_queries',0) for r in report['rows'])
                write(output,report)
            print(json.dumps({k:row[k] for k in ['condition_index','device','batch_size','seconds','maximum_energy_error_eV','maximum_force_error_eV_A']}),flush=True)
    if report['acknowledged_queries']!=report['expected_queries']:raise ValueError('Raw-query accounting differs')
    report['complete']=True;write(output,report)


if __name__=='__main__':main()
