#!/usr/bin/env python3
"""Cross-check shape-QMC on an explicit finite box using direct product cubature."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.triatomic_box_quadrature import triatomic_box_nodes
from cfm_mol.triatomic_shape_proposal import TriatomicShapeMixture,defensive_shape_log_density
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--orders',type=int,nargs='+',default=[20,32])
    args=p.parse_args();reference=args.reference/'reference.json';ref=json.loads(reference.read_text())
    if not ref['complete']:raise ValueError('Require complete QMC reference')
    condition=ref['condition'];recipe=ref['configuration'];kT=ref['kT_eV'][0];restraint=ref['restraint_eV_A2']
    if len(condition['numbers'])!=3:raise ValueError('Only three-atom conditions supported')
    if sha(Path(recipe['oracle']))!=ref['oracle_sha256']:raise ValueError('Oracle changed')
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    lower,upper=2.,3.2;rows=[];qmc_rows=[]
    proposal=TriatomicShapeMixture(ref['shape_centers'],ref['shape_stds'])
    for name,expected in ref['output_array_sha256'].items():
        path=args.reference/name
        if sha(path)!=expected:raise ValueError('Reference arrays changed')
        data=np.load(path);x=data['positions'].reshape(-1,3,3);energy=data['energy_eV'].ravel()
        ra=np.linalg.norm(x[:,1]-x[:,0],axis=-1);rb=np.linalg.norm(x[:,2]-x[:,0],axis=-1)
        inside=(ra>=lower)&(ra<=upper)&(rb>=lower)&(rb<=upper)
        logw=-(energy+restraint/2*np.sum(x*x,axis=(1,2)))/kT-defensive_shape_log_density(x,proposal,ref['defensive_std_A'])
        qmc_rows.append(float(logsumexp(logw[inside])-math.log(len(x))))
    reference_logz=float(logsumexp(qmc_rows)-math.log(len(qmc_rows)))
    scaled=np.exp(np.array(qmc_rows)-max(qmc_rows));reference_se=float(scaled.std(ddof=1)/math.sqrt(len(scaled))/scaled.mean())
    report={'complete':False,'scope':__doc__,'condition':condition,'kT_eV':kT,'restraint_eV_A2':restraint,
        'radial_box_A':[lower,upper],'cosine_box':[-1.,1.],'qmc_box_log_normalizer':reference_logz,
        'qmc_box_relative_se':reference_se,'qmc_source_sha256':sha(reference),'oracle_sha256':ref['oracle_sha256'],
        'rows':rows,'limitations':['Finite radial box; this is not a total-normalizer certificate.',
            'Cubature order differences are numerical diagnostics, not rigorous truncation errors.']}
    args.out.mkdir(parents=True,exist_ok=True);write_json(output,report);root=Path(__file__).resolve().parents[2];start=time.perf_counter()
    with EnergyOracle(Path(recipe['oracle_python']),root/'scripts/research/oracle_worker.py',Path(recipe['oracle']),
        numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle:
        for order in args.orders:
            x,logvolume=triatomic_box_nodes(order,lower,upper);pieces=[]
            for begin in range(0,len(x),64):
                energy,_=oracle.evaluate(x[begin:begin+64]);pieces.append(energy.numpy())
            energy=np.concatenate(pieces);logintegrand=logvolume-(energy+restraint/2*np.sum(x*x,axis=(1,2)))/kT
            logz=float(logsumexp(logintegrand))
            np.savez_compressed(args.out/f'order_{order}.npz',positions=x,energy_eV=energy,log_volume_weights=logvolume)
            row={'order':order,'queries':len(x),'log_normalizer_box':logz,'difference_to_qmc_box':logz-reference_logz,
                'output_sha256':sha(args.out/f'order_{order}.npz'),'seconds':time.perf_counter()-start}
            rows.append(row);write_json(output,report);print(json.dumps(row),flush=True)
        report.update(complete=True,oracle_evaluations=oracle.evaluated,seconds=time.perf_counter()-start);write_json(output,report)


if __name__=='__main__':main()
