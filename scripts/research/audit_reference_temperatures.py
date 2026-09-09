#!/usr/bin/env python3
"""Reuse frozen independent quadrature to diagnose target-temperature coverage.

All targets share existing quadrature points; no new oracle calls are made.
Lower-temperature results are coverage diagnostics, not automatically qualified
reference distributions. Four scrambles cannot reveal modes all scrambles miss.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.special import logsumexp

from cfm_mol.triatomic_reference import centered_gaussian_mixture_log_density,invariant_observables


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();source=args.reference/'reference.json';reference=json.loads(source.read_text())
    if not reference['complete']:raise ValueError('Require complete original reference')
    if args.out.exists():raise FileExistsError(args.out)
    table=Chem.GetPeriodicTable();numbers=[table.GetAtomicNumber(s) for s in reference['symbols']]
    radii=np.array([table.GetRcovalent(int(z)) for z in numbers]);i,j=np.triu_indices(3,k=1)
    kTs=[1.,.08617333262145,.025851999786435];repetitions={value:[] for value in kTs};sources={}
    restraint=reference['configuration']['restraint'];scales=reference['proposal_scales_A']
    for index,old in enumerate(reference['rows']):
        path=args.reference/f'scramble_{index}.npz';arrays=np.load(path)
        x=arrays['positions'];energy=arrays['energy_eV']
        if x.ndim!=4 or x.shape[0]!=len(scales) or x.shape[2:]!=(3,3) or energy.shape!=x.shape[:2]:raise ValueError('Stored quadrature shape changed')
        if not np.isfinite(x).all() or not np.isfinite(energy).all():raise ValueError('Nonfinite stored quadrature')
        sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        for kT in kTs:
            resolutions=[]
            for n in [x.shape[1]//2,x.shape[1]]:
                positions=x[:,:n].reshape(-1,3,3);e=energy[:,:n].reshape(-1)
                logw=-(e+restraint/2*(positions**2).sum((1,2)))/kT-centered_gaussian_mixture_log_density(positions,scales)
                logz=float(logsumexp(logw)-math.log(len(logw)))
                weights=np.exp(logw-logw.max());weights/=weights.sum()
                distances=np.linalg.norm(positions[:,i]-positions[:,j],axis=-1)
                ratios=distances/(radii[i]+radii[j]);components=3-np.minimum((ratios<=1.25).sum(-1),2)
                observables={**invariant_observables(positions),
                    'connected_contact_graph_probability':(components==1).astype(float),
                    'multiple_contact_components_probability':(components>1).astype(float),
                    'overlap_probability':np.any(ratios<.6,axis=-1).astype(float)}
                resolutions.append({'particles':len(logw),'log_normalizer_estimate':logz,
                    'weight_ess':float(1/np.sum(weights**2)),'maximum_weight':float(weights.max()),
                    'moments':{name:float(weights@values) for name,values in observables.items()}})
            if kT==reference['configuration']['kT']:
                if abs(resolutions[-1]['log_normalizer_estimate']-old['resolutions'][-1]['log_normalizer_estimate'])>1e-8:
                    raise ValueError('Stored quadrature does not reproduce original normalizer')
            repetitions[kT].append({'scramble':index,'resolutions':resolutions})
    targets=[]
    for kT,rows in repetitions.items():
        final=[r['resolutions'][-1] for r in rows];logs=np.array([r['log_normalizer_estimate'] for r in final]);z=np.exp(logs-logs.max())
        means={}
        for name in final[0]['moments']:
            values=np.array([r['moments'][name] for r in final]);mean=float(z@values/z.sum())
            influence=z*(values-mean)/z.mean()
            means[name]={'mean':mean,'delta_method_se_across_scrambles':float(influence.std(ddof=1)/math.sqrt(len(rows)))}
        targets.append({'kT_eV':kT,'temperature_K':kT/8.617333262145e-5,'rows':rows,
            'pooled_log_normalizer':float(logsumexp(logs)-math.log(len(rows))),
            'relative_se_across_scrambles':float(z.std(ddof=1)/math.sqrt(len(z))/z.mean()),
            'minimum_scramble_ess':min(r['weight_ess'] for r in final),
            'log_normalizer_range':float(np.ptp(logs)),
            'max_half_to_full_log_normalizer_change':max(abs(r['resolutions'][1]['log_normalizer_estimate']-r['resolutions'][0]['log_normalizer_estimate']) for r in rows),
            'pooled_moments':means,'convergence_certified':False})
    report={'complete':True,'scope':__doc__,'source_reference':str(source.resolve()),
        'source_reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'quadrature_sources':sources,
        'numbers':numbers,'charge':reference['charge'],'spin_multiplicity':reference['spin_multiplicity'],
        'restraint_eV_A2':restraint,'targets':targets,'new_oracle_queries':0,
        'limitations':['Identical points are reused across targets; their estimates are correlated.',
            'Low-temperature concentration can invalidate the original coverage.',
            'Small between-scramble error cannot exclude a common missed basin.',
            'Contact graphs are distance heuristics, not chemical bond labels.']}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in target.items() if k!='rows'} for target in targets],indent=2),flush=True)


if __name__=='__main__':main()
