#!/usr/bin/env python3
"""Compare frozen work samples to an independent three-atom reference.

Finite-sample normalizers and self-normalized moments remain statistical
estimates. Empirical errors cannot detect a tail or basin missed by all draws.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from cfm_mol.triatomic_reference import invariant_observables
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--unweighted',action='store_true',help='Compare endpoint moments without a density or normalizer claim')
    p.add_argument('--stages',nargs='+',choices=['initial','final'],default=['initial','final'])
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result=args.run/'results.json';reference=args.reference/'reference.json'
    training=json.loads(result.read_text());ref=json.loads(reference.read_text())
    if not training['complete'] or not ref['complete']:raise ValueError('Both runs must be complete')
    if any(training['condition'][key]!=ref['condition'][key] for key in ['numbers','charge','spin_multiplicity']):raise ValueError('Reference physical condition mismatch')
    if training['oracle_sha256']!=ref['oracle_sha256'] or training['configuration']['restraint']!=ref['restraint_eV_A2']:
        raise ValueError('Potential/restraint mismatch')
    kT=training['configuration']['kT'];matching=[r for r in ref['aggregate'] if abs(r['kT_eV']-kT)<1e-12]
    if not matching:raise ValueError('No matching reference temperature')
    target=max(matching,key=lambda r:r['particles_per_scramble']);rows=[];sources={}
    for stage in args.stages:
        path=args.run/f'{stage}_samples.pt';data=torch.load(str(path),map_location='cpu',weights_only=False)
        x=data['positions'].double().numpy();n=len(x)
        if n<2 or not np.isfinite(x).all():raise ValueError('Invalid endpoint sample panel')
        if args.unweighted:
            weight=np.full(n,1./n);ess=None;logz=None
        else:
            work=data['work'].double().numpy()
            if work.shape!=(n,) or not np.isfinite(work).all():raise ValueError('Invalid work sample panel')
            log_mean=float(logsumexp(-work)-math.log(n));weight=np.exp(-work+work.min());weight/=weight.sum()
            ess=float(1/(weight@weight));logz=log_mean-training['energy_zero_eV']/kT
        moments={}
        clusters=data.get('sample_cluster_ids')
        if clusters is not None:
            clusters=np.asarray(clusters)
            if not args.unweighted or clusters.shape!=(n,) or len(np.unique(clusters))<2:
                raise ValueError('Clustered comparisons require unweighted samples from at least two chains')
        for name,value in invariant_observables(x).items():
            estimate=float(weight@value);reference_moment=target['moments'][name]
            se=math.sqrt(n/(n-1)*float(np.sum((weight*(value-estimate))**2))) if args.unweighted or ess>=10 else None
            if clusters is not None:
                groups=np.unique(clusters);sums=np.array([np.sum(value[clusters==group]-estimate)/n for group in groups])
                se=math.sqrt(len(groups)/(len(groups)-1)*float(np.sum(sums*sums)))
            moments[name]={'unweighted_mean':float(np.mean(value)),'weighted_mean':None if args.unweighted else estimate,
                'reference_mean':reference_moment['mean'],'difference':estimate-reference_moment['mean'],
                'empirical_estimate_se':se,'reference_delta_se':reference_moment['delta_method_se']}
        rows.append({'stage':stage,'particles':n,'ess':ess,'maximum_weight':None if args.unweighted else float(weight.max()),
            'error_scope':('clustered by independent chain; excludes initialization bias' if clusters is not None else 'unweighted endpoint sampling variation; excludes initialization bias') if args.unweighted else 'self-normalized IS delta error, suppressed if ESS < 10',
            'log_normalizer_estimate':logz,'reference_log_normalizer':target['log_of_mean_normalizer_estimate'],
            'log_normalizer_difference':None if args.unweighted else logz-target['log_of_mean_normalizer_estimate'],
            'empirical_normalizer_relative_se':None if args.unweighted else math.sqrt(max(0,(n/ess-1)/(n-1))),
            'reference_normalizer_relative_se':target['relative_se_across_scrambles'],'moments':moments})
        sources[str(path.resolve())]=sha(path)
    report={'complete':True,'scope':__doc__,'unweighted_mode':args.unweighted,'condition':training['condition'],'kT_eV':kT,'rows':rows,
        'source_results_sha256':sha(result),'source_reference_sha256':sha(reference),'sample_sources':sources,
        'reference_convergence_certified':ref['convergence_certified'],
        'limitations':['No low-ESS delta error is presented as a confidence interval.',
            'Reference errors use only four independent scrambles.',
            'Matching a few moments or a normalizer does not prove full distributional accuracy.',
            'One training seed is not replication.','Unweighted finite-time MCMC moments are not assumed to be at equilibrium.']}
    args.out.parent.mkdir(parents=True,exist_ok=True);write_json(args.out,report)
    print(json.dumps([{k:v for k,v in row.items() if k!='moments'} for row in rows],indent=2),flush=True)


if __name__=='__main__':main()
