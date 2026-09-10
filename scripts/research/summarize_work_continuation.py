#!/usr/bin/env python3
"""Audit completed equal-query continuation arms against their saved 500-step parents."""
import argparse
import json
from pathlib import Path

import torch

from molecular_tempered_pilot import sha,write_json


def read_complete(path):
    data=json.loads(path.read_text())
    if not data.get('complete'):raise ValueError(f'Incomplete evidence: {path}')
    return data


def endpoint_summary(run):
    path=run/'results.json';result=read_complete(path);sample=run/'final_samples.pt'
    data=torch.load(str(sample),map_location='cpu',weights_only=False)
    x=data['positions'].double();energy=data['energy_eV'].double();work=data['work'].double()
    if x.shape[0]!=len(energy) or work.shape!=energy.shape or not torch.isfinite(work).all():raise ValueError('Invalid terminal sample panel')
    config=result['configuration'];restrained=energy+config['restraint']/2*x.square().sum((1,2))
    reduced=(restrained-result['energy_zero_eV'])/config['kT'];transport=work-reduced
    weights=torch.softmax(-work,0);ess=float(weights.square().sum().reciprocal())
    return result,{'particles':len(work),'ess':ess,'ess_fraction':ess/len(work),'maximum_weight':float(weights.max()),
        'mean_work':float(work.mean()),'work_std':float(work.std()),'mean_energy_eV':float(energy.mean()),
        'restrained_energy_std_eV':float(restrained.std()),'transport_factor_std':float(transport.std()),
        'energy_transport_correlation':float(torch.corrcoef(torch.stack([reduced,transport]))[0,1]),
        'unweighted_geometry':result['evaluations']['final']['geometry'],
        'weighted_geometry':result['evaluations']['final']['weighted_geometry'],
        'source_results_sha256':sha(path),'samples_sha256':sha(sample)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--assessment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    assessment=read_complete(args.assessment/'assessment.json');rows=[];reference_target=None
    xtb={row['arm']:row for row in assessment['xtb_summaries']}
    for arm,mode,power in [('fixed_joint','joint',0.),('annealed_joint','joint',.5),('annealed_energy','forward_energy_only',.5)]:
        parent=args.runs_root/f'work300_{arm}_5846_v1';run=args.runs_root/f'work300_{arm}_5846_1500_v1'
        previous,before=endpoint_summary(parent);current,after=endpoint_summary(run);config=current['configuration']
        if config['steps']!=1500 or current['training_start_step']!=500 or len(current['training'])!=1000:
            raise ValueError('Wrong continuation length')
        if [row['step'] for row in current['training']]!=list(range(501,1501)):raise ValueError('Training RNG/update step sequence changed')
        if config['mode']!=mode or config['noise_annealing_power']!=power:raise ValueError('Arm protocol mismatch')
        if current['oracle_evaluations']!=16512 or previous['oracle_evaluations']!=8512 or current['cumulative_oracle_evaluations']!=25024:
            raise ValueError('Oracle accounting mismatch')
        if current['resume']['source_results_sha256']!=sha(parent/'results.json'):raise ValueError('Resume parent changed')
        if current['resume']['source_checkpoint_sha256']!=sha(parent/'last.ckpt'):raise ValueError('Resume checkpoint changed')
        if not current['resume']['optimizer_restored'] or current['temperature_input_initialization']['applied_in_this_run']:
            raise ValueError('Invalid resume state/temperature handling')
        target=[current['condition'],config['kT'],config['restraint'],current['oracle_sha256'],current['energy_zero_eV']]
        if reference_target is None:reference_target=target
        if target!=reference_target:raise ValueError('Arms have different physical targets')
        source=next(row for row in assessment['sources'] if row['arm']==arm)
        if source['samples_sha256']!=sha(run/'final_samples.pt'):raise ValueError('xTB source samples changed')
        if xtb[arm]['attempted']!=32:raise ValueError('xTB denominator changed')
        rows.append({'arm':arm,'parent_500_steps':before,'continued_1500_steps':after,
            'new_oracle_queries':current['oracle_evaluations'],'cumulative_queries':current['cumulative_oracle_evaluations'],
            'program_seconds_this_run':current['seconds'],'xtb':xtb[arm]})
    hmc=args.runs_root/'hmc_300K_5846_25024_v1';hmc_run=read_complete(hmc/'sampling/results.json');hmc_xtb=read_complete(hmc/'xtb/assessment.json')
    if hmc_run['oracle_evaluations']!=25024:raise ValueError('HMC query budget mismatch')
    for key in ['numbers','charge','spin_multiplicity']:
        if hmc_run['condition'][key]!=reference_target[0][key]:raise ValueError('HMC condition mismatch')
    if [hmc_run['configuration']['kT'],hmc_run['configuration']['restraint'],hmc_run['oracle_sha256']]!=reference_target[1:4]:
        raise ValueError('HMC target mismatch')
    report={'complete':True,'scope':__doc__,'condition':reference_target[0],'rows':rows,
        'hmc':{'summary':hmc_run['summary'],'program_seconds':hmc_run['seconds'],'xtb':hmc_xtb['xtb_summaries'],
            'sampling_results_sha256':sha(hmc/'sampling/results.json'),'assessment_sha256':sha(hmc/'xtb/assessment.json')},
        'assessment_sha256':sha(args.assessment/'assessment.json'),'script_sha256':sha(Path(__file__)),
        'limitations':['One development condition and one training seed per arm; no broad benefit or novelty certificate.',
            'Path-weight ESS is not a certificate of global mode coverage.',
            'HMC contributes eight independent chains with correlated retained samples and unknown initialization bias.',
            'Successful-only strain medians must be read alongside every failure denominator.',
            'Equal potential-query counts do not equate GPU/CPU cost or independent output counts.',
            'Transport factors include forward/backward path terms and are not endpoint log densities.'],
        'scientific_submission_ready':False}
    write_json(args.out,report)
    for row in rows:print(row['arm'],row['parent_500_steps']['ess'],row['continued_1500_steps']['ess'],row['xtb']['converged'])


if __name__=='__main__':main()
