#!/usr/bin/env python3
"""Render completed same-condition calibration evidence without hiding failures."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--include-reference-500',action='store_true')
    p.add_argument('--include-native-500',action='store_true')
    args=p.parse_args();sources={};rows=[];condition=None
    def read(relative):
        nonlocal condition
        path=args.runs/relative;data=json.loads(path.read_text())
        if not data['complete']:raise ValueError('Incomplete requested source: '+str(path))
        identity={k:data['condition'][k] for k in ['numbers','charge','spin_multiplicity']}
        if condition is None:condition=identity
        elif condition!=identity:raise ValueError('Mismatched physical conditions')
        sources[relative]={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return data
    def add(label,assessment,arm,training=None,stage=None):
        geometry=next(r for r in assessment['geometry'] if r['arm']==arm)
        xtb=next(r for r in assessment['xtb_summaries'] if r['arm']==arm)
        work=training['evaluations'][stage] if training else None
        rows.append({'label':label,'particles':geometry['particles'],
            'xtb_attempted':xtb['attempted'],'xtb_converged':xtb['converged'],
            'strain_eV_success_only':xtb['strain_eV_success_only'],
            'initial_max_force_norm_eV_A':xtb['initial_max_force_norm_eV_A'],
            'force_values_present':xtb['force_values_present'],
            'overlap_count':geometry['overlap_count'],
            'multiple_contact_components_count':geometry['multiple_contact_components_count'],
            'contact_components':geometry['contact_components'],
            'pair_distance_profile_rms_A':geometry['pair_distance_profile_rms_A'],
            'weights':geometry['weights'],
            'mean_energy_eV':work['mean_energy_eV'] if work else None,
            'mean_shifted_work':work['mean_shifted_work'] if work else None,
            'work_std':work['work_std'] if work else None,
            'training_updates':training['configuration']['steps'] if training and stage=='final' else 0,
            'training_batch':training['configuration']['batch'] if training else None,
            'training_oracle_queries':training['configuration']['steps']*training['configuration']['batch'] if training and stage=='final' else 0})
    fm=read('work_fm_control_5846_v2/assessment/assessment.json')
    for arm in ['fm16','fm64']:add(arm.upper(),fm,arm)
    old=read('mean_work_joint_5846_v1/results.json');assessment=read('mean_work_xtb_100_v1/assessment.json')
    add('Reference mean: initial',assessment,'initial',old,'initial')
    add('Reference mean: joint 100',assessment,'joint',old,'final')
    native=read('native_mean_preflight_5846_v1/results.json');assessment=read('native_mean_xtb_5846_v1/assessment.json')
    add('Native mean: initial',assessment,'initial',native,'initial')
    add('Native mean: joint 2',assessment,'joint',native,'final')
    for enabled,prefix,xtb_run,label in [
        (args.include_reference_500,'mean_work','mean_work_xtb_500_v1','Reference mean'),
        (args.include_native_500,'native_work','native_work_xtb_500_v1','Native mean')]:
        if not enabled:continue
        assessment=read(xtb_run+'/assessment.json')
        for mode in ['joint','energy']:
            train=read(f'{prefix}_{mode}_b16_5846_v1/results.json')
            add(f'{label}: {mode} 500',assessment,mode,train,'final')
    report={'complete':True,'scope':'Single-condition, single-training-seed development; no useful Boltzmann sampling established.',
        'sources':sources,'condition':condition,'rows':rows,
        'limitations':['Different training budgets; the 500-update arms are the matched learning controls.',
            'Strain quantiles condition on convergence and are not confidence intervals.',
            'Contact graphs and distance profiles are incomplete geometry heuristics.',
            'FM importance weights were not evaluated and remain null.',
            'No observed ESS or convergence rate proves equilibrium mode coverage.']}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,4,figsize=(15,2.5+.5*len(rows)),sharey=True,layout='constrained')
    y=np.arange(len(rows));colors=['#40789e' if row['label'].startswith('FM') else '#b37040' for row in rows]
    axes[0].barh(y,[r['xtb_converged']/r['xtb_attempted'] for r in rows],color=colors)
    for i,row in enumerate(rows):axes[0].text(.02,i,f"{row['xtb_converged']}/{row['xtb_attempted']}",va='center',color='black')
    axes[0].set_xlim(0,1.05);axes[0].set_title('xTB relaxation convergence')
    for i,row in enumerate(rows):
        value=row['strain_eV_success_only']
        if value is not None:
            axes[1].errorbar(value['median'],i,xerr=[[value['median']-value['q10']],[value['q90']-value['median']]],
                fmt='o',color=colors[i],capsize=3)
        else:axes[1].text(.02,i,'no converged samples',transform=axes[1].get_yaxis_transform(),va='center')
        if row['weights'] is None:axes[3].text(.02,i,'not evaluated',transform=axes[3].get_yaxis_transform(),va='center')
        else:
            axes[3].barh(i,row['weights']['ess']/row['particles'],color=colors[i])
            axes[3].text(.01,i,f"{row['weights']['ess']:.1f}/{row['particles']}",va='center')
    axes[1].set_xscale('log');axes[1].set_title('Strain: median, 10–90%\nconverged samples only (eV)')
    axes[2].barh(y,[r['overlap_count']/r['particles'] for r in rows],color=colors)
    axes[2].set_xlim(0,1);axes[2].set_title('Overlap incidence\nall generated samples')
    axes[3].set_xlim(0,1);axes[3].set_title('Importance ESS / particles')
    axes[0].set_yticks(y,[r['label'] for r in rows]);axes[0].invert_yaxis()
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same molecular condition: initialization, training and independent structure checks\nDevelopment evidence; budgets differ across calibrations',fontsize=12)
    fig.savefig(args.out/'work_campaign.pdf');fig.savefig(args.out/'work_campaign.png',dpi=180);plt.close(fig)
    print(json.dumps({'rows':len(rows),'out':str(args.out.resolve())}),flush=True)


if __name__=='__main__':main()
