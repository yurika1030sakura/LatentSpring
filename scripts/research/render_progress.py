#!/usr/bin/env python3
"""Render completed molecular development evidence without filtering failures."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    files={
        'full':'position_density_v1/panel.json',
        'smooth':'smooth_density_v3/panel.json',
        'local':'local_density_mature_v2/panel.json',
        'sample_initial':'position_samples_v1/samples.json',
        'sample_mature':'position_assessment_10000_v1/sampling/samples.json',
        'xtb_initial':'position_xtb_v1/xtb.json',
        'xtb_mature':'position_assessment_10000_v1/xtb/xtb.json',
        'checkpointed':'highres_energy_smoke_v1/validation.json',
        'adjoint':'adjoint_energy_smoke_v1/validation.json'}
    data={};sources={}
    for key,file in files.items():
        path=args.runs/file;data[key]=json.loads(path.read_text())
        if not data[key]['complete']:raise ValueError(f'Incomplete input {file}')
        sources[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    table={}
    for name in ['full','smooth','local']:
        table[name]=[{'parent_id':r['parent_id'],'n_atoms':r['n_atoms'],
                      'steps':r['resolutions'][-1]['steps'],
                      'max_mean_centered_drift_nat':r['resolutions'][-1]['max_mean_centered_change_from_previous'],
                      'max_replica_centered_drift_nat':r['resolutions'][-1]['max_centered_change_from_previous']}
                     for r in data[name]['rows']]
    assert [r['parent_id'] for r in table['full']]==[r['parent_id'] for r in table['smooth']]
    table['sampling']=[];all_xtb=[]
    for sample_name,xtb_name in [('sample_initial','xtb_initial'),('sample_mature','xtb_mature')]:
        for arm in data[sample_name]['arms']:
            name=arm['name'];rows=[r for r in data[xtb_name]['rows'] if r['arm']==name]
            successes=[r for r in rows if r['success']]
            assert len(rows)==len(arm['samples'])==32
            table['sampling'].append({'arm':name,'attempted':len(rows),'converged':len(successes),
                'median_strain_eV_success_only':st.median(r['strain_eV'] for r in successes),
                'mean_strain_eV_success_only':st.mean(r['strain_eV'] for r in successes),
                'held_fm_mean':st.mean(r['loss'] for r in arm['held_fm_losses']),
                'max_coordinate_drift_A':max(r['coordinate_rms_64_128_A'] for r in arm['samples'])})
            all_xtb.extend(rows)
    arms=[r['arm'] for r in table['sampling']]
    paired={a:{(r['validation_index'],r['sample_id']):r for r in all_xtb if r['arm']==a} for a in arms}
    assert all(set(paired[a])==set(paired[arms[0]]) for a in arms)
    table['paired']=[]
    for before,after in zip(arms[:-1],arms[1:]):
        counts={'better':0,'worse':0,'both_failed':0,'tie':0}
        for key,left in paired[before].items():
            right=paired[after][key]
            if not left['success'] and not right['success']:outcome='both_failed'
            elif not left['success']:outcome='better'
            elif not right['success']:outcome='worse'
            else:
                delta=right['strain_eV']-left['strain_eV']
                outcome='tie' if abs(delta)<1e-6 else 'better' if delta<0 else 'worse'
            counts[outcome]+=1
        table['paired'].append({'before':before,'after':after,**counts,
                               'ranking':'nonconverged ranks below converged; otherwise lower strain'})
    table['compute']={key:{'seconds':data[key]['seconds'],'peak_GiB':data[key]['peak_gpu_bytes']/2**30}
                      for key in ['checkpointed','adjoint']}
    report={'sources':sources,'tables':table,'claim':'one-seed conditional FM and numerical engineering development; no energy-training performance claim'}
    (args.out/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    ax=axes[0,0];x=np.arange(8)
    for name,label,color in [('full','Original geometry','#ba4d45'),('smooth','Smooth geometry, no retraining','#548bb0')]:
        ax.plot(x,[r['max_mean_centered_drift_nat'] for r in table[name]],'o-',label=label,color=color)
    ax.set_xticks(x,[r['parent_id'] for r in table['full']],rotation=35)
    ax.set_yscale('log');ax.set_ylabel('Max centred mean density drift (nats)')
    ax.set_title('A. Full neighbourhood: 64 → 128 steps, 1,000-update head',loc='left',fontsize=10)
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[0,1];drift=[r['max_mean_centered_drift_nat'] for r in table['local']]
    ax.bar(x,drift,color=['#ba4d45' if y>.1 else '#548bb0' for y in drift])
    ax.axhline(.1,ls='--',color='black',lw=1,label='Necessary gate: 0.1 nat')
    ax.set_xticks(x,[r['parent_id'] for r in table['local']],rotation=35)
    ax.set_ylabel('Max centred mean density drift (nats)')
    ax.set_title('B. Local neighbourhood: 128 → 256, 10,000 updates',loc='left',fontsize=10)
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1,0];labels=['Checkpointed','Discrete adjoint']
    memories=[table['compute'][k]['peak_GiB'] for k in ['checkpointed','adjoint']]
    ax.bar(labels,memories,color=['#a3a3a3','#548bb0'])
    for i,key in enumerate(['checkpointed','adjoint']):
        item=table['compute'][key]
        ax.text(i,memories[i]+.5,f"{memories[i]:.2f} GiB\n{item['seconds']:.0f} s",ha='center')
    ax.set_ylim(0,24);ax.set_ylabel('Peak allocated GPU memory (GiB)')
    ax.set_title('C. Same four energy updates, 128 integration steps',loc='left',fontsize=10)
    ax=axes[1,1];vals=[r['median_strain_eV_success_only'] for r in table['sampling']]
    ax.plot(range(3),vals,'o-',color='#548bb0')
    for i,r in enumerate(table['sampling']):ax.annotate(f"{r['converged']}/32 converged",(i,vals[i]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(range(3),['Warm q_0.8','1,000 FM steps','10,000 FM steps'])
    ax.set_ylim(0,16);ax.set_xlim(-.3,2.3);ax.set_ylabel('Median xTB strain among successes (eV)')
    ax.set_title('D. Matched priors, eight parents; one training seed',loc='left',fontsize=10)
    ax.grid(axis='y',alpha=.2)
    fig.savefig(args.out/'molecular_progress.pdf');fig.savefig(args.out/'molecular_progress.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'sampling':table['sampling'],'paired':table['paired'],'compute':table['compute']},indent=2))


if __name__=='__main__':main()
