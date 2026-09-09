#!/usr/bin/env python3
"""Build figures and machine-readable tables from completed development runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    names={'affine':'toy_trace_v1/toy.json','nonlinear':'toy_nonlinear_v1/toy.json',
           'old_density':'numerical_panel_v1/panel.json','sampling':'position_samples_v1/samples.json',
           'xtb':'position_xtb_v1/xtb.json','split':'composition_audit_v2/composition_split.json'}
    data={};sources={}
    for name,path in names.items():
        file=args.runs/path;data[name]=json.loads(file.read_text())
        if data[name].get('complete') is False:raise ValueError(f'Incomplete source: {path}')
        sources[name]={'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    tables={}
    for ax,name,title,key in [(axes[0,0],'affine','A. Affine Gaussian: common probes cancel noise','exact_kl'),
                            (axes[0,1],'nonlinear','B. Nonlinear shear: cancellation is incomplete','exact_kl_q_model_to_target')]:
        rows=data[name]['rows'];arms=list(dict.fromkeys(r['arm'] for r in rows));summaries=[]
        for i,arm in enumerate(arms):
            selected=[r for r in rows if r['arm']==arm];ys=[r[key] for r in selected]
            ax.scatter(i+np.linspace(-.13,.13,len(ys)),ys,s=25,alpha=.8)
            ax.plot([i-.2,i+.2],[statistics.mean(ys)]*2,color='black',lw=1.5)
            summaries.append({'arm':arm,'seeds':len(selected),'mean_exact_kl':statistics.mean(ys),
                              'per_seed_exact_kl':ys,'per_seed_a':[r['a'] for r in selected]})
        labels={'exact':'Exact','squared_2':'Squared','product_2':'Product','common_2':'Common\nsquared',
                'rotated_squared_2':'Rotated\nsquared','squared_rademacher':'Rad.\nsquared',
                'product_rademacher':'Rad.\nproduct','squared_common_rademacher':'Common\nRad. sq.',
                'product_common_rademacher':'Common\nRad. prod.','squared_gaussian':'Gauss.\nsquared',
                'product_gaussian':'Gauss.\nproduct','squared_common_gaussian':'Common\nGauss. sq.'}
        ax.set_xticks(range(len(arms)),[labels[a] for a in arms],fontsize=7)
        ax.set_yscale('symlog',linthresh=1e-6);ax.set_ylim(bottom=-1e-7)
        ax.set_ylabel('Exact KL(model || target)');ax.set_title(title,loc='left',fontsize=10)
        ax.grid(axis='y',alpha=.2);tables[name]=summaries
    ax=axes[1,0];rows=data['old_density']['rows']
    drifts=[r['resolutions'][-1]['max_mean_centered_change_from_previous'] for r in rows]
    ax.bar(range(len(rows)),drifts,color=['#ba4d45' if v>1 else '#548bb0' for v in drifts])
    ax.set_xticks(range(len(rows)),[str(r['parent_id']) for r in rows],rotation=35,fontsize=8)
    ax.set_yscale('log');ax.set_ylabel('Max centred mean log-density change (nats)')
    ax.set_title('C. Old checkpoint: 64 → 128 steps is not converged',loc='left',fontsize=10)
    ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];xtb=data['xtb']['rows'];sample_arms=[a['name'] for a in data['sampling']['arms']]
    strain={}
    for arm in sample_arms+['reference']:
        rs=[r for r in xtb if r['arm']==arm];ok=[r for r in rs if r['success']]
        strain[arm]={'attempted':len(rs),'converged':len(ok),
            'median_strain_eV_success_only':statistics.median(r['strain_eV'] for r in ok),
            'mean_strain_eV_success_only':statistics.mean(r['strain_eV'] for r in ok)}
    parent_rows=[]
    for parent in data['sampling']['validation_indices']:
        item={'parent':parent}
        for arm in sample_arms:
            good=[r['strain_eV'] for r in xtb if r['arm']==arm and r['validation_index']==parent and r['success']]
            item[arm]=statistics.median(good) if good else None
            item[arm+'_converged']=len(good)
        if all(item[a] is not None for a in sample_arms):
            ax.scatter(item[sample_arms[0]],item[sample_arms[1]],s=35,color='#548bb0')
            ax.annotate(str(parent),(item[sample_arms[0]],item[sample_arms[1]]),xytext=(3,3),textcoords='offset points',fontsize=7)
        parent_rows.append(item)
    lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],'--',color='gray',lw=1)
    ax.set(xlabel='Warm-start q_0.8: median strain (eV)',ylabel='After 1,000 conditional FM steps (eV)')
    ax.set_title('D. Preliminary xTB check, eight development parents',loc='left',fontsize=10)
    fig.savefig(args.out/'development_evidence.pdf');fig.savefig(args.out/'development_evidence.png',dpi=180)
    plt.close(fig)
    tables['strain']=strain;tables['parent_strain']=parent_rows
    tables['sampling']=[]
    for arm in data['sampling']['arms']:
        rs=arm['samples'];drift=[r['coordinate_rms_64_128_A'] for r in rs]
        tables['sampling'].append({'arm':arm['name'],'n':len(rs),
            'held_fm_mean':statistics.mean(r['loss'] for r in arm['held_fm_losses']),
            'coordinate_drift_median_A':statistics.median(drift),'coordinate_drift_max_A':max(drift),
            'samples_above_0_01_A':sum(v>.01 for v in drift)})
    report={'sources':sources,'tables':tables,'claim':'development evidence only; molecular energy-training benefit unestablished'}
    (args.out/'evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'out':str(args.out),'strain':strain},indent=2))


if __name__=='__main__':main()
