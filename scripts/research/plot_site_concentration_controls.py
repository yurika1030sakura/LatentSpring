#!/usr/bin/env python3
"""Plot all completed cost-matched physical-width controls with uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.project
    files=['runs/fresh_reuse_summary_v1/results.json',
           'runs/transfer_reuse_summary_v1/results.json',
           'runs/site_concentration_summary_v1/results.json']
    original,transfer,controls=[json.loads((root/f).read_text()) for f in files]
    assert all(d['complete'] for d in [original,transfer,controls])
    assert original['primary_comparison']['available'] and transfer['primary_comparison']['available']
    original_stats={'primary':original['primary_comparison']['learned_minus_site_potential_eV'],
                    'transfer':transfer['primary_comparison']['paired_potential_difference_eV']}
    labels=[r'$\kappa=10$ (original)',r'$\kappa=64$',
            r'$\kappa=400$ (reuse)',r'$\kappa=400$ (calibration charged)']
    rows=[]
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.0),sharex=True,sharey=True)
    for ax,cohort,title in zip(axes,['primary','transfer'],
            ['Original composition (72 parents)','Six additional compositions (96 parents)']):
        entries=[dict(available=True,statistics=original_stats[cohort])]
        for k,scenario in [(64,'matched_total_incremental_calibration'),
                           (400,'matched_total_incremental_calibration'),
                           (400,'matched_total_fully_charged_calibration')]:
            found=[r for r in controls['rows'] if r['cohort']==cohort and r['kappa']==k and r['scenario']==scenario]
            assert len(found)==1
            row=found[0]
            entries.append(dict(available=row['available'],statistics=row.get('learned_minus_physical_potential_eV')))
        for i,(label,entry) in enumerate(zip(labels,entries)):
            record=dict(cohort=cohort,physical_control=label,available=entry['available'])
            if entry['available']:
                stats=entry['statistics'];mean=stats['mean']
                key='parent_bootstrap_95_percent_interval' if cohort=='primary' else 'hierarchical_95_percent_interval'
                lo,hi=stats[key]
                ax.plot([lo,hi],[i,i],color='#176B87',lw=2)
                ax.plot(mean,i,'o',color='#176B87',markersize=5)
                record.update(mean_eV=mean,lower_95_eV=lo,upper_95_eV=hi)
            else:ax.text(0,i,' unavailable',va='center',color='gray')
            rows.append(record)
        ax.axvline(0,color='black',ls='--',lw=.8)
        ax.set_title(title,fontsize=10)
        ax.set_yticks(range(4),labels);ax.set_ylim(3.6,-.6)
        ax.grid(axis='x',alpha=.15);ax.set_xlabel('Learned minus physical potential (eV)')
        for side in ['top','right']:ax.spines[side].set_visible(False)
    fig.text(.54,.025,'Negative values favor learning. Training/preparation costs included.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1])
    args.out.mkdir(parents=True,exist_ok=True)
    for name in ['controls.pdf','controls.png','values.csv','results.json']:
        if (args.out/name).exists():raise FileExistsError(args.out/name)
    fig.savefig(args.out/'controls.pdf',bbox_inches='tight')
    fig.savefig(args.out/'controls.png',dpi=180,bbox_inches='tight');plt.close(fig)
    with (args.out/'values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['cohort','physical_control','available','mean_eV','lower_95_eV','upper_95_eV'])
        writer.writeheader();writer.writerows(rows)
    result=dict(complete=True,input_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
        rows=rows,physical_calibration_cost_for_kappa400=10108,new_physical_queries=0,
        scientific_submission_ready=False,scope='Both cohorts and all prespecified matched-total control comparisons; intervals conditional on the fixed learned checkpoints. No equilibrium or wall-time claim.')
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
