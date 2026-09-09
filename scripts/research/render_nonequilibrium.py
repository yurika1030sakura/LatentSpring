#!/usr/bin/env python3
"""Render every seed and control of the completed known-target work experiment."""
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
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--figure-dir',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.source.read_text())
    if not data['complete'] or len(data['rows'])!=data['protocol']['seeds']:raise ValueError('Incomplete experiment')
    if not all(set(row['distilled'])=={'weighted','unweighted'} for row in data['rows']):raise ValueError('Missing distillation controls')
    arms=['unweighted_same_paths','terminal_energy_only','weighted_teacher','distilled_unweighted','distilled_weighted']
    labels=['Raw\npaths','Energy\nonly','Work\nweighted','FM from\nraw paths','FM from\nwork weights']
    colors=['#8093ab','#909090','#007e7a','#b3c3d4','#d17b26']
    def value(row,arm,metric):
        if arm.startswith('distilled_'):return row['distilled'][arm.removeprefix('distilled_')]['metrics'][metric]
        return row[arm][metric]
    metrics=['right_basin_probability','mean_nearest_mode_squared_distance']
    summary={'complete':True,'scope':data['scope'],'source':str(args.source.resolve()),
        'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'source_code_sha256':data['source_sha256'],'protocol':data['protocol'],'target':data['target'],
        'rows':data['rows'],'aggregate':data['aggregate'],
        'mean_ess_fraction':float(np.mean([row['ais']['ess_fraction'] for row in data['rows']])),
        'exponential_noise_counterexample':data['exponential_noise_counterexample']}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for ax,metric,truth,title in zip(axes,metrics,[.8,.5],['Basin population','Spread around the nearest mode']):
        values=np.array([[value(row,arm,metric) for arm in arms] for row in data['rows']])
        ax.bar(np.arange(5),values.mean(0),color=colors,width=.65,alpha=.85)
        for i in range(len(data['rows'])):
            ax.scatter(np.arange(5)+(i-(len(data['rows'])-1)/2)*.055,values[i],c='black',s=12,zorder=3)
        ax.axhline(truth,c='#b53a35',ls='--',lw=1.4,label=f'Target = {truth:g}')
        ax.set_xticks(np.arange(5),labels,fontsize=9)
        ax.set_title(title,fontsize=12);ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Probability of x > 0');axes[0].set_ylim(0,1)
    axes[1].set_ylabel('Mean squared distance')
    fig.suptitle('Correct work weights repair the teacher; distillation remains approximate',fontsize=13)
    fig.text(.5,.015,'Known 2D target; five seeds; all controls shown. This is not a molecular performance result.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93])
    args.evidence_dir.mkdir(parents=True,exist_ok=True);args.figure_dir.mkdir(parents=True,exist_ok=True)
    (args.evidence_dir/'nonequilibrium_toy.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig.savefig(args.figure_dir/'nonequilibrium_toy.pdf',bbox_inches='tight')
    fig.savefig(args.evidence_dir/'nonequilibrium_toy.png',dpi=170,bbox_inches='tight')
    print(json.dumps({'aggregate':summary['aggregate'],'mean_ess_fraction':summary['mean_ess_fraction']}))


if __name__=='__main__':main()
