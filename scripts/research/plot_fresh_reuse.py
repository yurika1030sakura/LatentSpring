#!/usr/bin/env python3
"""Export audited fresh-parent finite-cost comparisons as standalone figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.evaluate_chemical_policy import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    d=json.loads(args.summary.read_text())
    assert d['complete']
    primary=d['primary_comparison']
    if not primary['available']:
        raise ValueError('Primary comparison has missing endpoints; do not plot a selected subset')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'fresh_reuse.pdf').exists():
        raise FileExistsError(args.out/'fresh_reuse.pdf')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),gridspec_kw={'width_ratios':[1.2,1.25,.85]})
    n=primary['parents']
    for match,color,label,offset in [(False,'#2965a5','Same inference calls',-5),(True,'#b24920','Same total calls',5)]:
        rows=[r for r in d['comparisons'] if r['parents']==n and r['match_training_cost']==match and r['available']]
        x=np.array([r['learned_queries_per_parent'] for r in rows])
        values=np.array([r['learned_minus_site_potential_eV']['mean'] for r in rows])
        bounds=np.array([r['learned_minus_site_potential_eV']['parent_bootstrap_95_percent_interval'] for r in rows])
        axes[0].errorbar(x+offset,values,yerr=np.vstack([values-bounds[:,0],bounds[:,1]-values]),fmt='o-',
            color=color,label=label,capsize=3,lw=1.3,ms=4)
    axes[0].axhline(0,color='.5',lw=.8,ls='--')
    axes[0].set(xlabel='Learned inference calls / parent',ylabel='Learned − physical potential (eV)',title=f'{n} fresh parents; two replicas')
    axes[0].legend(frameon=False,fontsize=8)
    pairs=np.array(primary['paired_potential_differences_eV'])
    order=np.argsort(pairs.mean(0))
    axes[1].scatter(np.arange(n),pairs[0,order],s=12,color='#2965a5',alpha=.7,label='Replica 0')
    axes[1].scatter(np.arange(n),pairs[1,order],s=12,color='#b24920',alpha=.7,label='Replica 1')
    axes[1].axhline(0,color='.5',lw=.8,ls='--')
    axes[1].set(xlabel='Parent rank by mean paired difference',ylabel='Paired potential difference (eV)',title='Primary comparison: same total calls')
    axes[1].legend(frameon=False,fontsize=8)
    supported=d['supported_source_parents'];attempted=d['attempted_source_parents']
    axes[2].bar([0],[attempted-supported],color='#cccccc',label='Unsupported')
    axes[2].bar([0],[supported],bottom=[attempted-supported],color='#37826c',label='Supported')
    axes[2].set(xticks=[0],xticklabels=['Fresh FM source'],ylabel='Generated source attempts',title='All source attempts retained')
    axes[2].text(0,attempted*1.03,f'{supported}/{attempted} supported\n({100*supported/attempted:.2f}%)',ha='center',va='bottom',fontsize=9)
    axes[2].set_ylim(0,attempted*1.28)
    axes[2].legend(frameon=False,fontsize=8,loc='center')
    fig.text(.5,.01,'One development composition. Lower paired potential favors learning; finite-cost outputs are not equilibrium certification.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ['pdf','png']:
        fig.savefig(args.out/f'fresh_reuse.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (args.out/'provenance.json').write_text(json.dumps(dict(summary=str(args.summary),summary_sha256=sha(args.summary),
        files={ext:sha(args.out/f'fresh_reuse.{ext}') for ext in ['pdf','png']},scientific_submission_ready=False),indent=2)+'\n')


if __name__=='__main__':main()
