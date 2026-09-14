#!/usr/bin/env python3
"""Make a compact, all-controls source utility result artifact from a completed audit."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.audit.read_text());assert d['complete'] and len(d['summary'])==2
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    methods=['fixed','actual','shuffled','nll'];labels=['Fixed','Actual utility','Shuffled utility','Coordinate NLL']
    with (a.out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','method',*next(iter(d['summary']['0'].values())).keys()]);writer.writeheader()
        for seed,rows in d['summary'].items():
            for method in methods:writer.writerow(dict(seed=seed,method=method,**rows[method]))
    fig,axes=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
    for seed,ax in enumerate(axes):
        rows=d['summary'][str(seed)]
        bars=ax.bar(range(4),[100*rows[m]['graph']/rows[m]['attempts'] for m in methods],color=['#777777','#007f87','#c78d45','#8b6daf'])
        for bar,m in zip(bars,methods):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.8,str(rows[m]['graph'])+'/'+str(rows[m]['attempts']),ha='center',fontsize=8)
        ax.set_xticks(range(4),labels,rotation=18,ha='right');ax.set_title(f'Frozen decoder {seed+1}')
        ax.set_ylim(0,80);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Graph support per attempted draw (%)')
    fig.suptitle('Fresh source-head held-out compositions; flow-training-corpus domain',fontsize=10)
    fig.tight_layout();fig.savefig(a.out/'source_utility_results.pdf');fig.savefig(a.out/'source_utility_results.png',dpi=180);plt.close(fig)
    lines=['# Source utility pilot: audited fresh generation','',
        'These12 compositions are held out from the source head, but originate from the flow training corpus. Both frozen decoders and all four prespecified methods are retained.','',
        '| Decoder | Fixed | Actual utility | Shuffled utility | Coordinate NLL |','|---|---:|---:|---:|---:|']
    for s in ['0','1']:
        rows=d['summary'][s];lines.append('| '+str(int(s)+1)+' | '+' | '.join(f"{rows[m]['graph']}/{rows[m]['attempts']}" for m in methods)+' |')
    lines+=['','Pooled actual-minus-control graph-rate differences; paired draw and descriptive size-stratified composition intervals are both retained:','']
    for m in ['fixed','shuffled','nll']:
        r=d['comparisons'][m]['graph_supported'];ci=r['paired_draw95'];cc=r['descriptive_composition95']
        lines.append(f"- vs {m}: {100*r['mean']:+.2f} pp; paired95 [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}], composition95 [{100*cc[0]:+.2f}, {100*cc[1]:+.2f}].")
    lines+=['',f"Prespecified development gate passed: **{d['development_gate_passed']}**. This is not an ICLR readiness or final target-distribution result.",
        f"All{d['source_draws_replayed']} source draws and structural assays replay. No new molecular oracle calls. Train-bank preparation and source fitting must be included in cost comparisons.",
        'The existing negative coordinate-NLL, static-context and monomer studies remain unchanged. The pilot cannot establish broad failure of all source utility objectives or justify tuning this recipe on the observed validation outcomes.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
