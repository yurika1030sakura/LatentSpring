#!/usr/bin/env python3
"""Export structural success, unresolved validator cases, and paired uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();s=json.loads(a.comparison.read_text());assert s['complete']
    names=['warm','independent','rotation','steric','typed_rotation']
    labels=['Warm FM','Independent','Rotation','Collision-aware','Type matching']
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.6),gridspec_kw={'width_ratios':[1.15,1]})
    x=np.arange(len(names));bottom=np.zeros(len(names))
    counts=np.array([[s['methods'][m][k] for k in ['attempted','graph_supported','geometrically_supported','validator_errors']] for m in names])
    parts=[counts[:,1],counts[:,2]-counts[:,1]-counts[:,3],counts[:,3],counts[:,0]-counts[:,2]]
    for values,color,label in zip(parts,['#34856c','#e5bd61','#aaaaaa','#e7dfdc'],
            ['Graph accepted','Geometry passed, graph rejected','Validator exception','Geometry failed']):
        assert (values>=0).all()
        height=100*values/counts[:,0]
        axes[0].bar(x,height,bottom=bottom,color=color,label=label,width=.7);bottom+=height
    axes[0].set_xticks(x,labels,rotation=25,ha='right',fontsize=8)
    axes[0].set_ylim(0,100);axes[0].set_ylabel('All attempted structures (%)')
    axes[0].set_title('512 outputs per model; all 8 conditions',fontsize=10)
    controls=['warm','independent','rotation','typed_rotation']
    control_labels=['Warm FM','Independent','Rotation','Type matching']
    for i,control in enumerate(controls):
        row=s['comparisons']['steric minus '+control]
        mean=100*row['graph_support_fraction_difference']
        lo,hi=[100*v for v in row['conditional_paired_bootstrap95']]
        axes[1].plot([lo,hi],[i,i],color='#555555',linewidth=2)
        axes[1].plot(mean,i,'o',color='#923c70',markersize=5)
    axes[1].axvline(0,color='#999999',linestyle='--',linewidth=1)
    axes[1].set_yticks(range(4),control_labels,fontsize=8);axes[1].invert_yaxis()
    axes[1].set_xlabel('Collision-aware minus control\nGraph-acceptance difference (percentage points)',fontsize=9)
    axes[1].set_title('Paired bootstrap 95% intervals',fontsize=10)
    axes[1].grid(axis='x',alpha=.2)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Main-generator FM pairing pilot: one matched training seed',fontsize=12)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower left',bbox_to_anchor=(.05,.11),ncol=2,fontsize=8,frameon=False)
    fig.text(.5,.015,'Intervals condition on the fixed models and compositions.\nGraph perception is not an equilibrium or quantum-validity certificate.',fontsize=8,ha='center')
    fig.subplots_adjust(left=.08,right=.98,bottom=.38,top=.84,wspace=.4)
    a.out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        path=a.out/('orbit_pairing_v2.'+ext)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
