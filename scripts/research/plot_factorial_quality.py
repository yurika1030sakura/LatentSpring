#!/usr/bin/env python3
"""Plot the audited source-by-physical-update comparison without outcome selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[2]
    data=json.loads((root/'research/evidence/source_physical_factorial_audit_v1.json').read_text());assert data['complete']
    out=root/'research/figures/primary_quality_v2';out.mkdir(parents=True,exist_ok=True)
    methods=['gaussian','gaussian_physical','harmonic_tree','harmonic_physical']
    labels=['Gaussian','Gaussian + physical update','Harmonic','Harmonic + physical update']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.85),sharey=True)
    for ax,kind,title in zip(axes,['esen','xtb'],['eSEN','Independent GFN2-xTB']):
        for method,label,color,style in zip(methods,labels,['#7d8793','#5576b2','#21887b','#c15b38'],['--','-','--','-']):
            rows=data['summary'][kind][method]['force_yield']
            ax.plot([r['threshold'] for r in rows],[100*r['mean'] for r in rows],label=label,color=color,ls=style,marker='o',ms=3,lw=1.7)
        ax.set_xscale('log');ax.set_xticks([1,2,5,10,20,50,100],[1,2,5,10,20,50,100]);ax.set_ylim(0,57)
        ax.set_title(title);ax.set_xlabel('Force-RMS threshold (eV/Å)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Graph-valid joint yield (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.035))
    fig.tight_layout(rect=(0,0,1,.81))
    for suffix in ['pdf','png']:fig.savefig(out/f'primary_quality.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
