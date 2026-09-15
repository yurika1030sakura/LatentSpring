#!/usr/bin/env python3
"""Plot the frozen threshold protocol from audited physical-quality yields."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[2]
    report=json.loads((root/'research/evidence/generator_quality_audit_v1.json').read_text())
    assert report['complete'] and report['reference_inversion_gate']
    out=root/'research/figures/generator_quality_v1';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),sharey=True)
    labels=['Gaussian FM','Harmonic FM','EDM','GAGA'];colors=['#747474','#16817a','#d18a28','#405bb1']
    for ax,kind,title in zip(axes,['esen','xtb'],['eSEN','GFN2-xTB']):
        for m,label,color in zip(report['methods'],labels,colors):
            data=report['summary'][kind][m]['force_yield']
            ax.plot([r['threshold'] for r in data],[100*r['rate'] for r in data],label=label,color=color,marker='o',ms=3,lw=1.7)
        ax.set_xscale('log');ax.set_xticks([1,2,5,10,20,50,100],[1,2,5,10,20,50,100])
        ax.set_title(title);ax.set_xlabel('Force-RMS threshold (eV/Å)');ax.set_ylim(0,15.5)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Joint quality yield (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.03))
    fig.tight_layout(rect=(0,0,1,.90))
    for suffix in ['pdf','png']:fig.savefig(out/f'quality_yield.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
