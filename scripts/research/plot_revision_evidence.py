"""Consistent publication graphics from completed, audited experiment tables."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    for name in ['left','bottom']:ax.spines[name].set_color('#d1dade')
    ax.tick_params(color='#c3cdd2',labelcolor='#526773',labelsize=8)
    ax.grid(axis='y',color='#e9edef',linewidth=.6);ax.set_axisbelow(True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path);parser.add_argument('--publication-size',action='store_true')
    args=parser.parse_args();compact=args.publication_size
    root=Path(__file__).resolve().parents[2];out=args.out or root/'research/figures/evidence_revision_v1';out.mkdir(exist_ok=True,parents=True)
    width=5.5 if compact else 7.5
    plt.rcParams.update({'font.family':'Liberation Sans' if compact else 'DejaVu Sans','font.size':8 if compact else 9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'text.color':'#233746','axes.labelcolor':'#233746','axes.titleweight':'bold'})
    d=json.loads((root/'research/evidence/source_replication_audit_v1.json').read_text());assert d['complete']
    curves=d['new_three_seed_budget_curves'];x=np.array([1000,3000,6000]);y=np.array([curves[str(t)]['by_seed'] for t in x])*100
    fig,axes=plt.subplots(1,2,figsize=(width,2.6 if compact else 2.75),gridspec_kw={'width_ratios':[1.35,1]})
    for ax in axes:style(ax);ax.axhline(0,color='#88979f',lw=.8,ls=(0,(3,3)))
    for i,col in enumerate(['#719ea3','#9aa6b5','#c4a386']):
        axes[0].plot(x,y[:,i],color=col,lw=1,marker='o',ms=3,label=f'Run {i+3}')
    ci=np.array([curves[str(t)]['composition_ci95'] for t in x])*100
    axes[0].fill_between(x,ci[:,0],ci[:,1],color='#16877e',alpha=.10,lw=0)
    axes[0].plot(x,y.mean(1),color='#147e75',lw=2.2,marker='o',ms=4,label='Mean')
    axes[0].set_xticks(x,['1,000','3,000','6,000']);axes[0].set_xlabel('Optimizer updates')
    axes[0].set_ylabel('Graph-validity difference\n(percentage points)',fontsize=8)
    axes[0].set_title('a   Training budget',loc='left',fontsize=9 if compact else 10,pad=10)
    if compact:fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,fontsize=7.5,ncol=4,loc='lower center',bbox_to_anchor=(.52,-.01))
    else:axes[0].legend(frameon=False,fontsize=7.5,ncol=2,loc='upper left')
    s=d['fixed3000_five_continuations'];effect=np.array(s['by_seed'])*100
    axes[1].scatter(np.arange(1,6),effect,s=32,color=['#577184']*2+['#16877e']*3,zorder=5)
    mean=s['mean']*100;low,high=np.array(s['composition_ci95'])*100
    axes[1].axhspan(low,high,color='#16877e',alpha=.1,lw=0);axes[1].axhline(mean,color='#147e75',lw=1.4)
    axes[1].text(.97,.94,f'Mean +{mean:.2f} pp\n4 of 5 runs improve',ha='right',va='top',transform=axes[1].transAxes,fontsize=8,color='#147e75')
    axes[1].set_xticks(range(1,6));axes[1].set_xlabel('Fine-tuning run');axes[1].set_ylim(-5,13)
    axes[1].set_title('b   Five runs, 3,000 updates' if compact else 'b   All runs at 3,000 updates',loc='left',fontsize=9 if compact else 10,pad=10)
    fig.tight_layout(w_pad=2.4,rect=(0,.12 if compact else 0,1,1))
    for ext in ['pdf','svg','png']:fig.savefig(out/f'source_training.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    d=json.loads((root/'research/evidence/source_physical_factorial_audit_v1.json').read_text());assert d['complete']
    methods=['gaussian','gaussian_physical','harmonic_tree','harmonic_physical']
    cols=['#88949e','#b27b4c','#61a89f','#167e76'];labels=['Gaussian','Gaussian + physical update','Harmonic','Harmonic + physical update']
    fig,axes=plt.subplots(1,2,figsize=(width,2.7 if compact else 3.0),sharey=True)
    titles=['a   eSEN','b   Independent GFN2-xTB'] if compact else ['a   Training potential · eSEN','b   Independent potential · GFN2-xTB']
    for ax,kind,title in zip(axes,['esen','xtb'],titles):
        style(ax);ax.axvline(5,color='#c2cdd1',ls=(0,(2,3)),lw=1)
        for m,col,label in zip(methods,cols,labels):
            rows=d['summary'][kind][m]['force_yield'];v=np.array([r['mean'] for r in rows])*100;t=[r['threshold'] for r in rows]
            ax.plot(t,v,color=col,ls='-' if 'physical' in m else (0,(4,2)),lw=2 if m=='harmonic_physical' else 1.3,marker='o',ms=3,label=label)
        ax.set_xscale('log');ax.set_xticks([1,2,5,10,20,50,100],['1','2','5','10','20','50','100'])
        ax.set_ylim(0,60);ax.set_title(title,loc='left',fontsize=9,pad=12);ax.set_xlabel('Force-RMS threshold (eV/Å)')
    axes[0].set_ylabel('Graph-valid joint yield (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='lower center',frameon=False,fontsize=8,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.15,1,1),w_pad=2.2)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'primary_quality.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    for p in out.glob('*.svg'):p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')


if __name__=='__main__':main()
