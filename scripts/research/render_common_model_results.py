"""Publication-width comparison of graph support and joint physical quality."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--grayscale',action='store_true');a=p.parse_args();d=json.loads(a.audit.read_text());assert d['complete']
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[('gaga','GAGA · 128 calls','#667085'),('gaga_full','GAGA · 651 calls','#667085'),('published','Paired physical update','#b36d36'),('base','Frozen harmonic parent','#536c8a'),('pair_long','Learned force correction','#087f8c')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.labelsize':8.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.55})
    fig,ax=plt.subplots(figsize=(5.5,2.4));fig.subplots_adjust(left=.35,right=.96,bottom=.22,top=.85)
    for index,(method,label,color) in enumerate(rows):
        color='#555555' if a.grayscale else color
        y=len(rows)-1-index;r=d['summary'][method];graph=100*r['graph_rate'];joint=100*r['joint5']
        ax.plot([joint,graph],[y,y],color=color,lw=2,alpha=.35,zorder=1)
        ax.scatter([graph],[y],s=40,facecolor='white',edgecolor=color,linewidth=1.2,zorder=3)
        ax.scatter([joint],[y],s=40,facecolor=color,edgecolor='white',linewidth=.5,zorder=4)
        ax.text(joint-.95,y-.22,f'{joint:.2f}',ha='right',va='top',color=color,fontsize=7.5)
    ax.set_yticks(range(len(rows)),[r[1] for r in rows[::-1]]);ax.tick_params(axis='y',length=0,pad=9)
    ax.set_xlim(0,60);ax.set_xticks([0,10,20,30,40,50,60]);ax.set_ylim(-.55,4.5);ax.set_xlabel('Raw outputs passing the criterion (%)',labelpad=6)
    ax.tick_params(axis='x',length=3,width=.6,color='#667085');ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#b5bdc7')
    for x in [0,20,40,60]:ax.axvline(x,color='#edf0f3',lw=.55,zorder=0)
    fig.text(.035,.94,'Same compositions · raw generated coordinates',fontsize=9.3,weight='bold',color='#273444')
    fig.text(.035,.855,'●  Graph-valid and force RMS ≤ 5 eV/Å     ○  Graph-valid',fontsize=8.1,color='#4c596b')
    for suffix in ['pdf','svg','png']:fig.savefig(a.out/('common_model_quality.'+suffix),dpi=240,facecolor='white')
    # Human-readable design record, with numeric and statistical provenance.
    record=dict(audit=str(a.audit),audit_sha256=sha(a.audit),figure_width_inches=5.5,rows=[dict(method=m,label=l,**d['summary'][m]) for m,l,c in rows],
        design='Open circles show graph support; filled circles show the subset also passing the physical threshold. Zero-based axis covering all reported values. Paired confidence intervals are reported in the adjacent text/table, not implied by marker size.',
        comparison_scope=d['comparison_scope'])
    (a.out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');plt.close(fig)


if __name__=='__main__':main()
