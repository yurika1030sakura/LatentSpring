"""Compact shared-backbone transfer figure from audited unoptimized outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--grayscale',action='store_true');a=p.parse_args()
    d=json.loads(a.audit.read_text());assert d['complete'];a.out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.55})
    color='#555555' if a.grayscale else '#087f8c';gray='#6b7480'
    fig,(ax,bx)=plt.subplots(1,2,figsize=(5.5,2.55),gridspec_kw={'width_ratios':[1.25,1.]})
    fig.subplots_adjust(left=.10,right=.98,bottom=.27,top=.77,wspace=.55)
    for i,name in enumerate(['fm','gaga']):
        keys=[name+'_a0',name+'_a1'];values=[100*d['summary'][k]['joint_rate'] for k in keys];xs=np.array([i-.13,i+.13])
        ax.plot(xs,values,color='#b5bdc7',lw=1.5,zorder=1)
        for x,value,key,col in zip(xs,values,keys,[gray,color]):
            seeds=np.array(d['summary'][key]['joint_by_seed'])*100
            ax.scatter([x-.023,x+.023],seeds,s=12,marker='x',color=col,linewidth=.7,zorder=2)
            ax.scatter([x],[value],s=42,color=col,edgecolors='white',linewidths=.65,zorder=3)
            ax.annotate(f'{value:.1f}',(x,value),xytext=(-6 if key.endswith('a0') else 6,6),textcoords='offset points',ha='right' if key.endswith('a0') else 'left',fontsize=7.4,color=col)
    ax.set_xticks([0,1],['FM','GAGA']);ax.set_xlim(-.48,1.48)
    maximum=max(max(r['joint_by_seed']) for r in d['summary'].values())*100
    top=np.ceil((maximum+5)/10)*10;ax.set_ylim(0,top);ax.set_yticks(np.arange(0,top+1,10));ax.set_ylabel('Joint quality yield (%)',labelpad=5)
    for i,name in enumerate(['fm','gaga']):
        r=d['contrasts'][name+'_improvement'];mean=100*r['mean'];lo,hi=np.array(r['ci95'])*100
        bx.errorbar(mean,1-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,markersize=5,capsize=3,lw=1.3,zorder=3)
        bx.text(mean,1-i+.24,f'{mean:+.2f}',color=color,fontsize=7.8,ha='center')
    bx.set_yticks([1,0],['FM','GAGA']);bx.set_ylim(-.55,1.55);bx.axvline(0,color='#c2c8cf',lw=.8,zorder=0)
    allci=np.array([d['contrasts'][n+'_improvement']['ci95'] for n in ['fm','gaga']])*100
    bx.set_xlim(min(-1.,allci.min()-2),max(5.,allci.max()+2));bx.set_xlabel('Gain (percentage points)',labelpad=6)
    for axis in [ax,bx]:
        axis.spines[['top','right']].set_visible(False);axis.spines[['left','bottom']].set_color('#b5bdc7');axis.tick_params(length=3,width=.6,color='#667085')
    bx.spines['left'].set_visible(False);bx.tick_params(axis='y',length=0)
    fig.text(.02,.94,'a',weight='bold',fontsize=10);fig.text(.075,.94,'Same EGNN · new compositions',fontsize=9)
    fig.text(.57,.94,'b',weight='bold',fontsize=10);fig.text(.62,.94,'Paired improvement',fontsize=9)
    fig.text(.075,.84,'● Parent    ',color=gray,fontsize=8);fig.text(.23,.84,'● + physical head',color=color,fontsize=8)
    fig.text(.62,.84,'Mean and 95% interval',color=gray,fontsize=7.5)
    fig.text(.10,.06,'512 raw outputs per method · no geometry optimization',fontsize=7.5,color='#4c596b')
    for suffix in ['pdf','svg','png']:fig.savefig(a.out/f'matched_connection.{suffix}',dpi=240,facecolor='white')
    (a.out/'manifest.json').write_text(json.dumps(dict(audit=str(a.audit),audit_sha256=sha(a.audit),width_inches=5.5,
        design='Mean circles and two small seed crosses on a zero-based yield axis; paired composition-bootstrap intervals for within-parent improvements. Both GAGA and FM receive identical head-supervision budgets.',
        summary=d['summary'],contrasts=d['contrasts']),indent=2)+'\n');plt.close(fig)


if __name__=='__main__':main()
