"""Paper-width experimental graphics, with every fit and explicit missing results."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

TEAL='#16857b';SLATE='#84939e';INK='#263844';ORANGE='#c48149';GRID='#e4e9ec'


def style(ax,*,horizontal=False):
    for name in ['top','right']:ax.spines[name].set_visible(False)
    for name in ['bottom','left']:ax.spines[name].set_color('#bdc7ce');ax.spines[name].set_linewidth(.6)
    ax.tick_params(length=2.5,width=.6,color='#a8b5be',labelcolor=INK)
    ax.grid(axis='x' if horizontal else 'y',color=GRID,lw=.55);ax.set_axisbelow(True)


def save(fig,out,name):
    for ext in ['pdf','svg','png']:fig.savefig(out/f'{name}.{ext}',dpi=360,facecolor='white')
    Image.open(out/f'{name}.png').convert('L').save(out/f'{name}_grayscale.png')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.project.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=['research/evidence/seed_replication_audit_v2.json','research/evidence/other_baseline_transfer_audit_v1.json',
           'research/evidence/source_head_factorial_v1.json','research/evidence/combined_transfer_section_v1.json']
    replication,other,factorial,transfer=[json.loads((root/p).read_text()) for p in paths]
    assert all(d['complete'] for d in [replication,other,factorial,transfer])
    s=replication['summary'];t=other['summary']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'axes.titlesize':10})
    rows=[('Gaussian FM',t['gaussian_fm']['parent']),('EDM',t['edm']['parent']),('GAGA',s['gaga_parent']),('LatentSpring',s['fm_physical'])]
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.22));y=np.arange(4)[::-1]
    for mi,(ax,metric,title) in enumerate(zip(axes,['graph','joint'],['Graph validity','Joint yield'])):
        style(ax,horizontal=True)
        for i,((label,values),yy) in enumerate(zip(rows,y)):
            fits=np.array(values[metric+'_by_fit'])*100;mean=fits.mean();color=TEAL if i==3 else SLATE
            ax.barh(yy,mean,height=.43,color=color,alpha=1 if i==3 else .55,edgecolor='none')
            ax.scatter(fits,np.full(len(fits),yy-.27),s=11,facecolor='white',edgecolor=color if i==3 else '#536775',linewidths=.7,zorder=4)
            ax.text(mean+.65,yy,f'{mean:.2f}',va='center',ha='left',fontsize=8,color=TEAL if i==3 else INK,fontweight='bold' if i==3 else 'normal')
        ax.set_xlim(0,35);ax.set_xticks([0,10,20,30]);ax.set_ylim(-.55,3.65)
        ax.set_yticks(y,[r[0] for r in rows] if mi==0 else ['']*4);ax.tick_params(axis='y',length=0)
        ax.set_title(f'{chr(65+mi)}  {title}',loc='left',pad=7,fontweight='bold');ax.set_xlabel('Outputs (%)',labelpad=4)
        ax.spines['left'].set_visible(False)
    fig.subplots_adjust(left=.18,right=.985,bottom=.24,top=.82,wspace=.21)
    fig.text(.18,.025,'Bars: means     Open circles: individual training fits',fontsize=7.2,color='#627580')
    save(fig,out,'main_comparison')

    fig,axes=plt.subplots(1,2,figsize=(5.5,2.60),sharey=True)
    for mi,(ax,metric,title) in enumerate(zip(axes,['graph','joint'],['Graph validity','Joint yield'])):
        style(ax)
        for family,color,label,marker in [('gaussian_fm',SLATE,'Gaussian source','s'),('harmonic_fm',TEAL,'Harmonic source','o')]:
            values=np.array([t[family][arm][metric+'_by_fit'] for arm in ['parent','transferred']]).T*100
            for row in values:ax.plot([0,1],row,color=color,lw=.7,alpha=.5,marker=marker,markersize=3,mfc='white',mew=.6,zorder=2)
            means=values.mean(0);ax.plot([0,1],means,color=color,lw=2,marker=marker,markersize=5,label=label,zorder=3)
            offset=7 if family=='harmonic_fm' else -15
            for xx,yy in enumerate(means):ax.annotate(f'{yy:.2f}',(xx,yy),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=8,color=color,fontweight='bold')
        ax.set_xlim(-.18,1.18);ax.set_ylim(0,35);ax.set_yticks([0,10,20,30]);ax.set_xticks([0,1],['Correction\noff','Correction\non'])
        ax.set_title(f'{chr(65+mi)}  {title}',loc='left',pad=9,fontweight='bold')
    axes[0].set_ylabel('Outputs (%)');fig.subplots_adjust(left=.10,right=.975,bottom=.24,top=.77,wspace=.23)
    handles=[Line2D([0],[0],color=SLATE,lw=2,marker='s',markersize=4,label='Gaussian source'),Line2D([0],[0],color=TEAL,lw=2,marker='o',markersize=4,label='Harmonic source')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.015),ncol=2,frameon=False,fontsize=8,handlelength=1.7)
    fig.text(.10,.018,'Bold lines: means     Thin lines: the two training fits',fontsize=7.2,color='#627580')
    save(fig,out,'source_correction_ablation')

    fig,axes=plt.subplots(2,2,figsize=(5.5,4.48));baseline=transfer['baseline'];finished=transfer['combined_design_results_complete']
    complete=None
    if finished:
        path=root/'runs/joint_design_transfer_v1/audit.json';complete=json.loads(path.read_text());assert complete['complete'];paths.append(str(path.relative_to(root)))
    for mi,(ax,metric,title) in enumerate(zip(axes[0],['graph','joint'],['Both designs: graph validity','Both designs: joint yield'])):
        observed=[100*baseline[target][metric] for target in ['edm','gaga']]
        if finished:observed.extend(100*complete['summary'][target]['both'][metric] for target in ['edm','gaga'])
        upper=max(35,10*np.ceil((max(observed)+4)/10))
        style(ax);ax.set_xlim(-.18,1.25);ax.set_ylim(0,upper);ax.set_yticks(np.arange(0,upper+1,10));ax.set_xticks([0,1],['Original','Both designs'])
        ax.set_title(f'{chr(65+mi)}  {title}',loc='left',fontsize=8.8,fontweight='bold',pad=8)
        for target,color,marker,offset in [('edm',SLATE,'s',-13),('gaga',ORANGE,'o',7)]:
            old=100*baseline[target][metric];ax.scatter([0],[old],s=26,marker=marker,color=color,zorder=3)
            ax.annotate(f'{target.upper()}  {old:.2f}',(0,old),xytext=(7,offset),textcoords='offset points',fontsize=7.5,color=color)
            if finished:
                new=100*complete['summary'][target]['both'][metric];ax.plot([0,1],[old,new],lw=1.5,color=color,marker=marker,markersize=4)
                ax.annotate(f'{new:.2f}',(1,new),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=7.5,color=color)
        if not finished:
            ax.text(.81,.51,'Evaluation\npending',transform=ax.transAxes,ha='center',va='center',fontsize=8,color='#7a8790',linespacing=1.4)
        if mi==0:ax.set_ylabel('Outputs (%)')
    for ai,(ax,family,color) in enumerate(zip(axes[1],['fm','gaga'],[TEAL,ORANGE])):
        style(ax);values=np.array([s[family+'_'+arm]['joint_by_fit'] for arm in ['parent','physical','hydrogen']]).T*100
        for fit,row in enumerate(values):
            col='#9ca8af' if fit<2 else color
            ax.plot([0,1,2],row,color=col,lw=1.05,alpha=.9,marker='o',markersize=3.6,mfc='white' if fit<2 else col,mew=.7)
        ax.set_xticks([0,1,2],['Parent','+ physical','+ H flow']);ax.set_xlim(-.15,2.15);ax.set_ylim(0,40);ax.set_yticks([0,10,20,30,40])
        ax.set_title(f'{chr(67+ai)}  {family.upper()}: repeated training',loc='left',fontsize=8.8,fontweight='bold',pad=8)
        if ai==0:ax.set_ylabel('Joint yield (%)')
    fig.subplots_adjust(left=.105,right=.98,bottom=.11,top=.91,hspace=.61,wspace=.28)
    fig.text(.105,.018,'C–D: open gray = earlier fits; filled color = three new fits',fontsize=7.1,color='#627580')
    save(fig,out,'transfer_and_replication')

    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(complete=True,source_hashes={p:digest(root/p) for p in paths},figure_width_inches=5.5,
        pending_joint_design_outcomes=not finished,all_training_fits_shown=True,zero_based_axes=True,
        no_pending_outcome_markers_or_connecting_lines=not finished,files={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        design='Direct value labels, common semantic colors, full fit visibility, vector text. Paired source/correction plots expose the four configurations; pending outcomes have no data marks.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
