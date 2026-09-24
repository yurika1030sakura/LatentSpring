"""Spacious result figures from audited arrays and fit-level contrasts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

INK='#263746';TEAL='#197d78';GRAY='#87939f';BLUE='#4479a3';ORANGE='#c18546'
def clean(ax):
    for x in ['top','right']:ax.spines[x].set_visible(False)
    for x in ['left','bottom']:ax.spines[x].set_color('#bdc5cb');ax.spines[x].set_linewidth(.65)
    ax.tick_params(length=3,color='#aeb8c0');ax.set_axisbelow(True)
def save(fig,out,name):
    for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=360,facecolor='white')
    Image.open(out/(name+'.png')).convert('L').save(out/(name+'_grayscale.png'));plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.project;a.out.mkdir(parents=True,exist_ok=False)
    files=['research/evidence/seed_replication_audit_v2.json','research/evidence/other_baseline_transfer_audit_v1.json','research/evidence/source_head_factorial_v1.json','research/evidence/cross_generator_head_audit_v1.json']
    seed,other,factor,cross=[json.loads((r/f).read_text()) for f in files];s=seed['summary'];o=other['summary'];sa=dict(np.load(r/'research/evidence/seed_replication_audit_v2.npz'));oa=dict(np.load(r/'research/evidence/other_baseline_transfer_audit_v1.npz'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK})
    rows=[('Gaussian FM',o['gaussian_fm']['parent']['graph'],o['gaussian_fm']['parent']['joint'],GRAY),('EDM',o['edm']['parent']['graph'],o['edm']['parent']['joint'],BLUE),('GAGA',s['gaga_parent']['graph_rate'],s['gaga_parent']['joint_rate'],ORANGE),('LatentSpring',s['fm_physical']['graph_rate'],s['fm_physical']['joint_rate'],TEAL)]
    line_styles={'Gaussian FM':':','EDM':'--','GAGA':'-.','LatentSpring':'-'}
    fig=plt.figure(figsize=(5.5,3.10));left=fig.add_axes([.19,.24,.39,.58]);right=fig.add_axes([.73,.24,.25,.58])
    clean(left);clean(right)
    for i,(name,g,j,c) in enumerate(rows):
        y=3-i;left.plot([100*j,100*g],[y,y],color=c,lw=2.5,ls=line_styles[name],alpha=.5)
        left.scatter([100*j],[y],s=28,color=c,zorder=3);left.scatter([100*g],[y],s=32,facecolor='white',edgecolor=c,lw=1.2,zorder=4)
        left.annotate(f'{100*g:.2f}',(100*g,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8.2)
        left.annotate(f'{100*j:.2f}',(100*j,y),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=8.2)
    left.set_yticks(range(4),[v[0] for v in rows[::-1]]);left.set_xlim(0,32);left.set_ylim(-.55,3.6);left.set_xticks([0,10,20,30]);left.set_xlabel('All outputs (%)',labelpad=8);left.tick_params(axis='y',length=0,pad=7);left.spines['left'].set_visible(False);left.grid(axis='x',color='#e7ecef',lw=.5)
    left.set_title('A  Graph and force criteria',loc='left',pad=16,fontweight='bold')
    for name,fam,arm,color in [('Gaussian FM','gaussian_fm',0,GRAY),('EDM','edm',0,BLUE),('GAGA','gaga_parent',None,ORANGE),('LatentSpring','fm_physical',None,TEAL)]:
        if arm is None:
            mi=seed['methods'].index(fam);valid=sa['graph'][:,mi]&sa['success'][:,mi];force=sa['force'][:,mi]
        else:valid=oa[fam+'_graph'][:,arm]&oa[fam+'_success'][:,arm];force=oa[fam+'_force'][:,arm]
        thresholds=np.unique(np.r_[0,force[valid&(force<=10)],10]);yield_=np.array([100*np.mean(valid&(force<=v)) for v in thresholds]);right.step(thresholds,yield_,where='post',color=color,lw=1.4,ls=line_styles[name],label=name)
    right.axvline(5,color='#b6c0c8',ls='--',lw=.8);right.set_xlim(0,10);right.set_ylim(0,30);right.set_xticks([0,5,10]);right.set_yticks([0,10,20,30]);right.set_ylabel('Joint yield (%)',labelpad=7);right.set_xlabel('Force limit (eV/Å)',labelpad=8);right.set_title('B  Force limit',loc='left',pad=16,fontweight='bold');right.grid(axis='y',color='#e7ecef',lw=.5)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',mec=INK,mfc='white',label='Graph validity'),Line2D([0],[0],marker='o',color='none',mec=INK,mfc=INK,label='Joint yield')],loc='lower center',bbox_to_anchor=(.51,.01),ncol=2,frameon=False,fontsize=8.5,handletextpad=.5,columnspacing=2)
    save(fig,a.out,'quality_gap')

    fig=plt.figure(figsize=(5.5,3.05));left=fig.add_axes([.10,.31,.34,.49]);right=fig.add_axes([.62,.31,.35,.49]);clean(left);clean(right)
    for family,color,marker in [('gaussian_fm',GRAY,'s'),('harmonic_fm',TEAL,'o')]:
        vals=np.array([o[family][arm]['joint_by_fit'] for arm in ['parent','transferred']]).T*100
        for row in vals:left.plot([0,1],row,color=color,lw=.7,alpha=.35)
        mean=vals.mean(0);left.plot([0,1],mean,color=color,lw=1.7,marker=marker,markersize=5)
        for x,y in enumerate(mean):left.annotate(f'{y:.2f}',(x,y),xytext=(0,8 if family=='harmonic_fm' else -15),textcoords='offset points',ha='center',color=color,fontsize=8.5)
    left.set_xlim(-.22,1.22);left.set_ylim(0,32);left.set_yticks([0,10,20,30]);left.set_xticks([0,1],['Off','On']);left.set_xlabel('Physical correction',labelpad=8);left.set_ylabel('Joint yield (%)',labelpad=7);left.set_title('A  Four configurations',loc='left',pad=17,fontweight='bold');left.grid(axis='y',color='#e7ecef',lw=.5)
    for y,key,color in [(0,'source_only_vs_neither',GRAY),(1,'both_vs_physical_only',TEAL)]:
        item=factor['contrasts']['joint'][key];lo,hi=np.array(item['crossed_fit_composition_ci95'])*100;mean=item['mean']*100
        right.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=3,lw=1.3,ms=5)
        right.annotate(f'+{mean:.2f}',(mean,y),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9,color=color,fontweight='bold')
    right.axvline(0,color='#b6c0c8',ls='--',lw=.8);right.set_xlim(-1.5,8);right.set_ylim(-.5,1.6);right.set_yticks([0,1],['Off','On']);right.set_xticks([0,4,8]);right.set_xlabel('Source gain (pp)',labelpad=8);right.set_ylabel('Physical correction',labelpad=9);right.set_title('B  Harmonic source gain',loc='left',pad=17,fontweight='bold');right.grid(axis='x',color='#e7ecef',lw=.5)
    fig.legend(handles=[Line2D([0],[0],marker='s',color=GRAY,label='Gaussian source'),Line2D([0],[0],marker='o',color=TEAL,label='Harmonic source')],loc='lower center',bbox_to_anchor=(.50,.015),ncol=2,frameon=False,fontsize=8.5,columnspacing=2)
    save(fig,a.out,'components')

    fig,axes=plt.subplots(1,2,figsize=(5.5,2.95),sharey=True);fit_colors=['#bac2c7','#8e9ba5',BLUE,ORANGE,TEAL];fit_markers=['o','o','o','s','^']
    for ax,family,title in zip(axes,['gaga','fm'],['A  FM head → GAGA','B  GAGA head → FM']):
        clean(ax);values=np.array([s[family+'_parent']['joint_by_fit'],s[family+'_physical']['joint_by_fit'],cross['summary'][family]['joint_by_fit']]).T*100
        for i,row in enumerate(values):ax.plot([0,1,2],row,color=fit_colors[i],lw=1.1 if i<2 else 1.7,ls='--' if i<2 else '-',marker=fit_markers[i],ms=3.3 if i<2 else 4.4,mfc='white' if i<2 else fit_colors[i],alpha=.9)
        ax.set_xticks([0,1,2],['Parent','Own\nhead','Transferred\nhead']);ax.set_xlim(-.4,2.4);ax.set_ylim(0,36);ax.set_yticks([0,10,20,30]);ax.set_title(title,loc='left',pad=17,fontweight='bold');ax.grid(axis='y',color='#e7ecef',lw=.5)
    axes[0].set_ylabel('Joint yield (%)',labelpad=8)
    fig.subplots_adjust(left=.10,right=.975,top=.78,bottom=.27,wspace=.30)
    fig.legend(handles=[Line2D([0],[0],color='#9aa6af',ls='--',marker='o',mfc='white',label='Earlier fits'),*[Line2D([0],[0],color=fit_colors[i],marker=fit_markers[i],label=f'Fit {i+1}') for i in [2,3,4]]],loc='lower center',bbox_to_anchor=(.51,.025),ncol=4,frameon=False,fontsize=8,handlelength=1.4,columnspacing=1.2)
    save(fig,a.out,'head_transfer')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out/'manifest.json').write_text(json.dumps(dict(complete=True,inputs={f:digest(r/f) for f in files+['research/evidence/seed_replication_audit_v2.npz','research/evidence/other_baseline_transfer_audit_v1.npz']},files={p.name:digest(p) for p in a.out.iterdir() if p.is_file()},width_inches=5.5,all_fits_retained=True,primary_comparison_methods=4,threshold_curves='Empirical all-output step functions from raw force arrays',style='Direct labels, distinct panels, increased gaps, no in-panel disclaimer paragraphs.'),indent=2)+'\n')


if __name__=='__main__':main()
