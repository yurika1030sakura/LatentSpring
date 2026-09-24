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
    files=['research/evidence/geometric_correction_transfer_audit_v1.json','research/evidence/organic_geometry_audit_v1.json']
    factor,organic=[json.loads((r/f).read_text()) for f in files];o=factor['summary']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK})
    fig=plt.figure(figsize=(5.5,3.05));left=fig.add_axes([.10,.31,.34,.49]);right=fig.add_axes([.62,.31,.35,.49]);clean(left);clean(right)
    for family,color,marker in [('gaussian_fm',GRAY,'s'),('harmonic_fm',TEAL,'o')]:
        vals=np.array([o[family][arm]['graph_force']['by_fit'] for arm in ['parent','geometric_physical']]).T*100
        for row in vals:left.plot([0,1],row,color=color,lw=.7,alpha=.35)
        mean=vals.mean(0);left.plot([0,1],mean,color=color,lw=1.7,marker=marker,markersize=5)
        for x,y in enumerate(mean):left.annotate(f'{y:.2f}',(x,y),xytext=(0,8 if family=='harmonic_fm' else -15),textcoords='offset points',ha='center',color=color,fontsize=8.5)
    left.set_xlim(-.22,1.22);left.set_ylim(0,38);left.set_yticks([0,10,20,30]);left.set_xticks([0,1],['Off','On']);left.set_xlabel('Endpoint corrections',labelpad=8);left.set_ylabel('Joint yield (%)',labelpad=7);left.set_title('A  Four configurations',loc='left',pad=17,fontweight='bold');left.grid(axis='y',color='#e7ecef',lw=.5)
    for y,key,color in [(0,'source_only',GRAY),(1,'source_with_correction',TEAL)]:
        item=factor['source_contrasts']['graph_force'][key];lo,hi=np.array(item['crossed_fit_composition_ci95'])*100;mean=item['mean']*100
        right.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=3,lw=1.3,ms=5)
        right.annotate(f'+{mean:.2f}',(mean,y),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9,color=color,fontweight='bold')
    right.axvline(0,color='#b6c0c8',ls='--',lw=.8);right.set_xlim(-1.5,8);right.set_ylim(-.5,1.6);right.set_yticks([0,1],['Off','On']);right.set_xticks([0,4,8]);right.set_xlabel('Source gain (pp)',labelpad=8);right.set_ylabel('Endpoint corrections',labelpad=9);right.set_title('B  Harmonic source gain',loc='left',pad=17,fontweight='bold');right.grid(axis='x',color='#e7ecef',lw=.5)
    fig.legend(handles=[Line2D([0],[0],marker='s',color=GRAY,label='Gaussian source'),Line2D([0],[0],marker='o',color=TEAL,label='Harmonic source')],loc='lower center',bbox_to_anchor=(.50,.015),ncol=2,frameon=False,fontsize=8.5,columnspacing=2)
    save(fig,a.out,'components')

    fig,axes=plt.subplots(1,2,figsize=(5.5,2.65))
    for ax,metric,title,lim in zip(axes,['graph_force','geometry_force'],['A  Graph and force','B  Geometry and force'],[38,14]):
        clean(ax)
        for target,color,marker in [('edm',BLUE,'s'),('gaga',ORANGE,'o')]:
            values=np.array([o[target][arm][metric]['by_fit'] for arm in ['parent','geometric_physical']]).T*100
            for row in values:ax.plot([0,1],row,color=color,alpha=.3,lw=.7)
            means=values.mean(0);ax.plot([0,1],means,color=color,marker=marker,lw=1.5,ms=5)
            for j,v in enumerate(means):ax.annotate(f'{v:.2f}',(j,v),xytext=(0,9 if target=='gaga' else -15),textcoords='offset points',ha='center',color=color,fontsize=8.2)
        ax.set_xticks([0,1],['Parent','Both corrections']);ax.set_xlim(-.3,1.45);ax.set_ylim(0,lim);ax.set_ylabel('All outputs (%)');ax.set_title(title,loc='left',pad=16,fontweight='bold');ax.grid(axis='y',color='#e7ecef',lw=.5)
    fig.subplots_adjust(left=.10,right=.98,top=.79,bottom=.28,wspace=.35)
    fig.legend(handles=[Line2D([0],[0],color=BLUE,marker='s',label='EDM'),Line2D([0],[0],color=ORANGE,marker='o',label='GAGA')],loc='lower center',bbox_to_anchor=(.51,.01),ncol=2,frameon=False)
    save(fig,a.out,'geometric_transfer')
    fig,ax=plt.subplots(figsize=(5.5,2.7));clean(ax)
    keys=['gaussian_fm','edm','gaga','original_fm_physical','full_geometry_physics'];labels=['Gaussian FM','EDM','GAGA','Initial FM + physics','LatentSpring']
    colors=[GRAY,BLUE,ORANGE,'#87b5ae',TEAL]
    for i,(key,color) in enumerate(zip(keys,colors)):
        row=organic['summary'][key];v=row['geometry_force']['rate']*100;u=row['uncharged_geometry_force']['rate']*100;y=4-i
        ax.barh(y,v,height=.53,color=color,alpha=.24);ax.barh(y,u,height=.53,color=color)
        ax.text(v+.7,y,f'{v:.2f}',va='center',fontsize=8.5);ax.text(u/2,y,f'{u:.2f}',va='center',ha='center',fontsize=8,color='white')
    ax.set_yticks(range(5),labels[::-1]);ax.set_xlim(0,43);ax.set_xlabel('Geometry-and-force yield (%)');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.grid(axis='x',color='#e7ecef',lw=.5)
    ax.set_title('Common-element organic compositions',loc='left',pad=16,fontweight='bold')
    fig.subplots_adjust(left=.28,right=.98,top=.82,bottom=.22);save(fig,a.out,'organic_geometry')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out/'manifest.json').write_text(json.dumps(dict(complete=True,inputs={f:digest(r/f) for f in files},files={p.name:digest(p) for p in a.out.iterdir() if p.is_file()},width_inches=5.5,all_fits_retained=True),indent=2)+'\n')


if __name__=='__main__':main()
