"""Publication comparison using final five-fit geometry and archived baseline scores."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

COLORS=['#87939f','#4479a3','#c18546','#197d78']
STYLES=[':','--','-.','-']


def clean(ax):
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_color('#bdc5cb');ax.spines[side].set_linewidth(.65)
    ax.tick_params(length=3,color='#aeb8c0');ax.set_axisbelow(True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.project;a.out.mkdir(parents=True,exist_ok=False)
    inputs=['research/evidence/geometry_candidate_five_fit_audit_v1.npz','research/evidence/seed_replication_audit_v2.npz','research/evidence/seed_replication_audit_v2.json','research/evidence/other_baseline_transfer_audit_v1.npz','research/evidence/round2_chemical_geometry_records_v1.jsonl.gz','research/evidence/baseline_chemical_geometry_audit_v1.npz']
    final=dict(np.load(r/inputs[0]));seed=dict(np.load(r/inputs[1]));seed_meta=json.loads((r/inputs[2]).read_text());other=dict(np.load(r/inputs[3]));additional=dict(np.load(r/inputs[5]))
    import gzip
    records=[json.loads(line) for line in gzip.decompress((r/inputs[4]).read_bytes()).decode().splitlines()];gaga_geom=np.zeros((5,64,16),dtype=bool)
    for row in records:
        if row['method']=='gaga_parent':gaga_geom[row['fit'],row['condition'],row['sample']]=row['closed_shell_geometry_pass']
    gi=seed_meta['methods'].index('gaga_parent')
    series=[]
    for name,target in [('Gaussian FM','gaussian_fm'),('EDM','edm')]:
        series.append(dict(name=name,graph=other[target+'_graph'][:,0],success=other[target+'_success'][:,0],force=other[target+'_force'][:,0],geometry=additional[target+'_geometry'][:,0]))
    series.extend([dict(name='GAGA',graph=seed['graph'][:,gi],success=seed['success'][:,gi],force=seed['force'][:,gi],geometry=gaga_geom),dict(name='LatentSpring',graph=final['graph'][:,1],success=final['success'][:,1],force=final['force'][:,1],geometry=final['geometry'][:,1])])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'#263746','axes.labelcolor':'#263746'})
    fig=plt.figure(figsize=(5.5,2.75));left=fig.add_axes([.19,.25,.39,.57]);right=fig.add_axes([.73,.25,.25,.57]);clean(left);clean(right);values=[]
    for i,(s,color,style) in enumerate(zip(series,COLORS,STYLES)):
        graph=100*s['graph'].mean();joint=100*(s['graph']&s['success']&(s['force']<=5)).mean();geometry_joint=100*(s['geometry']&s['success']&(s['force']<=5)).mean();y=3-i
        left.plot([joint,graph],[y,y],color=color,lw=2.5,ls=style,alpha=.5);left.scatter([joint],[y],s=28,color=color,zorder=3);left.scatter([graph],[y],s=32,facecolor='white',edgecolor=color,lw=1.2,zorder=4)
        left.annotate(f'{graph:.2f}',(graph,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8.2);left.annotate(f'{joint:.2f}',(joint,y),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=8.2)
        valid=s['geometry']&s['success'];f=s['force'];thresholds=np.unique(np.r_[0,f[valid&(f<=10)],10]);curve=np.array([100*np.mean(valid&(f<=v)) for v in thresholds]);right.step(thresholds,curve,where='post',color=color,ls=style,lw=1.4)
        values.append(dict(method=s['name'],fits=len(s['graph']),attempts=int(s['graph'].size),graph=graph,graph_force=joint,geometry_force=geometry_joint))
    left.set_yticks(range(4),[s['name'] for s in series[::-1]]);left.set_xlim(0,43);left.set_ylim(-.55,3.6);left.set_xticks([0,20,40]);left.set_xlabel('All outputs (%)',labelpad=7);left.tick_params(axis='y',length=0,pad=7);left.spines['left'].set_visible(False);left.grid(axis='x',color='#e7ecef',lw=.5);left.set_title('A  Graph and force criteria',loc='left',pad=15,fontweight='bold')
    right.axvline(5,color='#b6c0c8',ls='--',lw=.8);right.set_xlim(0,10);right.set_ylim(0,12);right.set_xticks([0,5,10]);right.set_yticks([0,5,10]);right.set_ylabel('Geometry-and-force yield (%)',labelpad=7);right.set_xlabel('Force limit (eV/Å)',labelpad=7);right.set_title('B  Geometry',loc='left',pad=15,fontweight='bold');right.grid(axis='y',color='#e7ecef',lw=.5)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',mec='#263746',mfc='white',label='Graph validity'),Line2D([0],[0],marker='o',color='none',mec='#263746',mfc='#263746',label='Graph and force')],loc='lower center',bbox_to_anchor=(.51,.015),ncol=2,frameon=False,fontsize=8.5,handletextpad=.5,columnspacing=2)
    for ext in ['pdf','svg','png']:fig.savefig(a.out/('main_comparison.'+ext),dpi=360,facecolor='white')
    Image.open(a.out/'main_comparison.png').convert('L').save(a.out/'main_comparison_grayscale.png');plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out/'main_manifest.json').write_text(json.dumps(dict(complete=True,inputs={name:sha(r/name) for name in inputs},values=values,files={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},width_inches=5.5,height_inches=2.75),indent=2)+'\n');print(json.dumps(values,indent=2),flush=True)


if __name__=='__main__':main()
