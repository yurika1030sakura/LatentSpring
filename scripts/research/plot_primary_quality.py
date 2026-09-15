#!/usr/bin/env python3
"""Absolute physical quality of the already audited raw primary-model outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[2];evidence=root/'research/evidence'
    arrays={};hashes={}
    for kind,name in [('esen','fresh_physics_esen_audit_v1'),('xtb','fresh_physics_xtb_audit_v1')]:
        report=json.loads((evidence/(name+'.json')).read_text());path=evidence/(name+'.npz')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert report['complete'] and digest==report['arrays_sha256']
        arrays[kind]=np.load(path);hashes[name]=digest
    methods=['gaussian','harmonic_tree','escort_delta','work_delta'];graph=arrays['esen']['graph']
    thresholds=[1.,2.,5.,10.,20.,50.,100.];summary={};contrasts={}
    rng=np.random.default_rng(45203);indices=rng.integers(0,24,size=(20000,24))
    for kind,data in arrays.items():
        summary[kind]={};yields=[]
        success=data['success'] if kind=='xtb' else np.ones_like(graph,dtype=bool)
        for j,method in enumerate(methods):
            valid=graph[:,j]&success[:,j];force=data['force_rms'][:,j]
            grid=[]
            for threshold in thresholds:
                cell=(valid&(force<=threshold)).mean(-1)
                resampled=cell.mean(0)[indices].mean(1)
                grid.append(dict(threshold=threshold,rate=float(cell.mean()),by_seed=cell.mean(1).tolist(),composition_ci95=np.quantile(resampled,[.025,.975]).tolist()))
            summary[kind][method]=dict(attempted=1536,graph_supported=int(graph[:,j].sum()),successful_valid=int(valid.sum()),
                valid_force_quantiles_10_50_90=np.quantile(force[valid],[.1,.5,.9]).tolist(),force_yield=grid)
            yields.append((valid&(force<=5)).mean(-1))
        contrasts[kind]={}
        for name,left,right in [('force_minus_harmonic',2,1),('harmonic_minus_gaussian',1,0),('force_minus_gaussian',2,0)]:
            difference=yields[left]-yields[right];boot=difference.mean(0)[indices].mean(1)
            contrasts[kind][name]=dict(mean=float(difference.mean()),by_seed=difference.mean(1).tolist(),composition_ci95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(complete=True,source_arrays_sha256=hashes,methods=methods,thresholds_eV_A=thresholds,summary=summary,
        joint_yield_at5_contrasts=contrasts,new_generated_outputs=0,new_physical_queries=0,
        scope='Descriptive absolute readout of previously audited24-composition outputs. All attempts remain in joint-yield denominators; median forces use each method own valid subset. Intervals condition on fitted models.')
    (evidence/'primary_absolute_quality_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    out=root/'research/figures/primary_quality_v1';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),sharey=True)
    for ax,kind,title in zip(axes,['esen','xtb'],['eSEN','Independent GFN2-xTB']):
        for method,label,color in zip(methods[:3],['Gaussian source','Harmonic source','Harmonic + physical update'],['#7d8793','#21887b','#c15b38']):
            rows=summary[kind][method]['force_yield']
            ax.plot(thresholds,[100*r['rate'] for r in rows],label=label,color=color,marker='o',ms=3,lw=1.7)
        ax.set_xscale('log');ax.set_xticks(thresholds,[1,2,5,10,20,50,100]);ax.set_ylim(0,57)
        ax.set_title(title);ax.set_xlabel('Force-RMS threshold (eV/Å)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Graph-valid joint yield (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.04))
    fig.tight_layout(rect=(0,0,1,.88))
    for suffix in ['pdf','png']:fig.savefig(out/f'primary_quality.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(xtb_contrasts=contrasts['xtb'],rows={m:summary['xtb'][m] for m in methods[:3]})))


if __name__=='__main__':main()
