#!/usr/bin/env python3
"""Molecular structures and audited convergence curves for complete work."""
import io
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from cfm_mol.rotor_work import rotate_methyl
from scripts.research.molecular_figure_tools import draw_molecule,camera
from scripts.research.train_electronic_fm import sha


def rotor_view(ax,rotor,angle=0.):
    positions=np.asarray(rotor['positions']);rotation=camera(positions)
    bonds=[(i,j,float(v)) for i,row in enumerate(rotor['reference_bond_orders']) for j,v in enumerate(row) if j>i and v>0]
    x=rotate_methyl(positions,rotor['carbon'],rotor['anchor'],rotor['hydrogens'],angle)
    draw_molecule(ax,x,rotor['condition']['numbers'],rotation=rotation,bonds=bonds,radius_scale=1.1)
    trail=np.stack([rotate_methyl(positions,rotor['carbon'],rotor['anchor'],rotor['hydrogens'],a)[rotor['hydrogens'][0]] for a in np.linspace(0,1.8*np.pi,50)])@rotation
    ax.plot(trail[:,0],trail[:,1],color='#3e9991',lw=1.,alpha=.65,zorder=12)
    ax.annotate('',xy=trail[-1,:2],xytext=trail[-4,:2],arrowprops=dict(arrowstyle='-|>',color='#3e9991',lw=1.,mutation_scale=8),zorder=13)


def main():
    root=Path(__file__).resolve().parents[2];torch.set_num_threads(1)
    protocol=root/'research/evidence/rotor_work_v1.json';resultfile=root/'research/evidence/rotor_work_results_v1.json';auditfile=root/'research/evidence/rotor_work_audit_v1.json'
    spec=json.loads(protocol.read_text());result=json.loads(resultfile.read_text());audit=json.loads(auditfile.read_text())
    assert audit['complete'] and sha(resultfile)==audit['result_sha256'] and sha(protocol)==audit['protocol_sha256']
    out=root/'research/figures/molecular_rotor_v2';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(7.4,4.5),facecolor='white')
    fig.text(.02,.96,'a',fontsize=12,weight='bold',color='#263344')
    fig.text(.06,.96,'Controlled methyl rotation on three molecules',fontsize=10,weight='bold',color='#263344')
    for i,rotor in enumerate(spec['rotors']):
        ax=fig.add_axes([.11+i*.29,.63,.22,.28]);rotor_view(ax,rotor)
        ax.set_title(f'Rotor {i+1}',fontsize=8.5,pad=1,color='#536277')
    fig.text(.52,.625,'Teal arc: methyl rotation · Other internal coordinates fixed · 300 K target',ha='center',fontsize=7.5,color='#536277')
    axes=[fig.add_axes([.095,.13,.36,.36]),fig.add_axes([.60,.13,.36,.36])]
    methods=[('complete_work','Complete work','#337da6'),('energy_only','Energy only','#d57b65'),
        ('without_jacobian','No volume correction','#9a83b7'),('unweighted','Unweighted','#929ba4')]
    sizes=spec['sample_sizes']
    for method,label,color in methods:
        groups=[[r for study in result['studies'] for r in study['comparisons'] if r['method']==method and r['n']==n and r['amplitude']!=0] for n in sizes]
        axes[0].plot(sizes,[np.mean([r['mean_total_variation'] for r in g]) for g in groups],'-o',color=color,label=label,lw=1.7,ms=3)
        if method!='unweighted':axes[1].plot(sizes,[1000*np.mean([r['free_energy_rmse_eV'] for r in g]) for g in groups],'-o',color=color,lw=1.7,ms=3)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(sizes,[str(n) for n in sizes]);ax.tick_params(labelsize=8)
        ax.set_xlabel('Samples per estimate',fontsize=8.5);ax.grid(alpha=.15)
    axes[0].set_ylabel('Distribution error (TV)',fontsize=8.5);axes[1].set_ylabel('Free-energy RMSE (meV)',fontsize=8.5);axes[1].set_yscale('log')
    axes[0].set_title('b   Recovering angular probabilities',loc='left',fontsize=9,pad=10)
    axes[1].set_title('c   Estimating the normalizer',loc='left',fontsize=9,pad=10)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=7.7,bbox_to_anchor=(.53,-.005))
    for suffix in ['pdf','svg','png']:fig.savefig(out/f'rotor_work.{suffix}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    # The motion is the same prescribed rotor coordinate used by the mechanism test.
    images=[]
    for angle in np.linspace(-np.pi,np.pi,33):
        fig=plt.figure(figsize=(4.8,4.0),facecolor='white');ax=fig.add_axes([.05,.16,.90,.70]);rotor_view(ax,spec['rotors'][0],float(angle))
        fig.text(.5,.93,'Controlled molecular rotor',ha='center',fontsize=12,weight='bold',color='#263344')
        fig.text(.5,.095,f'Methyl angle ϕ = {np.rad2deg(angle):.0f}°',ha='center',fontsize=10,color='#3e9991')
        fig.text(.5,.035,'Prescribed angular scan; not a molecular-dynamics simulation',ha='center',fontsize=7.5,color='#74808c')
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=120);plt.close(fig);buffer.seek(0);images.append(Image.open(buffer).convert('RGB'))
    images[0].save(out/'rotor_scan.gif',save_all=True,append_images=images[1:],duration=130,loop=0,optimize=False)
    record=dict(protocol_sha256=sha(protocol),results_sha256=sha(resultfile),audit_sha256=sha(auditfile),
        aggregation='All three molecules and both nonidentity escorts, unchanged128 repetitions per sample count.',
        geometries='Original declared rotor references. Teal arcs and animation use the exact rotate_methyl coordinate map.',
        new_oracle_queries=0,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'})
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
