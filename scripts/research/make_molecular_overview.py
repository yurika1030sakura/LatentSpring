#!/usr/bin/env python3
"""Scientific method figure and animation from a recorded, unoptimized flow."""
import io
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from PIL import Image

from cfm_mol.chemical_moves import infer_chemical_graph
from scripts.research.molecular_figure_tools import draw_molecule,camera,bonds_from_graph,ATOM_COLORS,sphere
from scripts.research.train_electronic_fm import sha


def text(fig,x,y,label,**kwargs):
    kwargs.setdefault('fontsize',kwargs.pop('size',9))
    kwargs.setdefault('color','#263344')
    fig.text(x,y,label,ha='center',va='center',**kwargs)


def arrow(fig,x0,y0,x1,y1,color='#81909c',style='-|>'):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,
        arrowstyle=style,mutation_scale=12,linewidth=1.4,color=color))


def box(fig,x,y,w,h,label,color='#eef3f6'):
    fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,boxstyle='round,pad=.008,rounding_size=.012',
        facecolor=color,edgecolor='#d2dce3',linewidth=.8,zorder=-1))
    text(fig,x+w/2,y+h/2,label)


def main():
    root=Path(__file__).resolve().parents[2];torch.set_num_threads(1)
    folder=root/'runs/editorial_review_20260915/trajectory';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['complete'] and sha(folder/'trajectory.pt')==manifest['trajectory_sha256']
    data=torch.load(folder/'trajectory.pt',weights_only=False,map_location='cpu')
    selected=data['selected_sample'];frames=data['positions'][:,selected].numpy();final=data['raw_output'][selected].numpy()
    c=data['condition'];numbers=c['numbers'];rotation=camera(final);graph=infer_chemical_graph(torch.tensor(final),numbers,c['charge'])
    assert {z:numbers.count(z) for z in set(numbers)}=={1:9,6:10,7:3,8:5}
    bonds=bonds_from_graph(graph)
    allviews=np.concatenate([frames@rotation,final[None]@rotation],axis=0)
    width=max(np.ptp(allviews[:,:,0]),np.ptp(allviews[:,:,1]))+1.2
    extent=(-width/2,width/2,-width/2,width/2)
    out=root/'research/figures/molecular_overview_v2';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(8.2,4.3),facecolor='white')
    text(fig,.035,.945,'a',weight='bold',size=12)
    fig.text(.075,.945,'Generation from an atomic composition',ha='left',va='center',weight='bold',fontsize=11,color='#263344')
    # Composition supplies discrete identities; source and flow supply positions.
    iax=fig.add_axes([.025,.57,.12,.25]);iax.set_xlim(0,2);iax.set_ylim(0,2);iax.axis('off')
    for z,x,y,label in [(6,.5,1.4,'C × 10'),(1,1.5,1.4,'H × 9'),(7,.5,.5,'N × 3'),(8,1.5,.5,'O × 5')]:
        iax.imshow(sphere(ATOM_COLORS[z]),origin='lower',extent=(x-.25,x+.25,y-.25,y+.25))
        iax.text(x,y-.40,label,ha='center',fontsize=7.4,color='#263344')
    text(fig,.085,.86,'Input $c$',weight='bold')
    text(fig,.085,.53,'$Q=0$ · $S=1$ (singlet)',size=7.1)
    columns=[.25,.455,.66,.865];stages=[frames[0],frames[11],frames[22],final]
    for index,(xx,x) in enumerate(zip(columns,stages)):
        ax=fig.add_axes([xx-.079,.56,.158,.27])
        draw_molecule(ax,x,numbers,rotation=rotation,bonds=bonds if index==3 else (),
            scaffold=data['auxiliary_tree_edges'][selected] if index==0 else (),extent=extent,radius_scale=1.1)
        text(fig,xx,.85,['Random source $X_0$','$X_t$, $t=0.34$','$X_t$, $t=0.69$','Raw output $X_{\mathrm{gen}}$'][index],size=8.5)
        text(fig,xx,.53,['Latent spring scaffold','Current coordinates','Current coordinates','27 × 3 coordinates (Å)'][index],size=7.4)
    arrow(fig,.15,.70,.165,.70)
    for left,right in zip(columns[:-1],columns[1:]):arrow(fig,left+.080,.70,right-.080,.70,color='#3f8682')
    text(fig,.56,.895,r'Network output: coordinate velocity $v_\theta(X_t,t;c)$',size=8.6,color='#397f7b')
    # Physical targets are a training operation; inference remains the upper row.
    text(fig,.035,.445,'b',weight='bold',size=12)
    fig.text(.075,.445,'Physical feedback during training',ha='left',va='center',weight='bold',fontsize=11,color='#263344')
    teacherfile=root/'runs/thermal_distillation_v1/s0/study/teacher/refined_c2.pt'
    teacher=torch.load(teacherfile,weights_only=False,map_location='cpu');r=teacher['record'];j=int(torch.where(r['eligible'])[0][0])
    anchor=r['anchor'][j].numpy();tc=teacher['condition'];tgraph=infer_chemical_graph(torch.tensor(anchor),tc['numbers'],tc['charge'])
    tax=fig.add_axes([.02,.135,.16,.23]);draw_molecule(tax,anchor,tc['numbers'],bonds=bonds_from_graph(tgraph),forces=r['anchor_force_eV_A'][j].numpy())
    text(fig,.10,.385,'Generated training\nanchor $A$',size=8.2)
    text(fig,.10,.10,'Energy-model forces',size=7.5)
    box(fig,.25,.255,.19,.105,'Physical targets $Y$\nPerturb + force shift','#eef6f3')
    box(fig,.25,.105,.19,.095,'Replay targets $A$','#f2f4f6')
    arrow(fig,.18,.275,.24,.31);arrow(fig,.18,.21,.24,.155)
    box(fig,.515,.255,.185,.105,'Physical student\n'+r'$\theta_{\mathrm{physical}}$','#eef6f3')
    box(fig,.515,.105,.185,.095,'Replay student\n'+r'$\theta_{\mathrm{replay}}$','#f2f4f6')
    arrow(fig,.45,.31,.505,.31);arrow(fig,.45,.155,.505,.155)
    text(fig,.61,.395,'Same FM objective and reference data',size=7.5)
    arrow(fig,.71,.31,.76,.255);arrow(fig,.71,.155,.76,.22)
    box(fig,.775,.16,.19,.15,'One final generator\n'+r'$\theta_{\mathrm{base}}+\theta_{\mathrm{physical}}$'+'\n'+r'$-\,\theta_{\mathrm{replay}}$','#e9f0f7')
    text(fig,.50,.035,'Training learns the physical update; inference returns atomic coordinates with a single model.',size=8)
    for suffix in ['pdf','svg','png']:fig.savefig(out/f'method.{suffix}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    # Actual saved integration steps, without interpolating synthetic atom paths.
    images=[]
    for index,x in enumerate(list(frames)+[final]):
        fig=plt.figure(figsize=(5.4,4.2),facecolor='white');ax=fig.add_axes([.05,.14,.90,.72])
        draw_molecule(ax,x,numbers,rotation=rotation,extent=extent,bonds=bonds if index==33 else (),
            scaffold=data['auxiliary_tree_edges'][selected] if index==0 else (),radius_scale=1.1)
        fig.text(.5,.93,'LatentSpring · recorded coordinate flow',ha='center',fontsize=12,weight='bold',color='#263344')
        label=f'Flow time t = {index/32:.2f}' if index<33 else 'Raw output · terminal noise included'
        fig.text(.5,.085,label,ha='center',fontsize=10,color='#397f7b')
        fig.text(.5,.025,'Generation progress; not a molecular-dynamics trajectory',ha='center',fontsize=8,color='#74808c')
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=125);plt.close(fig);buffer.seek(0)
        images.append(Image.open(buffer).convert('RGB'))
    images[0].save(out/'generation.gif',save_all=True,append_images=images[1:],duration=[650]+[130]*32+[1500],loop=0,optimize=False)
    provenance=dict(trajectory_manifest_sha256=sha(folder/'manifest.json'),trajectory_sha256=sha(folder/'trajectory.pt'),
        teacher_sha256=sha(teacherfile),selected_condition=manifest['condition_index'],selected_sample=manifest['sample_index'],
        source_scaffold_edges_are_not_chemical_bonds=True,raw_coordinates_unchanged=True,
        structural_display_rule=manifest['selection'],camera='Proper PCA rotation plus fixed18/-16 degree viewing rotation, shared across all trajectory frames.',
        outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='provenance.json'})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
