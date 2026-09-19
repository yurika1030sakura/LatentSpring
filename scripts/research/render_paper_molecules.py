"""Ray-traced molecular artwork from immutable raw coordinates, with vector layout."""
import argparse
import json
from pathlib import Path
import subprocess
import os
import numpy as np


def export(root,out):
    import torch
    from cfm_mol.chemical_moves import infer_chemical_graph
    from scripts.research.molecular_figure_tools import camera,bonds_from_graph
    from scripts.research.train_electronic_fm import sha
    folder=root/'runs/editorial_review_20260915/trajectory'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert sha(folder/'trajectory.pt')==manifest['trajectory_sha256']
    data=torch.load(folder/'trajectory.pt',map_location='cpu',weights_only=False)
    j=data['selected_sample'];frames=data['positions'][:,j].numpy();final=data['raw_output'][j].numpy()
    numbers=data['condition']['numbers'];rot=camera(final)
    width=float(max(np.ptp(np.concatenate([frames,final[None]])@rot,axis=(0,1)))+1.8)
    scenes=[]
    for k,x in enumerate([frames[0],frames[11],frames[22],final]):
        graph=infer_chemical_graph(torch.tensor(x),numbers,data['condition']['charge']) if k==3 else None
        scenes.append(dict(name=f'flow{k}',positions=((x-x.mean(0))@rot).tolist(),numbers=numbers,
            bonds=bonds_from_graph(graph) if graph else [],width=width,
            scaffold=data['auxiliary_tree_edges'][j] if k==0 else []))
    gallery=json.loads((root/'research/figures/raw_molecular_gallery_v1/provenance.json').read_text())
    sample=root/manifest['original_sample'];assert sha(sample)==gallery['sample_sha256']
    raw=torch.load(sample,map_location='cpu',weights_only=False);c=raw['condition']
    for k,record in enumerate(gallery['examples']):
        x=raw['positions'][record['sample_index']].numpy();view=(x-x.mean(0))@camera(x)
        graph=infer_chemical_graph(torch.tensor(x),c['numbers'],c['charge'])
        scenes.append(dict(name=f'gallery{k}',positions=view.tolist(),numbers=c['numbers'],bonds=bonds_from_graph(graph),
            width=float(max(np.ptp(view,axis=0))+1.6),cycles=record['independent_cycles']))
    path=out/'scenes.json';path.write_text(json.dumps(dict(scenes=scenes,trajectory_sha256=sha(folder/'trajectory.pt'),
        gallery_source_sha256=sha(sample),selection=gallery['selection'],
        coordinates='Saved raw coordinates; centering and proper rotations only. No optimization.',new_samples=0),indent=2)+'\n')


def raytrace(out):
    import pymol
    pymol.finish_launching(['pymol','-cqk'])
    from pymol import cmd
    from chempy import Atom,Bond
    from chempy.models import Indexed
    from pymol.cgo import CYLINDER
    elements={1:'H',6:'C',7:'N',8:'O'}
    colors={'C':[.29,.36,.43],'H':[.92,.94,.96],'N':[.24,.47,.78],'O':[.85,.26,.29]}
    data=json.loads((out/'scenes.json').read_text())
    for scene in data['scenes']:
        file=out/(scene['name']+'.png')
        if file.exists():continue
        cmd.reinitialize();cmd.set('max_threads',2);cmd.bg_color('white')
        cmd.set('orthoscopic',1);cmd.set('ray_opaque_background',0);cmd.set('antialias',2)
        cmd.set('ambient',.42);cmd.set('direct',.58);cmd.set('specular',.22);cmd.set('shininess',32)
        cmd.set('ray_shadows',0);cmd.set('ambient_occlusion_mode',2);cmd.set('depth_cue',0)
        cmd.set('sphere_scale',.24);cmd.set('stick_radius',.105);cmd.set('valence',1)
        mol=Indexed()
        for i,(z,pos) in enumerate(zip(scene['numbers'],scene['positions'])):
            a=Atom();a.symbol=elements[z];a.name=f'{elements[z]}{i+1}';a.coord=pos;a.resn='MOL';a.resi='1';mol.atom.append(a)
        for i,j,order in scene['bonds']:
            b=Bond();b.index=[int(i),int(j)];b.order=max(1,int(round(order)));mol.bond.append(b)
        cmd.load_model(mol,'molecule');cmd.hide('everything');cmd.show('spheres','molecule')
        if scene['bonds']:cmd.show('sticks','molecule')
        for element,rgb in colors.items():cmd.set_color('atom_'+element,rgb);cmd.color('atom_'+element,'elem '+element)
        cmd.set('sphere_scale',.19,'elem H')
        if scene.get('scaffold'):
            cylinders=[];xyz=np.array(scene['positions']);color=[.18,.58,.53]
            for i,j,*rest in scene['scaffold']:
                for t in np.arange(.06,.94,.14):
                    p=xyz[i]*(1-t)+xyz[j]*t;q=xyz[i]*(1-t-.07)+xyz[j]*(t+.07)
                    cylinders.extend([CYLINDER,*p,*q,.035,*color,*color])
            cmd.load_cgo(cylinders,'auxiliary_springs')
        # Invisible camera box gives the same scale to every trajectory frame.
        half=scene['width']/2
        for pos in [(-half,-half,-half),(half,half,half)]:cmd.pseudoatom('camera_box',pos=pos)
        cmd.hide('everything','camera_box');cmd.zoom('camera_box',buffer=0,complete=1)
        cmd.png(str(file),width=1200,height=1200,dpi=300,ray=1)
        print(scene['name'],flush=True)


def compose(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    ink='#233746';teal='#167f79';muted='#697b86';orange='#b87b3e'
    def label(fig,x,y,s,size=9,weight='normal',color=ink,ha='left'):
        fig.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va='center')
    def arrow(fig,a,b,color=teal):
        fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=11,lw=1.25,color=color))
    def panel(fig,rect,color):
        fig.add_artist(FancyBboxPatch(rect[:2],rect[2],rect[3],transform=fig.transFigure,
            boxstyle='round,pad=0.004,rounding_size=0.013',facecolor=color,edgecolor='none',zorder=-5))
    def molecule(fig,name,rect):
        ax=fig.add_axes(rect);ax.imshow(Image.open(out/(name+'.png')))
        ax.set_xlim(170,1030);ax.set_ylim(1030,170);ax.axis('off')
    fig=plt.figure(figsize=(7.5,4.25),facecolor='white')
    label(fig,.025,.957,'a',12,'bold');label(fig,.065,.957,'From atomic composition to 3D coordinates',11,'bold')
    panel(fig,(.025,.455,.95,.435),'#f4f7f8')
    label(fig,.051,.832,'INPUT',7.3,'bold',muted)
    label(fig,.051,.759,r'$\mathrm{C_{10}H_9N_3O_5}$',13)
    label(fig,.051,.687,'Atomic identities\nCharge 0 · Singlet',7.5)
    label(fig,.051,.535,'No bond graph',7.5,color=muted)
    centers=[.31,.50,.69,.88]
    for i,c in enumerate(centers):
        molecule(fig,'flow'+str(i),[c-.092,.53,.184,.285])
        label(fig,c,.833,['Harmonic source','Flow time 0.34','Flow time 0.69','Generated molecule'][i],7.6,'bold',ha='center')
        label(fig,c,.504,[r'$X_0$',r'$X_t$',r'$X_t$',r'$X_{\mathrm{gen}}\in\mathbb{R}^{27\times3}$'][i],9,ha='center')
    arrow(fig,(.197,.66),(.211,.66))
    for c in centers[:-1]:arrow(fig,(c+.086,.66),(c+.103,.66))
    label(fig,.59,.433,r'At each step: coordinates + time + composition $\;\longrightarrow\;$ atom-wise velocity',7.5,color=teal,ha='center')
    label(fig,.025,.365,'b',12,'bold');label(fig,.065,.365,'Learning physical corrections during training',11,'bold')
    panel(fig,(.025,.054,.95,.265),'#faf8f4')
    label(fig,.055,.251,'Generated\ntraining anchors',8.6,'bold')
    label(fig,.055,.124,'Energy-model forces',7.1,color=orange)
    for y,top,bottom,col in [(.258,'Nearby force-shifted targets',r'$\theta_{\mathrm{physical}}$','#e7f2ee'),(.126,'Original anchor targets',r'$\theta_{\mathrm{replay}}$','#edf0f3')]:
        panel(fig,(.29,y-.041,.285,.082),col)
        label(fig,.307,y,top,8)
        arrow(fig,(.215,.203),(.278,y),orange)
        arrow(fig,(.581,y),(.599,y))
        label(fig,.639,y,bottom,10,ha='center')
        arrow(fig,(.703,y),(.74,.20))
    label(fig,.824,.263,'ONE FINAL NETWORK',7.5,'bold',teal,ha='center')
    label(fig,.828,.182,r'$\theta_{\mathrm{base}}+\theta_{\mathrm{physical}}-\theta_{\mathrm{replay}}$',8.8,ha='center')
    label(fig,.828,.112,'128 forward passes · No oracle at inference',6.5,color=muted,ha='center')
    for suffix in ['pdf','svg','png']:fig.savefig(out/('method.'+suffix),dpi=300,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)
    data=json.loads((out/'scenes.json').read_text());scenes={s['name']:s for s in data['scenes']}
    fig=plt.figure(figsize=(7.5,2.25),facecolor='white')
    label(fig,.025,.94,'One composition, different connectivities',11,'bold')
    label(fig,.975,.94,r'$\mathrm{C_{10}H_9N_3O_5}$',11,ha='right',color=teal)
    for i in range(4):
        x=.018+i*.247
        panel(fig,(x,.07,.232,.765),'#f4f7f8')
        molecule(fig,'gallery'+str(i),[x+.005,.165,.222,.63])
        label(fig,x+.012,.783,chr(97+i),10,'bold',muted)
        label(fig,x+.116,.112,str(scenes['gallery'+str(i)]['cycles'])+' independent cycles',8,ha='center')
    for suffix in ['pdf','svg','png']:fig.savefig(out/('gallery.'+suffix),dpi=300,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['export','render','compose'],required=True)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=='export':export(args.project,args.out)
    elif args.stage=='render':raytrace(args.out)
    else:compose(args.project,args.out)


if __name__=='__main__':main()
