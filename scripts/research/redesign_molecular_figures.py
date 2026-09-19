"""Publication-size diagrams using the scientific-figure-design skill's renders."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root,out):
    import torch
    from ase.data import chemical_symbols
    from cfm_mol.chemical_moves import infer_chemical_graph
    from scripts.research.molecular_figure_tools import camera,bonds_from_graph
    old=root/'research/figures/pymol_molecules_v1/scenes.json'
    d=json.loads(old.read_text());scenes=[]
    for row in d['scenes']:
        if row['name'] in ['flow1','flow2']:continue
        s=dict(row);s['symbols']=[chemical_symbols[n] for n in row['numbers']]
        if row['name'].startswith('flow'):s['camera_group']='source_output'
        scenes.append(s)
    teacher=root/'runs/thermal_distillation_v1/s0/study/teacher/refined_c2.pt'
    saved=torch.load(teacher,map_location='cpu',weights_only=False);r=saved['record'];c=saved['condition']
    i=int(torch.where(r['eligible'])[0][0]);j=int(torch.where(r['valid'][i])[0][0])
    a=r['anchor'][i].numpy();y=r['proposal'][i,j].numpy();rotation=camera(a);origin=a.mean(0)
    force=r['anchor_force_eV_A'][i].numpy()@rotation
    for name,x in [('anchor',a),('target',y)]:
        view=(x-origin)@rotation
        graph=infer_chemical_graph(torch.tensor(x),c['numbers'],c['charge'])
        s=dict(name=name,positions=view.tolist(),numbers=c['numbers'],symbols=[chemical_symbols[n] for n in c['numbers']],
            bonds=bonds_from_graph(graph),camera_group='physical_target')
        if name=='anchor':
            idx=np.argsort(np.linalg.norm(force,axis=-1))[-3:];scale=.8/np.linalg.norm(force,axis=-1).max()
            s['vectors']=[[view[k].tolist(),(view[k]+force[k]*scale).tolist()] for k in idx]
        scenes.append(s)
    record=dict(scenes=scenes,source_scene_sha256=digest(old),teacher_sha256=digest(teacher),
        teacher_anchor=i,teacher_particle=j,teacher_selection='First eligible anchor and first valid candidate, no energy ranking.',
        gallery_selection=d['selection'],coordinates='Original unoptimized coordinates; rigid viewing transformations only.',
        force_arrows='Three largest force vectors, a common display scale; not displacement magnitudes.')
    (out/'scenes.json').write_text(json.dumps(record,indent=2)+'\n')


def compose(out,style):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
    from matplotlib.lines import Line2D
    from PIL import Image,ImageOps,ImageDraw
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','mathtext.fontset':'stixsans'})
    ink='#26323b';blue='#466d91';orange='#b47540';grey='#64727d'
    renders=out/style;data=json.loads((out/'scenes.json').read_text());groups={}
    for s in data['scenes']:groups.setdefault(s.get('camera_group',s['name']),[]).append(s['name'])
    images={};crops={}
    for group,names in groups.items():
        bounds=[]
        for name in names:
            im=Image.open(renders/(name+'.png')).convert('RGBA');images[name]=im
            box=im.getchannel('A').point(lambda x:255 if x>12 else 0).getbbox();assert box is not None
            bounds.append(box)
        low=np.min(np.array(bounds)[:,:2],axis=0);high=np.max(np.array(bounds)[:,2:],axis=0)
        center=(low+high)/2;side=max(high-low)*1.12
        for name in names:crops[name]=[center[0]-side/2,center[0]+side/2,center[1]+side/2,center[1]-side/2]
    label_records=[]
    def label(fig,x,y,text,size=8,weight='normal',color=ink,ha='left'):
        fig.text(x,y,text,fontsize=size,fontweight=weight,color=color,ha=ha,va='center')
        label_records.append(dict(text=text,font_pt=size))
    def arrow(fig,a,b,color=grey,lw=.85,curve=0):
        fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',
            mutation_scale=8,linewidth=lw,color=color,connectionstyle=f'arc3,rad={curve}'))
    def box(fig,x,y,w,h,text,color=blue):
        fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,
            boxstyle='round,pad=.003,rounding_size=.008',fc='white',ec=color,lw=.85,zorder=-1))
        label(fig,x+w/2,y+h/2,text,7.6,color=color,ha='center')
    def molecule(fig,name,rect):
        ax=fig.add_axes(rect);ax.imshow(images[name]);ax.set_xlim(crops[name][:2]);ax.set_ylim(crops[name][2:]);ax.axis('off')
    fig=plt.figure(figsize=(5.5,3.18),facecolor='white')
    label(fig,.014,.963,'a',10,'bold');label(fig,.053,.963,'Generation from atomic composition',10,'bold')
    label(fig,.985,.963,r'$\mathrm{C_{10}H_9N_3O_5}$',10,color=blue,ha='right')
    label(fig,.156,.875,'Latent spring source',8.1,'bold',ha='center')
    label(fig,.505,.875,'Self-conditioned flow',8.1,'bold',ha='center')
    label(fig,.863,.875,'Raw coordinates',8.1,'bold',ha='center')
    molecule(fig,'flow0',[.015,.535,.275,.30]);molecule(fig,'flow3',[.735,.525,.25,.31])
    label(fig,.15,.510,r'$X_0\sim q_0(\,\cdot\mid c)$',8.2,ha='center')
    label(fig,.863,.510,r'$X_{\mathrm{gen}}\in\mathbb{R}^{27\times3}$',8.2,ha='center')
    box(fig,.383,.728,.25,.092,'Predict geometry')
    box(fig,.383,.565,.25,.092,'Refine velocity')
    label(fig,.507,.510,'Shared network',7.5,ha='center')
    arrow(fig,(.286,.69),(.375,.771),blue)
    label(fig,.321,.80,r'$X_t,t,c$',8,color=blue,ha='center')
    arrow(fig,(.507,.725),(.507,.66),blue)
    label(fig,.532,.692,r'$\widehat X_1$',8.5,color=blue)
    arrow(fig,(.64,.61),(.737,.688),blue)
    label(fig,.697,.765,'Integrate',7.6,color=blue,ha='center')
    label(fig,.697,.71,r'$v_\theta$',9,color=blue,ha='center')
    fig.add_artist(Line2D([.015,.985],[.458,.458],transform=fig.transFigure,color='#d7dfe4',lw=.65))
    label(fig,.014,.419,'b',10,'bold');label(fig,.053,.419,'Force-shifted targets',9,'bold')
    label(fig,.535,.419,'c',10,'bold');label(fig,.576,.419,'Paired fine-tuning',9,'bold')
    molecule(fig,'anchor',[.005,.142,.20,.228]);molecule(fig,'target',[.28,.142,.20,.228])
    label(fig,.101,.135,'Anchor $A$',7.8,ha='center')
    label(fig,.381,.135,'Target $Y$',7.8,ha='center')
    arrow(fig,(.208,.245),(.276,.245),orange)
    label(fig,.243,.325,'Perturb\n+ shift',7.5,color=orange,ha='center')
    label(fig,.242,.053,r'$\delta_A=\sigma_A^2 F_+(A)/(k_BT)$',8.3,color=orange,ha='center')
    label(fig,.761,.345,r'Start both from $\theta_{\mathrm{base}}$',7.5,ha='center')
    arrow(fig,(.722,.321),(.617,.273),blue);arrow(fig,(.797,.321),(.896,.273),grey)
    box(fig,.546,.147,.145,.114,'Fit to $Y$\n'+r'$\theta_{\mathrm{physical}}$',blue)
    box(fig,.816,.147,.155,.114,'Fit to $A$\n'+r'$\theta_{\mathrm{replay}}$',grey)
    arrow(fig,(.620,.139),(.719,.079),blue);arrow(fig,(.886,.139),(.803,.079),grey)
    label(fig,.755,.042,r'$\theta_{\mathrm{final}}=\theta_{\mathrm{base}}+(\theta_{\mathrm{physical}}-\theta_{\mathrm{replay}})$',8,ha='center')
    for suffix in ['pdf','svg','png']:fig.savefig(out/f'method_{style}.{suffix}',dpi=400)
    plt.close(fig)
    fig=plt.figure(figsize=(5.5,1.63),facecolor='white')
    label(fig,.018,.938,'One composition, distinct connectivities',9,'bold')
    label(fig,.985,.938,r'$\mathrm{C_{10}H_9N_3O_5}$',9,color=blue,ha='right')
    scene_lookup={s['name']:s for s in data['scenes']}
    for i in range(4):
        x=.01+i*.249
        molecule(fig,'gallery'+str(i),[x,.135,.239,.685])
        label(fig,x+.015,.80,chr(97+i),8.5,'bold',color=grey)
        label(fig,x+.119,.092,str(scene_lookup['gallery'+str(i)]['cycles'])+' independent cycles',7.5,ha='center')
    for suffix in ['pdf','svg','png']:fig.savefig(out/f'gallery_{style}.{suffix}',dpi=400)
    plt.close(fig)
    for f in out.glob(f'*_{style}.svg'):f.write_text('\n'.join(s.rstrip() for s in f.read_text().splitlines())+'\n')
    # A viewing aid at the target physical width, not a substitute for inspecting the PDF.
    im=Image.open(out/f'method_{style}.png').convert('RGB')
    ImageOps.grayscale(im).save(out/f'method_{style}_grayscale.png')
    record=dict(style=style,canvas_width_inches=5.5,placed_width_inches=5.5,
        minimum_label_font_pt=min(r['font_pt'] for r in label_records),text_labels=label_records,
        all_molecular_sources_unchanged=True,related_scene_crop_is_shared=True,
        render_manifest_sha256=digest(renders/'render_manifest.json'),scene_sha256=digest(out/'scenes.json'))
    (out/f'layout_{style}.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','compose'],required=True)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--style',choices=['soft','outline'],default='outline');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare(a.project,a.out)
    else:compose(a.out,a.style)
