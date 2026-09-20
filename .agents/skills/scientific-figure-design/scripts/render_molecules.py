"""Render explicit molecular scenes without guessing bonds or changing geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--style',choices=['soft','outline'],default='outline')
    parser.add_argument('--pixels',type=int,default=1200)
    args=parser.parse_args();scenes=json.loads(args.scenes.read_text())['scenes']
    assert len({s['name'] for s in scenes})==len(scenes)
    args.out.mkdir(parents=True,exist_ok=True)
    import pymol
    pymol.finish_launching(['pymol','-cqk'])
    from pymol import cmd
    from chempy import Atom,Bond
    from chempy.models import Indexed
    from pymol.cgo import CYLINDER,CONE
    colors={'C':[.77,.81,.84],'H':[.96,.97,.98],'N':[.32,.54,.78],'O':[.87,.34,.32],
        'S':[.87,.72,.31],'P':[.91,.59,.30],'F':[.45,.69,.48],'Cl':[.45,.69,.48],
        'B':[.80,.64,.57],'I':[.56,.44,.68]}
    if args.style=='soft':colors['C']=[.42,.49,.55]
    camera_groups={}
    for s in scenes:camera_groups.setdefault(s.get('camera_group',s['name']),[]).append(s)
    receipts=[]
    for group,members in camera_groups.items():
        points=[np.asarray(s['positions'],float) for s in members]
        points.extend(np.asarray(s['vectors'],float).reshape(-1,3) for s in members if s.get('vectors'))
        xyz=np.concatenate(points)
        low=xyz.min(0)-.45;high=xyz.max(0)+.45
        for s in members:
            out=args.out/(s['name']+'.png')
            if out.exists():raise FileExistsError(out)
            x=np.array(s['positions'],float);assert x.ndim==2 and x.shape[1]==3 and np.isfinite(x).all()
            assert len(s['symbols'])==len(x)
            cmd.reinitialize();cmd.set('max_threads',2)
            settings=dict(orthoscopic=1,ray_opaque_background=0,antialias=3,
                depth_cue=0,ray_shadows=0,sphere_quality=4,stick_quality=32,
                sphere_scale=.215,stick_radius=.065,valence=1,
                ambient=.48,direct=.55,specular=.08,shininess=20,
                ray_trace_mode=0,ambient_occlusion_mode=1)
            if args.style=='outline':settings.update(ray_trace_mode=1,ray_trace_gain=0.,specular=0.,ambient_occlusion_mode=0)
            for key,value in settings.items():cmd.set(key,value)
            cmd.bg_color('white');cmd.set_color('outline_ink',[.19,.23,.26]);cmd.set('ray_trace_color','outline_ink')
            cmd.set_color('bond_ink',[.33,.37,.40]);cmd.set('stick_color','bond_ink')
            mol=Indexed()
            for i,(element,pos) in enumerate(zip(s['symbols'],x)):
                atom=Atom();atom.symbol=element;atom.name=f'{element}{i+1}';atom.id=i+1
                atom.coord=pos.tolist();atom.resn='MOL';atom.resi='1';mol.atom.append(atom)
            for i,j,order in s['bonds']:
                assert 0<=i<len(x) and 0<=j<len(x) and i!=j
                bond=Bond();bond.index=[int(i),int(j)];bond.order=max(1,int(round(order)));mol.bond.append(bond)
            cmd.load_model(mol,'molecule');cmd.hide('everything')
            cmd.show('spheres','molecule')
            if s['bonds']:cmd.show('sticks','molecule')
            for element in set(s['symbols']):
                cmd.set_color('element_'+element,colors.get(element,[.65,.61,.68]));cmd.color('element_'+element,'elem '+element)
            cmd.set('sphere_scale',.13,'elem H')
            if s.get('scaffold'):
                primitives=[];col=[.27,.48,.68]
                for i,j,*_ in s['scaffold']:
                    distance=np.linalg.norm(x[j]-x[i]);count=max(3,int(distance/.20))
                    for k in range(count):
                        t=(k+.12)/count;u=(k+.65)/count
                        a=x[i]*(1-t)+x[j]*t;b=x[i]*(1-u)+x[j]*u
                        primitives.extend([CYLINDER,*a,*b,.035,*col,*col])
                cmd.load_cgo(primitives,'auxiliary_springs')
            if s.get('vectors'):
                primitives=[];col=[.76,.43,.20]
                for start,end in s['vectors']:
                    start=np.asarray(start);end=np.asarray(end);base=end*.76+start*.24
                    primitives.extend([CYLINDER,*start,*base,.025,*col,*col])
                    primitives.extend([CONE,*base,*end,.09,0.,*col,*col,1.,0.])
                cmd.load_cgo(primitives,'force_directions')
            for point in [low,high]:cmd.pseudoatom('framing',pos=point.tolist())
            cmd.hide('everything','framing');cmd.zoom('framing',buffer=0,complete=1)
            # PyMOL may sort atom names; restore identity order before checking.
            model=cmd.get_model('molecule');restored=np.empty_like(x)
            identities={f'{element}{i+1}':i for i,element in enumerate(s['symbols'])}
            assert len(model.atom)==len(x)
            for atom in model.atom:
                index=identities[atom.name];assert atom.symbol==s['symbols'][index]
                restored[index]=atom.coord
            error=float(abs(restored-x).max());assert error<1e-5
            view=list(cmd.get_view())
            cmd.png(str(out),width=args.pixels,height=args.pixels,dpi=400,ray=1)
            receipts.append(dict(name=s['name'],camera_group=group,view=view,atoms=len(x),bonds=len(s['bonds']),
                max_pymol_coordinate_roundoff_A=error,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
            print(s['name'],args.style,flush=True)
    receipt=dict(renderer='PyMOL',version=list(cmd.get_version()),style=args.style,settings=settings,
        scene_sha256=hashlib.sha256(args.scenes.read_bytes()).hexdigest(),images=receipts,
        geometry_optimized=False,bonds_inferred_by_renderer=False)
    (args.out/'render_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        # PyMOL's shutdown hooks can otherwise mask a Python failure as exit0.
        import traceback,sys,os
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
