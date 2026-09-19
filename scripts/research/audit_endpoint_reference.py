"""Independent reference audit for the fresh endpoint integration diagnostic.

Re-evaluate MMFF energies on the EXACT saved SDF geometry, because SDF rounds
coordinates and the previous pilot grid used the unrounded embedding. Retain
the previous fixed basin classifier and map. This changes only the reference
readout, not a teacher, training target, model, or sampling setting.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from rdkit import Chem
from rdkit.Chem import AllChem,rdMolTransforms
from cfm_mol.weighted_endpoints import file_sha256,WeightedEndpointStore
from scripts.research.check_cartesian_endpoint_integration import PANEL,TORSIONS,R


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);result=[];start=time.perf_counter()
    for name,smiles in PANEL:
        path=a.inputs/(name+'_base.sdf');mol=Chem.SDMolSupplier(str(path),removeHs=False)[0]
        conf=mol.GetConformer();base=conf.GetPositions().copy()
        props=AllChem.MMFFGetMoleculeProperties(mol,mmffVariant='MMFF94s');ff=AllChem.MMFFGetMoleculeForceField(mol,props)
        refpath=a.inputs/('landscape_'+name+'.npz')
        with np.load(refpath) as f:labels=f['labels'].copy();oldref=f['pref'].copy();minimum_angles=f['minima'].copy()
        n=512;axis=-np.pi+2*np.pi*np.arange(n)/n;energy=np.empty((n,n))
        for i,x in enumerate(axis):
            for j,y in enumerate(axis):
                for k,pos in enumerate(base):conf.SetAtomPosition(k,pos)
                for tor,v in zip(TORSIONS,[x,y]):rdMolTransforms.SetDihedralRad(conf,*tor,float(v))
                xyz=conf.GetPositions();xyz-=xyz.mean(0)
                energy[i,j]=ff.CalcEnergy(tuple(xyz.ravel()))
        probs={}
        for size in [128,256,512]:
            en=energy[::n//size,::n//size].ravel();lw=-en/(R*300.);w=np.exp(lw-logsumexp(lw))
            ax=-np.pi+2*np.pi*np.arange(size)/size;aa,bb=np.meshgrid(ax,ax,indexing='ij')
            ii=np.floor((aa.ravel()+np.pi)*len(labels)/(2*np.pi)+.5).astype(int)%len(labels)
            jj=np.floor((bb.ravel()+np.pi)*len(labels)/(2*np.pi)+.5).astype(int)%len(labels)
            probs[size]=np.bincount(labels[ii,jj],weights=w,minlength=len(oldref))
        grad=[]
        for phi in minimum_angles:
            values=[]
            for d in range(2):
                energies=[]
                for sign in [-1,1]:
                    pert=phi.copy();pert[d]+=sign*1e-5
                    for k,pos in enumerate(base):conf.SetAtomPosition(k,pos)
                    for tor,v in zip(TORSIONS,pert):rdMolTransforms.SetDihedralRad(conf,*tor,float(v))
                    energies.append(ff.CalcEnergy(tuple(conf.GetPositions().ravel())))
                values.append((energies[1]-energies[0])/2e-5)
            grad.append(float(np.linalg.norm(values)))
        np.savez_compressed(a.out/(name+'.npz'),energy_grid_512=energy,p128=probs[128],p256=probs[256],p512=probs[512])
        row=dict(name=name,reference_probs=probs[512].tolist(),
            TV_256_vs512=float(.5*abs(probs[256]-probs[512]).sum()),
            TV_old_unrounded_reference_vs_saved_SDF=float(.5*abs(oldref-probs[512]).sum()),
            maximum_cached_minimum_angular_gradient_kcal_mol_rad=max(grad),
            reference_oracle_queries=n*n+4*len(minimum_angles),source_sdf_sha256=file_sha256(path))
        result.append(row);print(json.dumps(row),flush=True)
        (a.out/'progress.json').write_text(json.dumps(dict(complete=False,rows=result),indent=2))
    (a.out/'summary.json').write_text(json.dumps(dict(complete=True,rows=result,
        seconds=time.perf_counter()-start,total_reference_queries=sum(r['reference_oracle_queries'] for r in result),
        scope='fixed old basin map, re-evaluated MMFF94s target on exact saved-SDF geometry'),indent=2))


if __name__=='__main__':main()
