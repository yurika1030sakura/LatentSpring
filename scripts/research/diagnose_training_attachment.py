"""Attribute disconnected TRAIN trajectories without evaluating new molecules."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def connected(adjacency):
    if len(adjacency)<=1:return True
    reached=np.zeros(len(adjacency),bool);reached[0]=True
    for _ in range(len(adjacency)-1):reached|=(adjacency&reached[:,None]).any(0)
    return bool(reached.all())


def classify(x,numbers):
    z=np.array(numbers);r=covalent_radii(numbers).numpy();length=r[:,None]+r[None,:]
    distance=np.linalg.norm(x[:,None]-x[None,:],axis=-1);off=~np.eye(len(x),dtype=bool);contact=(distance<=1.25*length)&off
    heavy=z!=1;hydrogen=~heavy;full=connected(contact);core=connected(contact[np.ix_(heavy,heavy)])
    h_contacts=contact[np.ix_(hydrogen,heavy)].sum(1)
    closest=(distance/length)[np.ix_(hydrogen,heavy)].min(1) if hydrogen.any() and heavy.any() else np.array([])
    return dict(connected=full,heavy_connected=core,only_hydrogen_fragmentation=not full and core,
        heavy_fragmentation=not core,hydrogen_bridge=full and not core,overlap=bool(((distance<.6*length)&off).any()),
        hydrogen_without_heavy=int((h_contacts==0).sum()),hydrogen_multiple_heavy=int((h_contacts>1).sum()),
        isolated_by_element=dict(Counter(str(int(zz)) for zz in z[contact.sum(1)==0])),nearest_h_heavy_scaled=closest.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project.resolve();assert not a.out.exists();torch.set_num_threads(2)
    proto=root/'research/evidence/matched_connection_v1.json';spec=json.loads(proto.read_text());ph=sha(proto);rows=[];sources={}
    for si in [0,1]:
        for name in ['fm','gaga']:
            folder=root/f'runs/matched_connection_v1/validation/s{si}/{name}/teacher';done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
            bankfile=folder/'bank.pt';assert sha(bankfile)==done['bank_sha256'];bank=torch.load(bankfile,map_location='cpu',weights_only=False)
            for slot,index in enumerate(spec['training_rows']):
                file=folder/'records'/f'c{slot}.pt';digest=sha(file);r=torch.load(file,map_location='cpu',weights_only=False)
                assert r['protocol_sha256']==ph and r['training_row']==index and all(v['record_sha256']==digest for v in bank['rows'] if v['composition_slot']==slot)
                sources[str(file.relative_to(root))]=digest;c=r['condition'];frames=[('final',r['final_positions'])]+[(f'provisional_{v["call"]}',v['endpoint']) for v in r['observed']]
                for stage,positions in frames:
                    for j,x in enumerate(positions):rows.append(dict(seed=si,method=name,stage=stage,composition_slot=slot,training_row=index,sample=j,
                        n_atoms=c['n_atoms'],n_hydrogens=c['numbers'].count(1),**classify(x.double().numpy(),c['numbers'])))
    summary={}
    for name in ['fm','gaga']:
        for stage in sorted({r['stage'] for r in rows if r['method']==name}):
            selected=[r for r in rows if r['method']==name and r['stage']==stage];isolated=Counter();nearest=[]
            for r in selected:isolated.update(r['isolated_by_element']);nearest+=r['nearest_h_heavy_scaled']
            counts={key:sum(r[key] for r in selected) for key in ['connected','heavy_connected','only_hydrogen_fragmentation','heavy_fragmentation','hydrogen_bridge','overlap','hydrogen_without_heavy','hydrogen_multiple_heavy']}
            summary[name+'/'+stage]=dict(frames=len(selected),**counts,isolated_by_element=dict(isolated),
                nearest_h_heavy_quantiles=np.quantile(nearest,[.1,.5,.9,.99]).tolist(),hydrogens=sum(r['n_hydrogens'] for r in selected),
                hydrogens_past_contact=int((np.array(nearest)>1.25).sum()),hydrogens_past_1p5=int((np.array(nearest)>1.5).sum()))
    write(a.out,dict(complete=True,protocol_sha256=ph,scope='Only previously cached TRAIN parent trajectories and provisional endpoints; no evaluation-panel coordinates, new generation, training, or oracle queries. Contact thresholds are unchanged. Hydrogen-only fragmentation and heavy fragmentation are structural diagnostics, not assigned chemical bonds.',
        summary=summary,sources=sources,rows=rows,new_neural_outputs=0,new_physical_queries=0));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
