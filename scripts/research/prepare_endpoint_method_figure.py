"""Prepare a true-coordinate method diagram for endpoint physical correction."""
import json
from pathlib import Path
import numpy as np
import torch
from ase.data import chemical_symbols
from cfm_mol import matched_egnn as base
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii
from cfm_mol.tree_prior_controls import cayley_tree,edge_embedding
from scripts.research.molecular_figure_tools import camera,bonds_from_graph
from scripts.research.train_electronic_fm import sha

ROOT=Path(__file__).resolve().parents[2]
def main():
    folder=ROOT/'runs/hydrogen_physical_confirmation_v1/s0/parents/fm';report=json.loads((folder/'generation.json').read_text());selected=None
    for row in sorted(report['rows'],key=lambda r:r['condition_index']):
        file=folder/row['file'];saved=torch.load(file,map_location='cpu',weights_only=False);c=saved['condition']
        if not 17<=len(c['numbers'])<=28:continue
        good=[i for i,r in enumerate(row['records']) if r['graph_supported']]
        if good:selected=(row,file,saved,good[0]);break
    assert selected is not None;row,file,saved,j=selected;c=saved['condition'];numbers=c['numbers'];source=base.HarmonicSource();source.sample(numbers,np.random.default_rng(0))
    # Replay the source RNG only, checking the exact saved float32 coordinates.
    rng=np.random.default_rng(saved['seed']*1000003+row['condition_index']*100003+(j//8)*8);u,scale=source.cache[tuple(numbers)]
    for _ in range(j%8+1):
        edges=cayley_tree(u,rng);std=np.asarray([scale[i,k] for i,k in edges]);x0=torch.from_numpy(edge_embedding(len(numbers),edges)@(std[:,None]*rng.normal(size=(len(numbers)-1,3)))).float().double()
    torch.testing.assert_close(x0,saved['initial_positions'][j],atol=0,rtol=0)
    final=saved['positions'][j].numpy();rotation=camera(final);origin=final.mean(0);scenes=[]
    for name,x,bonds,scaffold in [('source',x0.numpy(),[],edges),('output',final,bonds_from_graph(infer_chemical_graph(torch.tensor(final),numbers,c['charge'])),[])]:
        scenes.append(dict(name=name,camera_group='generation',positions=((x-origin)@rotation).tolist(),symbols=[chemical_symbols[z] for z in numbers],bonds=bonds,scaffold=scaffold))
    teacher=None
    for index in range(128):
        path=ROOT/f'runs/matched_connection_v1/validation/s0/fm/teacher/records/c{index}.pt';record=torch.load(path,map_location='cpu',weights_only=False);z=record['condition']['numbers']
        if 17<=len(z)<=24 and set(z)<=set([1,6,7,8]):teacher=(path,record);break
    assert teacher is not None;path,record=teacher;z=record['condition']['numbers'];n=len(record['observed'][0]['endpoint']);k=n
    h=record['observed'][1]['endpoint'][0].numpy();shift=record['shift'][k].numpy();force=record['centered_even_force'][k].numpy();rotation=camera(h);origin=h.mean(0);radii=covalent_radii(z,dtype=torch.float64).numpy()
    for name,x in [('endpoint',h),('shifted_endpoint',h+shift)]:
        d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);ii,jj=np.where(np.triu(d<=1.25*(radii[:,None]+radii[None,:]),1));view=(x-origin)@rotation
        scene=dict(name=name,camera_group='teacher',positions=view.tolist(),symbols=[chemical_symbols[v] for v in z],bonds=[[int(i),int(j),1] for i,j in zip(ii,jj)])
        if name=='endpoint':
            f=force@rotation;ids=np.argsort(np.linalg.norm(f,axis=-1))[-3:];vscale=.7/max(np.linalg.norm(f,axis=-1).max(),1e-12);scene['vectors']=[[view[i].tolist(),(view[i]+vscale*f[i]).tolist()] for i in ids]
        scenes.append(scene)
    out=ROOT/'research/figures/endpoint_method_v1';out.mkdir(parents=True,exist_ok=False)
    receipt=dict(scenes=scenes,generation_record=str(file.relative_to(ROOT)),generation_record_sha256=sha(file),generation_condition_index=row['condition_index'],generation_sample_index=j,
        generation_numbers=numbers,source_rng_replayed_exactly=True,teacher_record=str(path.relative_to(ROOT)),teacher_record_sha256=sha(path),teacher_local_state=k,
        generation_selection='First condition with17--28 atoms, then first graph-valid output. No energy selection. A CHON-only screen had no eligible output and is not used.',
        teacher_selection='First CHON TRAIN record with17--24 atoms, later provisional endpoint, first trajectory; no validity/energy selection.',
        source_edges='Actual random source tree, reconstructed and checked against stored initial coordinates.',output_edges='Inferred chemical bonds.',teacher_edges='Distance contacts, not chemical bond orders.',
        force_arrows='Three largest recorded force directions with one common display scale. Coordinate shifts themselves are not exaggerated.',geometry_optimized=False)
    (out/'scenes.json').write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k not in ['scenes']})

if __name__=='__main__':main()
