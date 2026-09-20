"""Diagnose valid repaired outputs and choose a molecular example without energy ranking."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.molecular_figure_tools import camera
from scripts.research.train_electronic_fm import sha

ROOT=Path(__file__).resolve().parents[2]
def main():
    protocol=json.loads((ROOT/'research/evidence/hydrogen_physical_confirmation_v1.json').read_text())
    entries=[];provenance={};candidates=[]
    for seed in [0,1]:
        folder=ROOT/f'runs/hydrogen_physical_confirmation_v1/s{seed}/readouts'
        pf=folder/'physical.json';gf=folder/'generation.json'
        provenance[str(pf.relative_to(ROOT))]=sha(pf);provenance[str(gf.relative_to(ROOT))]=sha(gf)
        rows=json.loads(pf.read_text())['rows'];mapping={(r['method'],r['condition_index'],r['sample_index']):r for r in rows}
        generations={(r['method'],r['condition_index']):r for r in json.loads(gf.read_text())['rows']}
        for r in rows:
            if r['method']!='fm_molecule_start0':continue
            i,j=r['condition_index'],r['sample_index'];b=mapping['fm_base',i,j];control=mapping['fm_radial',i,j];c=protocol['test_rows'][i]
            assert r['physical']['result']['success'] and control['physical']['result']['success']
            delta=(r['physical']['result']['energy_eV']-control['physical']['result']['energy_eV'])/len(c['numbers'])
            jointly_valid=r['graph'] and control['graph'];repaired=jointly_valid and not b['graph']
            item=dict(seed=seed,condition_index=i,sample_index=j,composition_hex=c['composition_hex'],natoms=len(c['numbers']),
                jointly_valid=jointly_valid,repaired=repaired,changed=r['coordinate_sha256']!=control['coordinate_sha256'],energy_delta_eV_per_atom=delta)
            entries.append(item)
            if repaired and len(c['numbers'])<=28:
                item=dict(item,selection_hash=hashlib.sha256(f"hydrogen-figure-v1:{seed}:{c['composition_hex']}:{j}".encode()).hexdigest())
                candidates.append((item,folder,generations))
    assert len(entries)==960
    stats={}
    for name,predicate in [('all',lambda r:True),('jointly_valid',lambda r:r['jointly_valid']),('repaired_and_jointly_valid',lambda r:r['repaired'])]:
        subset=[r for r in entries if predicate(r)];stats[name]=dict(count=len(subset),changed=sum(r['changed'] for r in subset),
            mean_energy_delta_eV_per_atom=float(np.mean([r['energy_delta_eV_per_atom'] for r in subset])),
            by_seed=[float(np.mean([r['energy_delta_eV_per_atom'] for r in subset if r['seed']==s])) for s in [0,1]])
    diagnostic=ROOT/'research/evidence/hydrogen_valid_subset_diagnostic_v1.json';assert not diagnostic.exists()
    diagnostic.write_text(json.dumps(dict(complete=True,statistics=stats,rows=entries,provenance=provenance,
        scope='Post-confirmation descriptive subsets, not new selection gates or free-energy estimates.'),indent=2)+'\n')
    item,folder,generation=min(candidates,key=lambda row:row[0]['selection_hash']);i,j=item['condition_index'],item['sample_index']
    records={name:torch.load(folder/generation['fm_'+name,i]['file'],map_location='cpu',weights_only=False) for name in ['base','radial','molecule_start0']}
    c=records['base']['condition'];numbers=np.array(c['numbers']);heavy=numbers!=1;x=records['base']['positions'][j].numpy();rotation=camera(x[heavy]);scenes=[]
    radii=covalent_radii(numbers.tolist(),dtype=torch.float64).numpy();periodic=Chem.GetPeriodicTable()
    for name,record in records.items():
        x=record['positions'][j].numpy();view=(x-x[heavy].mean(0))@rotation
        distance=np.linalg.norm(x[:,None]-x[None,:],axis=-1);threshold=1.25*(radii[:,None]+radii[None,:]);a,b=np.where(np.triu(distance<=threshold,1))
        # All three scenes use distance contacts, never inferred chemical bonds.
        scenes.append(dict(name=name,camera_group='same_output',positions=view.tolist(),symbols=[periodic.GetElementSymbol(int(z)) for z in numbers],
            bonds=[[int(u),int(v),1] for u,v in zip(a,b)],contact_threshold=1.25,
            raw_record=str((folder/generation['fm_'+name,i]['file']).relative_to(ROOT)),raw_record_sha256=sha(folder/generation['fm_'+name,i]['file'])))
    out=ROOT/'research/figures/hydrogen_readout_v1';out.mkdir(parents=True,exist_ok=False)
    (out/'scenes.json').write_text(json.dumps(dict(scenes=scenes,selected=item,condition=c,eligible_count=len(candidates),
        selection_rule='Smallest metadata hash among FM examples with <=28 atoms, invalid parent and graph-valid radial and learned outputs; energy not used.',
        bonds='Distance contacts in every panel; not inferred bond orders.',camera_rotation=rotation.tolist(),geometry_optimized=False),indent=2)+'\n')
    print(json.dumps(dict(statistics=stats,selected=item,figure_directory=str(out)),indent=2))

if __name__=='__main__':main()
