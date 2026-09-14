#!/usr/bin/env python3
"""Separate fragmented coordinates from overlap and radical-representation failures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from rdkit import Chem,rdBase
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def perceive(x,numbers,charge,allow_charged):
    builder=Chem.RWMol()
    for z in numbers:
        atom=Chem.Atom(int(z));atom.SetNoImplicit(True);builder.AddAtom(atom)
    mol=builder.GetMol();conformer=Chem.Conformer(len(numbers))
    for i,xyz in enumerate(x.tolist()):conformer.SetAtomPosition(i,Point3D(*xyz))
    mol.AddConformer(conformer,assignId=True)
    try:
        with rdBase.BlockLogs():
            rdDetermineBonds.DetermineBonds(mol,charge=charge,covFactor=1.25,allowChargedFragments=allow_charged,embedChiral=False,useVdw=True)
            Chem.SanitizeMol(mol)
        return dict(success=Chem.GetFormalCharge(mol)==charge and len(Chem.GetMolFrags(mol))==1,
            charge=Chem.GetFormalCharge(mol),radical_electrons=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    except (ValueError,RuntimeError,IndexError) as exc:
        return dict(success=False,error_type=type(exc).__name__,reason=str(exc))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    source=a.project/'research/evidence/orbit_pairing_five_method_v1.json';summary=json.loads(source.read_text());assert summary['complete']
    rows=[]
    for row in summary['rows']:
        method,index=row['method'],row['condition_index']
        stem='runs/orbit_pairing_typed_v1/evaluation' if method=='typed_rotation' else 'runs/orbit_pairing_eval_v2/evaluation'
        path=a.project/stem/f'{method}_c{index}.pt';assert sha(path)==row['sample_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False);x=saved['positions'];condition=saved['condition'];numbers=condition['numbers']
        radii=covalent_radii(numbers);distance=torch.cdist(x,x);cut=radii[:,None]+radii[None,:]
        n=len(numbers);off=~torch.eye(n,dtype=torch.bool)
        adjacent=(distance<=1.25*cut)&off;overlap=((distance<.6*cut)&off).any((1,2))
        reached=torch.zeros(x.shape[:2],dtype=torch.bool);reached[:,0]=True
        for _ in range(n-1):reached=reached|(adjacent&reached[:,:,None]).any(1)
        disconnected=~reached.all(1);geometry=~(overlap|disconnected)
        assert int(geometry.sum())==row['geometrically_supported']
        isolated=adjacent.sum(-1)==0
        isolated_by_element={str(z):int(isolated[:,torch.tensor(numbers)==z].sum()) for z in sorted(set(numbers))}
        record=dict(method=method,condition_index=index,attempted=len(x),disconnected=int(disconnected.sum()),overlap=int(overlap.sum()),
            both=int((overlap&disconnected).sum()),geometry_supported=int(geometry.sum()),isolated_atoms_by_element=isolated_by_element,
            legacy_graph_supported=row['graph_supported'],legacy_validator_exceptions=row['validator_errors'],source_sha256=sha(path))
        if condition['charge']==0 and (sum(numbers)-condition['charge'])%2==1:
            secondary=[]
            for i in torch.where(geometry)[0].tolist():
                secondary.append(dict(sample=i,**perceive(x[i],numbers,0,False)))
            record['neutral_odd_electron_radical_diagnostic']=dict(rows=secondary,
                successful_assignments=sum(r['success'] for r in secondary),
                radical_count_distribution=dict(Counter(str(r['radical_electrons']) for r in secondary if r['success'])),
                scope='Alternative neutral radical Lewis representation only; neither radical count nor sanitization certifies the original quantum multiplicity or energy.')
        rows.append(record)
    methyl=torch.tensor([[0.,0.,0.],[1.08,0.,0.],[-.54,.935307,0.],[-.54,-.935307,0.]],dtype=torch.float64)
    controls=dict(idealized_methyl_radical=dict(numbers=[6,1,1,1],charge=0,spin_multiplicity=2,
        legacy=perceive(methyl,[6,1,1,1],0,True),radical_mode=perceive(methyl,[6,1,1,1],0,False)))
    assert not controls['idealized_methyl_radical']['legacy']['success']
    assert controls['idealized_methyl_radical']['radical_mode']['success'] and controls['idealized_methyl_radical']['radical_mode']['radical_electrons']==1
    write(a.out,dict(complete=True,source_summary_sha256=sha(source),rows=rows,controls=controls,new_molecular_oracle_calls=0,
        original_results_modified=False,source='https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html',
        scope='Diagnostic attribution only. Preserve the original graph-readout counts and failed branch decisions; alternative radical perception is not a new-model improvement.'))
    print(json.dumps([r for r in rows if r['method']=='typed_rotation']),flush=True)


if __name__=='__main__':main()
