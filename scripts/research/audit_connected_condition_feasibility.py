#!/usr/bin/env python3
"""Necessary degree-cap check for the CURRENT zero-formal-charge graph builder.

This does not modify frozen panels or the validator. Passing the bound does not
prove that the target domain is nonempty; failing proves incompatibility with
this connected graph criterion, not chemical impossibility of the composition.
"""
import argparse
import hashlib
import json
from pathlib import Path
from rdkit import Chem,rdBase
from rdkit.Chem import rdDetermineBonds

CAPS={1:1,5:4,6:4,7:4,8:3,9:1,14:4,15:5,16:6,17:1,32:4,35:1,53:1}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.project
    assert rdBase.rdkitVersion=='2025.03.2'
    source=root/'runs/literature/20260912/DetermineBonds_2025_03_2.cpp'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='53611667ebb285ed09104f09b6e0c4c3a7d6c9f571cbfb26a8e45c9dee3df9e4'
    panels=['development_panel_v1.json','transfer_development_panel_v1.json','proposal_training_panel_v1.json']
    rows=[];checked={};table=Chem.GetPeriodicTable()
    for panel in panels:
        path=root/'research/evidence'/panel;data=json.loads(path.read_text());assert data['complete']
        for index,row in enumerate(data['rows']):
            numbers=row['atomic_numbers'];caps=[];unknown=[]
            for z in numbers:
                options=[v for v in table.GetValenceList(z) if v>=0]
                cap=CAPS.get(z,max(options) if options else None)
                if cap is None:unknown.append(z)
                caps.append(cap)
                if z in CAPS and z not in checked:
                    mol=Chem.RWMol();atom=Chem.Atom(z);atom.SetNoImplicit(True);mol.AddAtom(atom)
                    for j in range(cap+1):
                        h=Chem.Atom(1);h.SetNoImplicit(True);mol.AddAtom(h);mol.AddBond(0,j+1,Chem.BondType.SINGLE)
                    with rdBase.BlockLogs():
                        try:rdDetermineBonds.DetermineBondOrders(mol,charge=0,allowChargedFragments=True,embedChiral=False)
                        except ValueError as exc:
                            assert 'Valence of atom 0' in str(exc);checked[z]=str(exc)
                        else:raise AssertionError('Runtime permits a degree exceeding the pinned algorithm cap')
            bound=sum(caps) if not unknown else None;required=2*(len(numbers)-1)
            rows.append(dict(panel=panel,index=index,candidate_index=row['candidate_index'],numbers=numbers,
                maximum_degree_sum=bound,minimum_connected_degree_sum=required,unknown_elements=sorted(set(unknown)),
                ruled_out_by_degree_bound=bound is not None and bound<required,
                status='unqualified_valence_bound' if unknown else 'connected_support_impossible' if bound<required else 'not_ruled_out'))
    result=dict(complete=True,rdkit_version=rdBase.rdkitVersion,rows=rows,runtime_degree_cap_failures_checked=checked,
        original_panels_unchanged=True,new_physical_queries=0,
        scope='A connected N-atom graph needs degree sum >=2(N-1). The pinned DetermineBonds degree caps apply before charge assignment because our builder starts all formal charges at zero and disables Hueckel connectivity. No quantum or positive-domain certificate follows from passing.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in rows if r['ruled_out_by_degree_bound']],indent=2))


if __name__=='__main__':main()
