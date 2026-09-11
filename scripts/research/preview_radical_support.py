#!/usr/bin/env python3
"""Bounded training-only preview of NEW radical-aware support, not sampling."""
import argparse
import hashlib
import json
from pathlib import Path
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.chemical_support_v2 import infer_chemical_graph_v2,ChemicalValidatorError
from cfm_mol.geometric_domain import connected_nonoverlapping


def write(path,r):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=Path(__file__).resolve().parents[2]
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    source=load_entropy_source(args.source,5,protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
        manifest_path=root/'research/evidence/development_panel_v1.json')
    c=source['condition'];x=source['training']['positions']
    indices=connected_nonoverlapping(x,covalent_radii(c['numbers'])).nonzero().flatten().tolist()[:32]
    report=dict(complete=False,scope=__doc__,condition=c,training_source_sha256=source['training_sha256'],
        selected_parent_ids=indices,selection='First32 geometrically supported training parents in source order, without energy outcomes.',
        physical_queries=0,new_target_support=True,reference_coordinates_loaded=False,records=[],
        support_source_sha256=hashlib.sha256((root/'cfm_mol/chemical_support_v2.py').read_bytes()).hexdigest(),
        scientific_submission_ready=False)
    write(args.out,report)
    for i in indices:
        try:
            graph=infer_chemical_graph_v2(x[i],c['numbers'],c['charge'],c['spin_multiplicity'])
            row=dict(parent_id=i,success=True,assignment_mode=graph['assignment_mode'],
                smiles=graph['connectivity_smiles'],radical_electrons=sum(graph['radical_electrons']))
        except (ValueError,ChemicalValidatorError) as exc:
            row=dict(parent_id=i,success=False,error_type=type(exc).__name__,reason=str(exc))
        report['records'].append(row);write(args.out,report);print(json.dumps(row),flush=True)
    report.update(complete=True,successful=sum(r['success'] for r in report['records']))
    write(args.out,report)


if __name__=='__main__':main()
