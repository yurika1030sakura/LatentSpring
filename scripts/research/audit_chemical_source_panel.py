#!/usr/bin/env python3
"""Census chemical support and exchange coverage without inspecting energy outcomes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph,terminal_exchange_actions
from cfm_mol.geometric_domain import connected_nonoverlapping


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.out.exists():raise FileExistsError(args.out)
    report=dict(complete=False,scope=__doc__,energy_outcomes_used=False,reference_coordinates_loaded=False,
        physical_queries=0,scientific_submission_ready=False,conditions=[])
    for index in range(8):
        source=load_entropy_source(args.source,index,
            protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
            manifest_path=root/'research/evidence/development_panel_v1.json')
        c=source['condition'];radii=covalent_radii(c['numbers'])
        row=dict(index=index,condition=c,source_results_sha256=source['source_results_sha256'],streams={})
        for stream in ['training','development']:
            x=source[stream]['positions'];mask=connected_nonoverlapping(x,radii)
            successes=[];failures=[];identities=Counter();action_counts=Counter();classes=Counter()
            for i in mask.nonzero().flatten().tolist():
                try:
                    graph=infer_chemical_graph(x[i],c['numbers'],c['charge'])
                    actions=terminal_exchange_actions(c['numbers'],graph['bond_orders'])
                    identities[graph['connectivity_smiles']]+=1;action_counts[len(actions)]+=1
                    kinds=sorted(set(tuple(sorted([c['numbers'][a[0]],c['numbers'][a[1]]])) for a in actions))
                    for kind in kinds:classes[str(kind)]+=1
                    successes.append(dict(parent_id=i,connectivity_smiles=graph['connectivity_smiles'],
                        eligible_terminal_exchanges=len(actions),exchange_element_pairs=kinds,
                        inferred_radical_electrons=sum(graph['radical_electrons'])))
                except (ValueError,IndexError,RuntimeError) as exc:
                    failures.append(dict(parent_id=i,reason=str(exc),error_type=type(exc).__name__,
                        classification='bond_assignment_rejected' if isinstance(exc,ValueError) else 'validator_algorithm_error'))
            result=dict(attempted=len(x),geometrically_supported=int(mask.sum()),
                chemically_supported=len(successes),with_exchange=sum(r['eligible_terminal_exchanges']>0 for r in successes),
                connectivity_counts=dict(identities),action_count_histogram=dict(action_counts),
                states_by_exchange_element_pair=dict(classes),supported_parents=successes,
                graph_failures=failures,validator_algorithm_errors=sum(r['classification']=='validator_algorithm_error' for r in failures),
                geometrically_rejected_parent_ids=(~mask).nonzero().flatten().tolist(),
                source_sha256=source['training_sha256' if stream=='training' else 'evaluation_sha256'])
            row['streams'][stream]=result
            print(json.dumps(dict(index=index,stream=stream,**{k:result[k] for k in
                ['attempted','geometrically_supported','chemically_supported','with_exchange','action_count_histogram']})),flush=True)
        report['conditions'].append(row)
    report.update(complete=True,script_sha256=sha(Path(__file__)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
