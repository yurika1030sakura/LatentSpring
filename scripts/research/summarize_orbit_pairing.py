#!/usr/bin/env python3
"""Export the audited main-generator pairing comparison without outcome filtering."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','run','audit','out','csv']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() or args.csv.exists():raise FileExistsError('Preserve older summaries')
    protocol=json.loads(args.protocol.read_text())
    result=json.loads((args.run/'results.json').read_text())
    audit=json.loads(args.audit.read_text())
    assert result['complete'] and audit['complete']
    assert result['protocol_sha256']==audit['protocol_sha256']==sha(args.protocol)
    assert audit['source_results_sha256']==sha(args.run/'results.json')
    assert audit['all_structural_outcomes_replayed']==4*len(protocol['conditions'])*protocol['samples_per_condition']
    flat=[];methods={}
    for method in ['warm']+protocol['methods']:
        rows=[r for r in result['rows'] if r['method']==method]
        assert sorted(r['condition_index'] for r in rows)==protocol['conditions']
        attempted=sum(r['attempted'] for r in rows)
        supported=sum(r['graph_supported'] for r in rows)
        assert supported==audit['graph_supported_totals'][method]
        methods[method]=dict(attempted=attempted,graph_supported=supported,
            graph_support_fraction=supported/attempted,geometrically_supported=sum(r['geometrically_supported'] for r in rows),
            validator_errors=sum(r['validator_errors'] for r in rows),
            mean_distinct_connectivities_per_condition=float(np.mean([r['distinct_connectivity'] for r in rows])),
            generation_seconds=sum(r['generation_seconds'] for r in rows),
            additional_training_seconds=0. if method=='warm' else audit['training'][method]['seconds'])
        for row in rows:
            c=row['condition']
            flat.append(dict(method=method,condition=row['condition_index'],n_atoms=c['n_atoms'],charge=c['charge'],spin=c['spin_multiplicity'],
                attempted=row['attempted'],graph_supported=row['graph_supported'],geometrically_supported=row['geometrically_supported'],
                distinct_connectivity=row['distinct_connectivity'],validator_errors=row['validator_errors'],
                radius_of_gyration_mean_A=row['radius_of_gyration_mean_A'],generation_seconds=row['generation_seconds']))
    write(args.out,dict(complete=True,protocol_sha256=sha(args.protocol),source_results_sha256=sha(args.run/'results.json'),
        audit_sha256=sha(args.audit),methods=methods,rows=flat,comparisons=audit['comparisons'],training=audit['training'],
        proceed_to_frozen_energy_check=audit['proceed_to_frozen_energy_check'],new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='Single-seed, same-data FM continuation and fresh fixed-composition molecular generation. All8 conditions and2048 attempts retained. Algorithmic chemical support and observed diversity do not establish Boltzmann sampling or quantum validity.',
        limits=['Conditional bootstrap over Gaussian/noise starts is not training-seed replication.','Frozen warm pretraining is shared historical cost, not free model creation.','The prospective energy gate is an operational screen, not a significance or ICLR-novelty criterion.','Kabsch pairing is an existing baseline; a positive steric increment would still need replication and a prior-art-aware contribution assessment.']))
    args.csv.parent.mkdir(parents=True,exist_ok=True)
    with args.csv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(methods=methods,comparisons=audit['comparisons'],energy_gate=audit['proceed_to_frozen_energy_check'])),flush=True)


if __name__=='__main__':main()
