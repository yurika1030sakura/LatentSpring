#!/usr/bin/env python3
"""Calibrate the unchanged structural assay on the frozen panel's raw references."""
import argparse
import json
from pathlib import Path

import torch

from scripts.research.audit_generator_output_support import assess
from scripts.research.tree_prior_fm import geometry_counts
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['references', 'generation-audit', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    references = json.loads(args.references.read_text())
    generated = json.loads(args.generation_audit.read_text())
    assert references['complete'] and generated['complete']
    assert references['panel_sha256'] == generated['panel_sha256']
    torch.set_num_threads(2)
    rows = []
    for row in references['rows']:
        index = row['condition_index']
        c = dict(row['condition'], numbers=row['condition']['atomic_numbers'])
        x = torch.tensor(row['positions'], dtype=torch.float64)[None]
        result = assess(x, c, [0])
        geometry = geometry_counts(x, c['numbers'])
        counts = {f's{study["seed_index"]}/{r["method"]}': r['graph_supported']
                  for study in generated['studies'] for r in study['rows'] if r['condition_index'] == index}
        assert len(counts) == 6
        rows.append(dict(condition_index=index, charge=c['charge'], spin_multiplicity=c['spin_multiplicity'],
            n_atoms=len(c['numbers']), reference_geometry=geometry, reference_assessment=result,
            generated_graph_counts=counts, all_six_generated_arms_zero=not any(counts.values())))
    zero = [row for row in rows if row['all_six_generated_arms_zero']]
    totals = dict(conditions=len(rows), reference_graph_supported=sum(r['reference_assessment']['graph_supported'] for r in rows),
        reference_geometrically_supported=sum(r['reference_assessment']['geometrically_supported'] for r in rows),
        reference_disconnected=sum(r['reference_geometry']['disconnected'] for r in rows),
        reference_overlap=sum(r['reference_geometry']['overlap'] for r in rows),
        reference_validator_errors=sum(r['reference_assessment']['validator_errors'] for r in rows),
        all_six_generated_arms_zero=len(zero),
        reference_graph_supported_in_zero_arms=sum(r['reference_assessment']['graph_supported'] for r in zero),
        reference_geometric_supported_in_zero_arms=sum(r['reference_assessment']['geometrically_supported'] for r in zero))
    write(args.out, dict(complete=True, totals=totals, rows=rows, reference_sha256=sha(args.references), generation_audit_sha256=sha(args.generation_audit),
        new_molecular_oracle_calls=0, scientific_submission_ready=False,
        scope='Post-experiment assay calibration. All generated outcomes and original denominators remain unchanged. No raw reference is supplied to a generator or training.',
        limits=['A raw OMol25 reference can be reactive, distorted or multicomponent; reference rejection alone does not prove a validator bug.',
                'Passing graph perception does not certify electronic spin or physical stability.',
                'Reference-qualified subsets cannot retroactively replace the frozen primary comparison.']))
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
