#!/usr/bin/env python3
"""Qualify original reference geometry and freeze a monomer panel before generation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from scripts.research.audit_generator_output_support import assess
from scripts.research.tree_prior_fm import geometry_counts
from scripts.research.evaluate_chemical_policy import write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.audit.exists():
        raise FileExistsError('Use new panel and audit output paths')
    pool = json.loads(args.pool.read_text())
    assert pool['complete'] and pool['role'] == 'prospective_monomer_reference_pool'
    seed = 30891
    decisions, qualified = [], []
    for row in pool['rows']:
        c = dict(row['condition'], numbers=row['condition']['atomic_numbers'])
        x = torch.tensor(row['reference_positions'], dtype=torch.float64)[None]
        assessment = assess(x, c, [0])
        geometry = geometry_counts(x, c['numbers'])
        passed = assessment['graph_supported'] == 1 and assessment['validator_errors'] == 0 and geometry == dict(disconnected=0, overlap=0)
        decision = dict(candidate_index=c['candidate_index'], composition_hex=c['composition_hex'], qualified=passed,
                        geometry=geometry, assessment=assessment)
        decisions.append(decision)
        if passed:
            rank = hashlib.sha256(f'{seed}|{c["composition_hex"]}|{c["source"]}'.encode()).hexdigest()
            qualified.append((rank, row['condition']))
    rows, counts, selected = [], {}, set()
    for lo, hi in [(8, 16), (17, 28), (29, 40)]:
        eligible = [(rank, c) for rank, c in qualified if lo <= c['n_atoms'] <= hi]
        counts[f'{lo}-{hi}'] = len({c['composition_hex'] for _, c in eligible})
        added = 0
        for rank, condition in sorted(eligible):
            if condition['composition_hex'] in selected:
                continue
            c = dict(condition, panel_stratum=[lo, hi, 'neutral', 'singlet'], monomer_selection_sha256=rank)
            rows.append(c)
            selected.add(c['composition_hex'])
            added += 1
            if added == 8:
                break
    enough = len(rows) >= 16 and min(counts.values(), default=0) >= 4
    audit = dict(complete=True, source_pool_sha256=sha(args.pool), decisions=decisions, eligible_by_size=counts,
                 proposed_conditions=len(rows), enough_for_frozen_test=enough, new_molecular_oracle_calls=0)
    write(args.audit, audit)
    if not enough:
        raise ValueError('Insufficient reference-qualified coverage; retain audit and do not generate')
    assert not selected.intersection(pool['excluded_compositions'])
    write(args.out, dict(complete=True, role='new_development', intended_use='prospective_monomer_generation', seed=seed,
        source_pool_sha256=sha(args.pool), qualification_audit_sha256=sha(args.audit), excluded_panels=pool['excluded_panels'],
        excluded_compositions=pool['excluded_compositions'], domain=pool['domain'], eligible_by_size=counts, rows=rows,
        selection='Up to8 minimum metadata-hash ranks per size8-16/17-28/29-40, distinct compositions. Require original reference connected/nonoverlapping and accepted by the unchanged graph assay before querying any model output. At least4 eligible per size and16 total.',
        scope='Neutral singlet organic monomers; scope is narrower than unrestricted OMol25. References qualify the assay and are not model inputs. Full pretraining overlap remains to be audited.',
        reserved_outcomes_allowed=False, new_molecular_oracle_calls=0, scientific_submission_ready=False))
    print(json.dumps(dict(qualified=sum(d['qualified'] for d in decisions), eligible=counts, selected=len(rows))), flush=True)


if __name__ == '__main__':
    main()
