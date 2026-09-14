#!/usr/bin/env python3
"""Recompute the fixed composition-only covariance calibration, without an oracle."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cfm_mol.tree_prior_controls import TreePriorControl
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    source = args.project/'research/evidence/tree_covariance_calibration_v1.json'
    report = json.loads(source.read_text())
    manifest = args.project/'research/evidence/development_panel_v1.json'
    conditions = json.loads(manifest.read_text())['rows']
    low = TreePriorControl(**report['configuration'])
    high = TreePriorControl(**report['reference_configuration'])
    assert report['complete'] and len(report['rows']) == len(conditions)
    for row, condition in zip(report['rows'], conditions):
        numbers, charge, spin = condition['atomic_numbers'], condition['charge'], condition['spin_multiplicity']
        a = low.covariance_for(numbers, charge, spin)
        b = high.covariance_for(numbers, charge, spin)
        np.testing.assert_allclose(float((a-b).norm()/b.norm()), row['relative_frobenius_error'], atol=1e-13, rtol=0)
        np.testing.assert_allclose(float(a.trace()/b.trace()-1), row['relative_trace_error'], atol=1e-13, rtol=0)
    write(args.out, dict(complete=True, conditions_replayed=len(conditions),
        source_sha256=sha(source), condition_manifest_sha256=sha(manifest),
        implementation_sha256=sha(args.project/'cfm_mol/tree_prior_controls.py'),
        reference_is_monte_carlo=True, new_molecular_oracle_calls=0))
    print('All composition-only covariance calibration values replayed.', flush=True)


if __name__ == '__main__':
    main()
