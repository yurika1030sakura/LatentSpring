#!/usr/bin/env python3
"""Secondary stationarity screen of audited joint-kernel histories, no new oracle."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfm_mol.mcmc_diagnostics import diagnose_chains
from scripts.research.evaluate_chemical_policy import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    rows = []
    for method in ['deterministic', 'uniform', 'site', 'vector', 'tensor']:
        continued = method in ('deterministic', 'uniform', 'site')
        run = 'joint_chemical_full_cost_v1' if continued else 'joint_chemical_v1'
        audit_path = root/'research/evidence'/('joint_chemical_full_cost_audit_v1.json' if continued else 'joint_chemical_audit_v1.json')
        audit = json.loads(audit_path.read_text())
        assert audit['complete']
        for replica in [0, 1]:
            path = root/f'runs/{run}/{method}_s{replica}/results.json'
            data = json.loads(path.read_text())
            audited = [row for row in audit['rows'] if row['method'] == method and row['replica'] == replica]
            assert len(audited) == 1 and data['trace_sha256'] == audited[0]['trace_sha256']
            begin = len(data['history'])//2
            history = data['history'][begin:]
            energy = np.asarray([h['energy_eV'] for h in history]).T
            potential = np.asarray([h['potential_eV'] for h in history]).T
            # The restraint gives gamma/2 * sum_i |x_i|^2, with COM zero.
            rg2 = 2*(potential-energy)/(physical['restraint_eV_A2']*len(data['condition']['numbers']))
            rows.append(dict(method=method, replica=replica, first_retained_step=history[0]['step'],
                last_retained_step=history[-1]['step'], results_sha256=sha(path), audit_sha256=sha(audit_path),
                diagnostics={key: diagnose_chains(value) for key, value in [('raw_energy_eV', energy),
                    ('restrained_potential_eV', potential), ('radius_of_gyration_squared_A2', rg2)]}))
    out = dict(complete=True, analysis='Post hoc diagnostic of the final half of every completed history; no extra burn-in optimization.',
        new_physical_queries=0, rows=rows, scientific_submission_ready=False,
        limitation='Correlated microsteps, four starts and few observables; neither favorable Rhat nor ESS on these histories establishes graph/conformer coverage or physical target accuracy.')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(out, indent=2)+'\n')
    for row in rows:
        print(row['method'], row['replica'], {key: round(value['split_rank_rhat'], 4) for key, value in row['diagnostics'].items()})


if __name__ == '__main__':
    main()
