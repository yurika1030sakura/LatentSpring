#!/usr/bin/env python3
"""Decompose existing TRAINING-parent proposal KL into width and direction error.

This uses completed probes and checkpoints only. The teacher is a local vMF
approximation, so the decomposition does not measure the true molecular KL.
"""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.angular_distillation import vmf_teacher_kl
from cfm_mol.local_site_guide import StiffnessSiteGuide
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.normalized_site_guide import physical_site_parameter
from scripts.research.evaluate_chemical_policy import sha


def decompose(teacher, student):
    """KL = same-axis width KL + k_student A(k_teacher) (1 - cos angle)."""
    kt = teacher.norm(dim=1)
    ks = student.norm(dim=1)
    assert (kt > 1e-3).all() and (ks > 1e-3).all()
    cosine = (teacher * student).sum(1) / (kt * ks)
    direction = ks * (1 / torch.tanh(kt) - 1 / kt) * (1 - cosine)
    # Independent scalar log C(k) formula, stable for the observed k > 1e-3.
    def logc(k):
        return torch.log(k) - torch.log(k.new_tensor(2 * torch.pi)) - k - torch.log1p(-torch.exp(-2 * k))
    width = logc(kt) - logc(ks) + (kt - ks) * (1 / torch.tanh(kt) - 1 / kt)
    exact = vmf_teacher_kl(teacher, student)
    torch.testing.assert_close(width + direction, exact, atol=1e-9, rtol=1e-10)
    assert float(width.min()) > -1e-9 and float(direction.min()) > -1e-9
    return dict(teacher_kl=exact, width_kl=width, direction_kl=direction,
                angle_degrees=torch.rad2deg(torch.acos(cosine.clamp(-1, 1))),
                concentration=ks)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.project
    paths = ['runs/angular_curvature_probe_v1/results.json',
             'runs/angular_curvature_probe_v1/audit.json',
             'runs/angular_curvature_probe_v1/trace.pt']
    report = json.loads((root / paths[0]).read_text())
    audit = json.loads((root / paths[1]).read_text())
    assert report['complete'] and audit['complete']
    assert audit['results_sha256'] == sha(root / paths[0])
    assert audit['trace_sha256'] == report['trace_sha256'] == sha(root / paths[2])
    source = torch.load(root / paths[2], map_location='cpu', weights_only=False)
    ids = [d['context'] for d in report['diagnostics']]
    assert ids == list(range(32)) and all(d['full_rank'] for d in report['diagnostics'])
    x = torch.stack([source['states'][i]['positions'] for i in ids])
    bonds = torch.stack([source['states'][i]['graph']['bond_orders'] for i in ids])
    roots = torch.tensor([[row['leaf'], row['anchor']] for row in source['selected']])
    condition = report['condition']
    numbers = torch.tensor(condition['numbers'], dtype=torch.long)
    protocol_path = root / 'research/evidence/stiffness_pilot_protocol_v1.json'
    protocol = json.loads(protocol_path.read_text())
    electronic = torch.tensor([condition['charge'], condition['spin_multiplicity'],
                               protocol['kT_eV']], dtype=torch.float64)
    teacher = torch.tensor([d['fitted_parameter'] for d in report['diagnostics']], dtype=torch.float64)
    masked, _, roles = masked_angular_context(x, roots)
    physical = {f'physical_kappa_{k}': physical_site_parameter(masked, roles, bonds, roots, k)
                for k in [10, 64, 400]}
    rows = []
    for rep in [0, 1]:
        base = Path(f'runs/stiffness_pilot_v1/replica_{rep}')
        rp, cp = base / 'results.json', base / 'model.pt'
        paths.extend([str(rp), str(cp)])
        result = json.loads((root / rp).read_text())
        assert result['complete'] and result['checkpoint_sha256'] == sha(root / cp)
        assert result['protocol_sha256'] == sha(protocol_path)
        ck = torch.load(root / cp, map_location='cpu', weights_only=False)
        assert ck['protocol_sha256'] == sha(protocol_path)
        model = StiffnessSiteGuide(**ck['configuration']).double().eval()
        parameters = dict(physical)
        for stage, key in [('initial', 'initial_state_dict'), ('final', 'state_dict')]:
            model.load_state_dict(ck[key])
            parameters[stage] = model(x, bonds, numbers, electronic, roots)[0][:, 0]
        for label, indices in [('fit', result['fit_context_ids']),
                               ('withheld_training_parents', result['withheld_context_ids'])]:
            assert indices == ck['fit_context_ids' if label == 'fit' else 'withheld_context_ids']
            for name, eta in parameters.items():
                parts = decompose(teacher[indices], eta[indices])
                means = {key: float(value.mean()) for key, value in parts.items()}
                if name in ['initial', 'final']:
                    assert abs(means['teacher_kl'] - result[name][label]['teacher_kl']) < 1e-8
                row = dict(replica=rep, subset=label, model=name, contexts=len(indices),
                           means=means, per_context={key: value.tolist() for key, value in parts.items()},
                           context_ids=indices)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ['per_context', 'context_ids']}), flush=True)
    output = dict(complete=True, rows=rows, new_physical_queries=0, checkpoint_metric_replay=True,
                  independent_scalar_kl_identity_checks=True,
                  input_sha256={p: sha(root / p) for p in paths},
                  protocol_sha256=sha(protocol_path), scientific_submission_ready=False,
                  scope='Post hoc TRAINING-only diagnostic of local vMF surrogate error. No fitting, new oracle calls, transfer-coordinate access or sampling superiority claim.')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
