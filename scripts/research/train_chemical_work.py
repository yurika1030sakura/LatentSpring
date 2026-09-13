#!/usr/bin/env python3
"""Bounded parent-balanced work learning, with no new physical queries."""
import argparse
import json
import time
from pathlib import Path
import torch
from cfm_mol.chemical_work import PairedChemicalWork, LinearBondWork, informed_log_prob
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def load_data(project, protocol):
    pp = project / protocol['plan']
    assert sha(pp) == protocol['plan_sha256']
    plan = torch.load(pp, map_location='cpu', weights_only=False)
    summary_path = project / protocol['catalogue_summary']
    assert sha(summary_path) == protocol['catalogue_summary_sha256']
    summary = json.loads(summary_path.read_text())
    assert summary['complete'] and summary['all_physical_potentials_and_work_reconstructed']
    groups = []
    for index in [1, 2, 3, 5]:
        directory = project / protocol['catalogue_run'] / f'condition_{index:02d}'
        assert sha(directory/'trace.pt') == summary['sources'][str(index)]['trace_sha256']
        data = torch.load(directory/'trace.pt', map_location='cpu', weights_only=False)
        for source in (s for s in plan['sources'] if s['index'] == index):
            assert source['fit_only']
            rows = [r for r in data['rows'] if r['source_id'] == source['source_id'] and r['valid']]
            assert rows and all(r['parent'] == source['parent'] for r in rows)
            old = [data['states'][r['source_state_id']] for r in rows]
            new = [data['states'][r['candidate_state_id']] for r in rows]
            condition = source['condition']
            group = dict(source_id=source['source_id'], index=index, parent=source['parent'],
                x=torch.stack([r['positions'] for r in old]), y=torch.stack([r['positions'] for r in new]),
                bonds=torch.stack([r['graph']['bond_orders'] for r in old]),
                new_bonds=torch.stack([r['graph']['bond_orders'] for r in new]),
                numbers=torch.tensor(condition['numbers'], dtype=torch.long),
                electronic=torch.tensor([[condition['charge'], condition['spin_multiplicity'], protocol['kT_eV']]]*len(rows), dtype=torch.float64),
                active=torch.tensor([r['action'][:2] for r in rows], dtype=torch.long),
                target=torch.tensor([r['potential_change_eV'] for r in rows], dtype=torch.float64),
                force=torch.tensor([r['linear_prediction_eV'] for r in rows], dtype=torch.float64),
                log_volume=torch.tensor([r['log_volume'] for r in rows], dtype=torch.float64),
                actions=[r['action'] for r in rows])
            # The shared context is valid only if passive geometry/graph agree.
            for j, active in enumerate(group['active']):
                mask = torch.ones(len(group['numbers']), dtype=torch.bool); mask[active] = False
                a, b = group['x'][j, mask], group['y'][j, mask]
                torch.testing.assert_close(a-a.mean(0), b-b.mean(0), atol=1e-10, rtol=0)
                assert torch.equal(group['bonds'][j][mask][:, mask], group['new_bonds'][j][mask][:, mask])
            groups.append(group)
    assert len(groups) == 36 and sum(len(g['target']) for g in groups) == 447
    return groups


def predict(model, group):
    return model(*(group[key] for key in ('x', 'y', 'bonds', 'new_bonds', 'numbers', 'electronic', 'active')))


@torch.no_grad()
def metrics(model, groups, protocol):
    rows = []
    for group in groups:
        pred = predict(model, group) if isinstance(model, torch.nn.Module) else (
            group['force'] if model == 'force' else torch.zeros_like(group['target']))
        assert torch.isfinite(pred).all()
        probability = informed_log_prob(pred, group['log_volume'], protocol['kT_eV'],
            1. if model == 'uniform' else protocol['uniform_fraction']).exp()
        target = group['target']
        rows.append(dict(source_id=group['source_id'], index=group['index'], parent=group['parent'],
            MAE_eV=float((pred-target).abs().mean()),
            expected_selected_work_eV=float(probability @ target),
            probability_downhill=float(probability @ (target < 0).double()),
            available_minimum_work_eV=float(target.min()), predicted_work_eV=pred.tolist(),
            probability=probability.tolist(), target_work_eV=target.tolist(), actions=group['actions']))
    keys = ('MAE_eV', 'expected_selected_work_eV', 'probability_downhill')
    return dict(parent_balanced={key: sum(r[key] for r in rows)/len(rows) for key in keys}, rows=rows,
                scope='Offline proposal-selection diagnostic. No reverse policy or real MH gain is inferred.')


def make_model(variant, protocol):
    if variant == 'linear':
        return LinearBondWork(protocol['elements'], restraint=protocol['restraint_eV_A2']).double()
    return PairedChemicalWork(**protocol['model'], geometry=variant == 'geometry',
                              restraint=protocol['restraint_eV_A2']).double()


def train(project, out, protocol_path, replica):
    protocol = json.loads(protocol_path.read_text()); assert protocol['frozen'] and replica in [0, 1]
    groups = load_data(project, protocol)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    held_ids = set(protocol['diagnostic_parent_source_ids'])
    held = [g for g in groups if g['source_id'] in held_ids]
    fit = [g for g in groups if g['source_id'] not in held_ids]
    assert len(held) == 12 and len(fit) == 24
    write(out/'controls.json', {split: {method: metrics(method, selected, protocol) for method in ('uniform', 'force')}
                               for split, selected in [('diagnostic', held), ('all_fit', groups)]})
    for phase in ('diagnostic', 'full_fit'):
        training = fit if phase == 'diagnostic' else groups
        seed = protocol['seeds'][replica] + (10000 if phase == 'full_fit' else 0)
        for variant in protocol['variants']:
            start = time.monotonic(); torch.manual_seed(seed)
            model = make_model(variant, protocol)
            optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
            generator = torch.Generator().manual_seed(seed+1000)
            directory = out / phase / variant; directory.mkdir(parents=True)
            trace = []
            for step in range(protocol['steps']):
                group = training[int(torch.randint(len(training), (1,), generator=generator))]
                optimizer.zero_grad(set_to_none=True)
                pred = predict(model, group)
                loss = torch.nn.functional.smooth_l1_loss(pred, group['target'], beta=protocol['huber_beta_eV'])
                if not torch.isfinite(loss): raise ValueError('Nonfinite work objective')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                trace.append(dict(step=step, source_id=group['source_id'], loss=float(loss), gradient_norm=float(norm)))
                if step % 100 == 0:
                    print(json.dumps(dict(phase=phase, variant=variant, replica=replica, **trace[-1])), flush=True)
            torch.save(dict(configuration=model.configuration, state_dict=model.state_dict(), variant=variant,
                seed=seed, phase=phase, training_source_ids=[g['source_id'] for g in training],
                protocol_sha256=sha(protocol_path)), directory/'model.pt')
            report = dict(complete=True, phase=phase, variant=variant, replica=replica, seed=seed,
                protocol_sha256=sha(protocol_path), model_sha256=sha(directory/'model.pt'),
                fit=metrics(model, training, protocol), trace=trace, elapsed_seconds=time.monotonic()-start,
                new_physical_queries=0, scientific_submission_ready=False)
            if phase == 'diagnostic': report['diagnostic'] = metrics(model, held, protocol)
            write(directory/'results.json', report)
            print(json.dumps({k: report[k] for k in ('complete', 'phase', 'variant', 'replica', 'elapsed_seconds')}), flush=True)
    write(out/'results.json', dict(complete=True, protocol_sha256=sha(protocol_path), replica=replica,
        models=6, new_physical_queries=0, scientific_submission_ready=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'out', 'protocol'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--replica', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train(args.project, args.out, args.protocol, args.replica)


if __name__ == '__main__': main()
