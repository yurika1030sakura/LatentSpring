#!/usr/bin/env python3
"""Replay saved work predictions, fitting splits, streams and pair reversal."""
import argparse
import json
from pathlib import Path
import torch
from scripts.research.train_chemical_work import load_data, make_model, metrics, predict
from scripts.research.audit_masked_angular import equal, sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'run', 'out', 'protocol'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    groups = load_data(args.project, protocol); diagnostic_ids = set(protocol['diagnostic_parent_source_ids'])
    held = [g for g in groups if g['source_id'] in diagnostic_ids]
    fit = [g for g in groups if g['source_id'] not in diagnostic_ids]
    rows = []; streams = {}; reverse_checks = 0
    for replica in [0, 1]:
        header = json.loads((args.run/f's{replica}/results.json').read_text())
        assert header['complete'] and header['protocol_sha256'] == sha(args.protocol)
        controls = json.loads((args.run/f's{replica}/controls.json').read_text())
        for split, data in [('diagnostic', held), ('all_fit', groups)]:
            for method in ('uniform', 'force'): equal(metrics(method, data, protocol), controls[split][method])
        for phase in ('diagnostic', 'full_fit'):
            training = fit if phase == 'diagnostic' else groups
            for variant in protocol['variants']:
                path = args.run/f's{replica}'/phase/variant
                report = json.loads((path/'results.json').read_text())
                assert report['complete'] and report['protocol_sha256'] == sha(args.protocol)
                assert sha(path/'model.pt') == report['model_sha256']
                saved = torch.load(path/'model.pt', map_location='cpu', weights_only=False)
                assert saved['training_source_ids'] == [g['source_id'] for g in training]
                trace = [row['source_id'] for row in report['trace']]
                assert len(trace) == protocol['steps'] and set(trace) <= set(saved['training_source_ids'])
                key = (replica, phase)
                if key in streams: assert streams[key] == trace
                streams[key] = trace
                model = make_model(variant, protocol); model.load_state_dict(saved['state_dict']); model.eval()
                if variant.endswith('_residual'):
                    spec = protocol['frozen_backbones'][str(replica)][phase]
                    assert saved['frozen_backbone'] == spec and sha(args.project/spec['path']) == spec['sha256']
                    base = torch.load(args.project/spec['path'], map_location='cpu', weights_only=False)
                    assert base['training_source_ids'] == saved['training_source_ids']
                    for key, value in model.backbone.state_dict().items():
                        torch.testing.assert_close(value, base['state_dict'][key], atol=0, rtol=0)
                equal(metrics(model, training, protocol), report['fit'])
                if phase == 'diagnostic': equal(metrics(model, held, protocol), report['diagnostic'])
                with torch.no_grad():
                    for group in groups:
                        value = predict(model, group)
                        backward = model(group['y'], group['x'], group['new_bonds'], group['bonds'],
                                         group['numbers'], group['electronic'], group['active'])
                        torch.testing.assert_close(value, -backward, atol=1e-8, rtol=1e-8)
                        reverse_checks += len(value)
                rows.append(dict(replica=replica, variant=variant, phase=phase, model_sha256=sha(path/'model.pt'),
                    results_sha256=sha(path/'results.json'), fit=report['fit']['parent_balanced'],
                    diagnostic=report.get('diagnostic', {}).get('parent_balanced')))
    if args.out.exists(): raise FileExistsError(args.out)
    write(args.out, dict(complete=True, protocol_sha256=sha(args.protocol), models=rows, controls=controls['diagnostic'],
        exact_metric_replay=True, identical_parent_streams_across_variants=True, pair_reversal_checks=reverse_checks,
        parents=36, valid_edits=447, diagnostic_parents=12, new_physical_queries=0, scientific_submission_ready=False,
        interpretation='Paired-work fitting and diagnostic prediction replay only. Molecular usefulness requires the separate corrected-policy experiment; no novelty or sampler advantage follows from this audit.'))


if __name__ == '__main__': main()
