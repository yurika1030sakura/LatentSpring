#!/usr/bin/env python3
"""Continue the frozen comparison per composition, retaining completed draws."""
import argparse
import json
from pathlib import Path

import torch
from flowmol.model_utils.load import read_config_file

from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import restore_model, evaluate, geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def link(source, target):
    if target.exists():
        assert sha(target) == sha(source)
    else:
        target.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'previous', 'previous-log', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    ph = sha(args.protocol)
    assert spec['frozen']
    torch.set_num_threads(2)
    config = args.project/spec['config']
    manifest = args.project/spec['condition_manifest']
    assert sha(config) == spec['config_sha256'] and sha(manifest) == spec['condition_manifest_sha256']
    overlap_path = args.project/spec['overlap_audit']
    assert sha(overlap_path) == spec['overlap_audit_sha256']
    overlap = json.loads(overlap_path.read_text())
    assert overlap['complete'] and all(r['overlapping_compositions'] == 0 for r in overlap['corpora'].values())
    panel = json.loads(manifest.read_text())['rows']
    cfg = read_config_file(config)
    cfg['mol_fm'].pop('bgfm', None)
    args.out.mkdir(parents=True, exist_ok=True)
    timings = {}
    for line in args.previous_log.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and all(k in record for k in ['method', 'condition_index', 'generation_seconds']):
            timings[(record['method'], record['condition_index'])] = record['generation_seconds']
    totals = dict(reused_previous=0, reused_continuation=0, newly_generated=0, timing_missing=0)
    for method, reference in spec['frozen_references'].items():
        checkpoint = args.project/reference['checkpoint']
        assert sha(checkpoint) == reference['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model, prior = restore_model(cfg, state), prior_from_checkpoint(state)
        rows = []
        for index in spec['conditions']:
            name = f'{method}_c{index}.pt'
            old = args.previous/name
            piece = args.out/'pieces'/f'{method}_c{index}'
            piece_report = piece/f'{method}_results.json'
            if old.exists():
                saved = torch.load(old, weights_only=False, map_location='cpu')
                c = saved['condition']
                assert c['composition_hex'] == panel[index]['composition_hex']
                assert c['atomic_numbers'] == panel[index]['atomic_numbers']
                x, x0 = saved['positions'], saved['initial_positions']
                count = spec['samples_per_condition']
                assert x.shape == x0.shape == (count, c['n_atoms'], 3)
                assert torch.isfinite(x).all() and torch.isfinite(x0).all()
                assert saved['seeds'] == [spec['evaluation_seed']*1000003+index*100003+j for j in range(count)]
                timing = timings.get((method, index))
                row = dict(method=method, condition_index=index, condition=c, sample_sha256=sha(old),
                    checkpoint_sha256=reference['checkpoint_sha256'], generation_seconds=timing or 0.,
                    generation_timing_missing=timing is None, initial_geometry=geometry_counts(x0, c['numbers']),
                    final_geometry=geometry_counts(x, c['numbers']), **assess(x, c, list(range(count))))
                link(old, args.out/name)
                totals['reused_previous'] += count
                totals['timing_missing'] += int(timing is None)
            else:
                if not piece_report.exists():
                    if piece.exists():
                        # Keep an interrupted fragment, including any saved raw coordinates.
                        attempt = 0
                        while piece.with_name(piece.name+f'_interrupted{attempt}').exists():
                            attempt += 1
                        piece.rename(piece.with_name(piece.name+f'_interrupted{attempt}'))
                    piece.mkdir(parents=True)
                    evaluate(model, prior, method, cfg, dict(spec, conditions=[index]), ph, piece,
                             reference['checkpoint_sha256'], manifest)
                    totals['newly_generated'] += spec['samples_per_condition']
                else:
                    totals['reused_continuation'] += spec['samples_per_condition']
                report = json.loads(piece_report.read_text())
                assert report['complete'] and report['protocol_sha256'] == ph and len(report['rows']) == 1
                row = report['rows'][0]
                assert row['condition_index'] == index and row['checkpoint_sha256'] == reference['checkpoint_sha256']
                assert sha(piece/name) == row['sample_sha256']
                link(piece/name, args.out/name)
            rows.append(row)
        write(args.out/f'{method}_results.json', dict(complete=True, protocol_sha256=ph,
            checkpoint_sha256=reference['checkpoint_sha256'], rows=rows,
            generation_timing_complete=not any(r.get('generation_timing_missing', False) for r in rows),
            previous_run=str(args.previous), previous_stdout_sha256=sha(args.previous_log),
            scope='Unchanged frozen models and sample streams; completed raw outputs retained across scheduler continuation.',
            new_molecular_oracle_calls=0))
        del state, model, prior
        torch.cuda.empty_cache()
    attempt = len(list(args.out.glob('continuation_attempt_*.json')))
    write(args.out/f'continuation_attempt_{attempt}.json', totals)
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=ph,
        methods=list(spec['frozen_references']), totals=totals, new_molecular_oracle_calls=0))
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
