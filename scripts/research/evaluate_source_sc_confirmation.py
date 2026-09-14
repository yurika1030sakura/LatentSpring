#!/usr/bin/env python3
"""Frozen source-SC models on the remaining reference-qualified monomer compositions."""
import argparse
import json
from pathlib import Path

import torch
from flowmol.model_utils.load import read_config_file
from scripts.research.tree_prior_fm import restore_model, load_prior, evaluate
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = json.loads(args.protocol.read_text())
    assert protocol['frozen']
    overlap_path=args.project/protocol['overlap_audit'];assert sha(overlap_path)==protocol['overlap_audit_sha256']
    overlap=json.loads(overlap_path.read_text());assert overlap['complete'] and overlap['panel_sha256']==protocol['condition_manifest_sha256']
    assert all(r['overlapping_compositions']==0 for r in overlap['corpora'].values())
    config = args.project / protocol['config']
    assert sha(config) == protocol['config_sha256']
    cfg = read_config_file(config)
    cfg['mol_fm'].pop('bgfm', None)
    manifest = args.project / protocol['condition_manifest']
    assert sha(manifest) == protocol['condition_manifest_sha256']
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    for method, reference in protocol['frozen_references'].items():
        checkpoint = args.project / reference['checkpoint']
        assert sha(checkpoint) == reference['checkpoint_sha256']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        recipe = saved['research_protocol']
        for key in ['data_sha256', 'fm_seed', 'fm_steps', 'fm_lr', 'feedback_lr', 'warm_checkpoint_sha256']:
            assert recipe[key] == protocol[key], key
        assert recipe['geometry_self_conditioning']==dict(edge_feedback='clamped',**protocol['self_conditioning'])
        model = restore_model(cfg, saved)
        if protocol.get('use_checkpoint_prior', False):
            from cfm_mol.source_checkpoint import prior_from_checkpoint
            assert recipe['source_prior_kind'] == method
            prior = prior_from_checkpoint(saved)
        else:
            prior = load_prior(method, None, protocol, sha(args.protocol))
        evaluate(model, prior, method, cfg, protocol, sha(args.protocol), args.out,
                 reference['checkpoint_sha256'], manifest)
        del model, saved
        torch.cuda.empty_cache()
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=sha(args.protocol),
        methods=list(protocol['frozen_references']), new_molecular_oracle_calls=0))


if __name__ == '__main__':
    main()
