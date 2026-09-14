#!/usr/bin/env python3
"""Fresh draws from frozen Gaussian/fixed references for the moment-control study."""
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
    config = args.project / protocol['config']
    assert sha(config) == protocol['config_sha256']
    cfg = read_config_file(config)
    cfg['mol_fm'].pop('bgfm', None)
    cfg['mol_fm']['prior_config']['x']['align'] = False
    manifest = args.project / protocol['condition_manifest']
    assert sha(manifest) == protocol['condition_manifest_sha256']
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    for method, reference in protocol['frozen_references'].items():
        checkpoint = args.project / reference['checkpoint']
        assert sha(checkpoint) == reference['checkpoint_sha256']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        recipe = saved['research_protocol']
        for key in ['data_seed', 'fm_seed', 'fm_steps', 'fm_lr', 'target_domain', 'warm_checkpoint_sha256']:
            assert recipe[key] == protocol[key], key
        model = restore_model(cfg, saved)
        prior = load_prior(method, None, protocol, sha(args.protocol))
        evaluate(model, prior, method, cfg, protocol, sha(args.protocol), args.out,
                 reference['checkpoint_sha256'], manifest)
        del model, saved
        torch.cuda.empty_cache()
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=sha(args.protocol),
        methods=list(protocol['frozen_references']), new_molecular_oracle_calls=0))


if __name__ == '__main__':
    main()
