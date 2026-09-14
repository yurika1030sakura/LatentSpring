#!/usr/bin/env python3
"""Frozen-decoder source utility pilot; all controls and budgets prespecified."""
import argparse
import json
from pathlib import Path
import time
import torch
from flowmol.model_utils.load import read_config_file
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.source_utility import TrustMixturePrior, importance_utility
from scripts.research.tree_prior_fm import evaluate, restore_model
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def verify_selection(directory, protocol_hash):
    state = json.loads((directory/'selection.json').read_text())
    assert state['complete'] and state['protocol_sha256'] == protocol_hash
    for name, digest in state['artifacts'].items(): assert sha(directory/name) == digest
    return state


def fit_source(base, spec, seed, bank_dir, references, out, protocol_hash):
    bank_report = json.loads((bank_dir/'fixed_results.json').read_text())
    assert bank_report['complete'] and bank_report['protocol_sha256'] == protocol_hash
    bank, label_audit = [], []
    for i, row in enumerate(bank_report['rows']):
        file = bank_dir/f'fixed_c{i}.pt'; assert sha(file) == row['sample_sha256']
        data = torch.load(file, map_location='cpu', weights_only=False)
        c = data['condition']; x0 = data['initial_positions'].double()
        assert c['composition_hex'] == references[i]['condition']['composition_hex']
        reward = torch.tensor([.8*r['graph_supported']+.2*r['geometrically_supported'] for r in row['records']], dtype=torch.double)
        with torch.no_grad():
            logq0 = torch.stack([base.log_prob(x, c['numbers'], c['charge'], c['spin_multiplicity']) for x in x0])
        permutation = torch.randperm(len(reward), generator=torch.Generator().manual_seed(31031+seed*1000+i))
        bank.append(dict(condition=c, positions=x0, logq0=logq0, reward=reward, shuffled=reward[permutation],
                         reference=torch.tensor(references[i]['reference_positions'], dtype=torch.double)[None]))
        label_audit.append(dict(condition_index=i, rewards=reward.tolist(), shuffle=permutation.tolist(),
                                source_logq0=logq0.tolist(), sample_sha256=row['sample_sha256']))
    write(out/'bank_labels.json', dict(rows=label_audit, protocol_sha256=protocol_hash))
    fitted = {}; reports = {}
    for method in ['actual', 'shuffled', 'nll']:
        torch.manual_seed(spec['prior_seed']+seed*1000)
        candidate = TreeMixturePrior(spec['prior_mode'], width=spec['edge_log_width']).double()
        prior = TrustMixturePrior(candidate, base, spec['trust_delta_nats'])
        optimizer = torch.optim.Adam(candidate.parameters(), lr=spec['prior_lr'])
        start = time.perf_counter(); metrics = []
        for epoch in range(spec['prior_epochs']):
            # Same fixed condition ordering across all objectives. No early stopping.
            order = torch.randperm(len(bank), generator=torch.Generator().manual_seed(31041+seed*1000+epoch))
            for i in order.tolist():
                b = bank[i]; c = b['condition']
                x = b['reference'] if method == 'nll' else b['positions']
                logq, trust = prior.log_prob_batch(x, c['numbers'], c['charge'], c['spin_multiplicity'])
                if method == 'nll':
                    objective = logq.mean()/(3*(len(c['numbers'])-1)); diagnostics = {}
                else:
                    objective, diagnostics = importance_utility(logq, b['logq0'], b['reward' if method == 'actual' else 'shuffled'])
                loss = -objective+spec['tree_kl_penalty']*trust['tree_kl']
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite source objective; no silent sample censoring')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), spec['prior_clip'], error_if_nonfinite=True)
                optimizer.step()
                row = dict(epoch=epoch, condition_index=i, objective=float(objective.detach()), loss=float(loss.detach()),
                    gradient_norm=float(norm), **{k:float(v.detach()) for k,v in {**trust, **diagnostics}.items()})
                metrics.append(row)
            print(json.dumps(dict(method=method, epoch=epoch+1, seconds=time.perf_counter()-start,
                mean_objective=sum(r['objective'] for r in metrics[-len(bank):])/len(bank))), flush=True)
        candidate.eval(); candidate.requires_grad_(False)
        file = out/f'{method}.pt'
        torch.save(dict(configuration=candidate.configuration, state_dict=candidate.state_dict(), delta=spec['trust_delta_nats'],
            protocol_sha256=protocol_hash, bank_report_sha256=sha(bank_dir/'fixed_results.json'),
            base_decoder=spec['frozen_references'][seed], method=method, seed=seed), file)
        write(out/f'{method}_metrics.json', metrics)
        reports[method] = dict(checkpoint_sha256=sha(file), training_seconds=time.perf_counter()-start,
            parameters=sum(p.numel() for p in candidate.parameters()), steps=len(metrics), metrics_sha256=sha(out/f'{method}_metrics.json'))
        fitted[method] = prior
    write(out/'training.json', dict(complete=True, protocol_sha256=protocol_hash, models=reports,
        bank_labels_sha256=sha(out/'bank_labels.json'), reference_fit_count=len(references),
        utility_scope='Offline training quantities only; fresh validation is required.'))
    return fitted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'selection', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--seed-index', type=int, choices=[0, 1], required=True)
    a = p.parse_args(); spec = json.loads(a.protocol.read_text()); protocol_hash = sha(a.protocol)
    assert spec['frozen']; torch.set_num_threads(2)
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    selection = verify_selection(a.selection, protocol_hash)
    refs = json.loads((a.selection/'references.json').read_text())
    start = time.perf_counter()
    source = spec['frozen_references'][a.seed_index]
    file = a.project/source['checkpoint']; assert sha(file) == source['checkpoint_sha256']
    checkpoint = torch.load(file, map_location='cpu', weights_only=False)
    base = prior_from_checkpoint(checkpoint)
    assert isinstance(base, TreeMixturePrior) and base.mode == 'fixed' and base.width == spec['edge_log_width']
    assert not checkpoint['research_protocol'].get('latent_tree_context')
    cfg_path = a.project/spec['config']; assert sha(cfg_path) == spec['config_sha256']
    cfg = read_config_file(cfg_path); cfg['mol_fm'].pop('bgfm', None)
    assert cfg['dataset']['max_atoms'] == 200 and cfg['mol_fm']['total_loss_weights']['e'] == 0
    model = restore_model(cfg, checkpoint); model.eval(); model.requires_grad_(False)
    del checkpoint
    bank_dir = a.out/'bank'; bank_dir.mkdir()
    bank_protocol = dict(spec, conditions=list(range(len(refs['fit']))), samples_per_condition=spec['bank_samples_per_condition'],
                         evaluation_seed=spec['bank_seeds'][a.seed_index])
    evaluate(model, base, 'fixed', cfg, bank_protocol, protocol_hash, bank_dir, source['checkpoint_sha256'], a.selection/'fit_panel.json')
    training_dir = a.out/'training'; training_dir.mkdir()
    priors = fit_source(base, spec, a.seed_index, bank_dir, refs['fit'], training_dir, protocol_hash)
    priors['fixed'] = TrustMixturePrior(base, base, spec['trust_delta_nats'])
    evaluation_dir = a.out/'evaluation'; evaluation_dir.mkdir()
    evaluation_protocol = dict(spec, conditions=list(range(len(refs['held']))), evaluation_seed=spec['evaluation_seeds'][a.seed_index])
    trust = []
    for method in spec['methods']:
        for i, row in enumerate(refs['held']):
            c = row['condition']
            _, _, _, kl, epsilon = priors[method].components(c['numbers'], c['charge'], c['spin_multiplicity'])
            assert float(kl*epsilon) <= spec['trust_delta_nats']+1e-8
            trust.append(dict(method=method, condition_index=i, tree_kl=float(kl), epsilon=float(epsilon), output_kl_bound=float(kl*epsilon)))
        evaluate(model, priors[method], method, cfg, evaluation_protocol, protocol_hash, evaluation_dir,
                 source['checkpoint_sha256'], a.selection/'held_panel.json')
    write(a.out/'trust.json', trust)
    # Check that all generation used the exact frozen decoder parameters.
    original = torch.load(file, map_location='cpu', weights_only=False)['state_dict']
    assert all(torch.equal(value.detach().cpu(), original[key]) for key, value in model.state_dict().items())
    write(a.out/'results.json', dict(complete=True, protocol_sha256=protocol_hash, selection_sha256=sha(a.selection/'selection.json'),
        frozen_decoder=source, frozen_decoder_state_rechecked=True, seed_index=a.seed_index, elapsed_seconds=time.perf_counter()-start,
        training_sha256=sha(training_dir/'training.json'), trust_sha256=sha(a.out/'trust.json'),
        bank_attempts=len(refs['fit'])*spec['bank_samples_per_condition'], evaluation_attempts=len(refs['held'])*spec['samples_per_condition']*len(spec['methods']),
        new_molecular_oracle_calls=0, scientific_submission_ready=False))


if __name__ == '__main__': main()
