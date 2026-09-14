#!/usr/bin/env python3
"""Replay source draws and structural assays; summarize frozen source utility pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.source_utility import TrustMixturePrior
from scripts.research.tree_prior_fm import sample_source, geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def flags(row, metric):
    if metric == 'utility':
        return np.array([.8*r['graph_supported']+.2*r['geometrically_supported'] for r in row['records']])
    return np.array([float(r[metric]) for r in row['records']])


def intervals(differences, strata, rng, repeats=5000):
    # Shape: frozen decoder, composition, draw. Composition uncertainty is
    # descriptive for these training-corpus-derived head-held-out conditions.
    s, c, n = differences.shape
    indices = rng.integers(n, size=(repeats, s, c, n))
    draws = differences[np.arange(s)[None,:,None,None], np.arange(c)[None,None,:,None], indices].mean((1,2,3))
    group_means = differences.mean((0, 2))
    selections = np.concatenate([rng.choice(np.where(np.asarray(strata)==b)[0], size=(repeats, sum(x==b for x in strata)), replace=True) for b in sorted(set(strata))], axis=1)
    composition = group_means[selections].mean(1)
    return dict(mean=float(differences.mean()), paired_draw95=np.quantile(draws,[.025,.975]).tolist(),
                descriptive_composition95=np.quantile(composition,[.025,.975]).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'selection', 'run', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--seed-index', type=int, choices=[0,1])
    a = p.parse_args(); spec = json.loads(a.protocol.read_text()); ph = sha(a.protocol)
    if a.out.exists(): raise FileExistsError(a.out)
    torch.set_num_threads(2)
    refs = json.loads((a.selection/'references.json').read_text())
    selection = json.loads((a.selection/'selection.json').read_text()); assert selection['protocol_sha256']==ph
    for name,digest in selection['artifacts'].items(): assert sha(a.selection/name)==digest
    assert not {r['condition']['composition_hex'] for r in refs['fit']} & {r['condition']['composition_hex'] for r in refs['held']}
    # Independent replay of every original qualification and its selection.
    for role in ['fit','held']:
        for r in refs[role]:
            result=assess(torch.tensor(r['reference_positions'],dtype=torch.double)[None],r['condition'],[0])
            assert result['graph_supported']==1 and result['validator_errors']==0
    seeds=[a.seed_index] if a.seed_index is not None else [0,1]
    all_results={}; audit_rows=[]; total_sources=0; costs={}
    for s in seeds:
        root=a.run/f's{s}'
        completed=json.loads((root/'results.json').read_text())
        assert completed['complete'] and completed['protocol_sha256']==ph and completed['frozen_decoder_state_rechecked']
        assert completed['selection_sha256']==sha(a.selection/'selection.json')
        source=spec['frozen_references'][s]
        file=a.project/source['checkpoint']; assert sha(file)==source['checkpoint_sha256']
        state=torch.load(file,map_location='cpu',weights_only=False)
        base=prior_from_checkpoint(state); del state
        labels=json.loads((root/'training/bank_labels.json').read_text())['rows']
        training=json.loads((root/'training/training.json').read_text())
        assert sha(root/'training/training.json')==completed['training_sha256']
        assert sha(root/'training/bank_labels.json')==training['bank_labels_sha256']
        for name, record in training['models'].items():
            assert record['steps']==spec['prior_epochs']*len(refs['fit'])
            assert sha(root/'training'/f'{name}_metrics.json')==record['metrics_sha256']
        all_results[s]={}
        bank_cost=json.loads((root/'bank/fixed_results.json').read_text())
        bank_seconds=sum(r['generation_seconds'] for r in bank_cost['rows'])
        costs[s]=dict(total_run_elapsed_seconds=completed['elapsed_seconds'],bank_generation_seconds=bank_seconds,
            bank_attempts=completed['bank_attempts'],source_training_seconds={m:r['training_seconds'] for m,r in training['models'].items()},
            independent_method_preparation_seconds={m:(bank_seconds if m in ['actual','shuffled'] else 0.)+(training['models'][m]['training_seconds'] if m!='fixed' else 0.) for m in spec['methods']},
            scope='Existing decoder training is common and reused. Actual/shuffled each require the full bank when deployed independently; the study generates it once and shares it. NLL does not require that bank. Training-bank density preprocessing is included in run elapsed, not these component timers. Corpus qualification/selection ran separately and its wall time was not recorded; reported preparation times are lower bounds excluding that stage.')
        for phase,methods,refrows in [('bank',['fixed'],refs['fit']),('evaluation',spec['methods'],refs['held'])]:
            for method in methods:
                if phase=='bank': prior=base
                elif method=='fixed': prior=TrustMixturePrior(base,base,spec['trust_delta_nats'])
                else:
                    file=root/'training'/f'{method}.pt'
                    assert sha(file)==training['models'][method]['checkpoint_sha256']
                    saved=torch.load(file,map_location='cpu',weights_only=False)
                    assert saved['protocol_sha256']==ph and saved['base_decoder']==source and saved['method']==method and saved['seed']==s
                    assert saved['configuration']['mode']==spec['prior_mode']
                    assert saved['bank_report_sha256']==sha(root/'bank/fixed_results.json')
                    candidate=TreeMixturePrior(**saved['configuration']).double()
                    candidate.load_state_dict(saved['state_dict'],strict=True);candidate.requires_grad_(False)
                    assert saved['delta']==spec['trust_delta_nats']
                    prior=TrustMixturePrior(candidate,base,saved['delta'])
                report_path=root/phase/f'{method}_results.json'
                report=json.loads(report_path.read_text()); assert report['complete'] and report['protocol_sha256']==ph
                assert report['checkpoint_sha256']==source['checkpoint_sha256']
                assert len(report['rows'])==len(refrows)
                for index,row in enumerate(report['rows']):
                    file=root/phase/f'{method}_c{index}.pt'; assert sha(file)==row['sample_sha256']
                    data=torch.load(file,map_location='cpu',weights_only=False);c=data['condition']
                    assert c['composition_hex']==refrows[index]['condition']['composition_hex']
                    x0,x=data['initial_positions'],data['positions'];assert torch.isfinite(x).all()
                    expected_n=spec['bank_samples_per_condition'] if phase=='bank' else spec['samples_per_condition']
                    assert len(x)==len(x0)==expected_n
                    seed0=spec['bank_seeds' if phase=='bank' else 'evaluation_seeds'][s]
                    assert data['seeds']==[seed0*1000003+index*100003+j for j in range(expected_n)]
                    for j,draw_seed in enumerate(data['seeds']):
                        replay,tree=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed)
                        torch.testing.assert_close(replay,x0[j],rtol=0,atol=0)
                        assert tree==data['auxiliary_tree_edges'][j]
                        total_sources+=1
                    result=assess(x,c,list(range(len(x))))
                    assert all(result[k]==row[k] for k in result)
                    assert geometry_counts(x,c['numbers'])==row['final_geometry']
                    assert geometry_counts(x0,c['numbers'])==row['initial_geometry']
                    if phase=='bank':
                        label=labels[index];assert label['sample_sha256']==row['sample_sha256']
                        reward=flags(row,'utility');np.testing.assert_array_equal(reward,label['rewards'])
                        expected_shuffle=torch.randperm(len(x),generator=torch.Generator().manual_seed(31031+s*1000+index)).tolist()
                        assert label['shuffle']==expected_shuffle
                        with torch.no_grad():
                            q=torch.stack([base.log_prob(y,c['numbers'],c['charge'],c['spin_multiplicity']) for y in x0])
                        torch.testing.assert_close(q,torch.tensor(label['source_logq0'],dtype=torch.double))
                audit_rows.append(dict(seed=s,phase=phase,method=method,attempts=sum(r['attempted'] for r in report['rows']),report_sha256=sha(report_path)))
                if phase=='evaluation': all_results[s][method]=report['rows']
                print(json.dumps(audit_rows[-1]),flush=True)
    strata=[r['condition']['size_bin'] for r in refs['held']]
    summary={}; rng=np.random.default_rng(31491)
    for s in seeds:
        summary[s]={}
        for m,rows in all_results[s].items():
            summary[s][m]=dict(attempts=sum(r['attempted'] for r in rows),graph=sum(r['graph_supported'] for r in rows),
                geometry=sum(r['geometrically_supported'] for r in rows),validator_errors=sum(r['validator_errors'] for r in rows),
                distinct_connectivity=sum(r['distinct_connectivity'] for r in rows),
                generation_seconds=sum(r['generation_seconds'] for r in rows),utility=float(np.concatenate([flags(r,'utility') for r in rows]).mean()))
    comparisons={}
    for comparator in ['fixed','shuffled','nll']:
        comparisons[comparator]={}
        for metric in ['graph_supported','geometrically_supported','utility']:
            diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(all_results[s]['actual'],all_results[s][comparator])]) for s in seeds])
            comparisons[comparator][metric]=intervals(diff,strata,rng)
    gate=False
    if len(seeds)==2:
        point=all(summary[s]['actual']['graph']>summary[s][m]['graph'] for s in seeds for m in ['fixed','shuffled'])
        bounds=all(comparisons[m]['graph_supported']['paired_draw95'][0]>0 for m in ['fixed','shuffled'])
        distinct=sum(summary[s]['actual']['distinct_connectivity']-summary[s]['fixed']['distinct_connectivity'] for s in seeds)/sum(summary[s]['fixed']['attempts'] for s in seeds)
        gate=point and bounds and distinct>=-.02
    write(a.out,dict(complete=True,protocol_sha256=ph,source_draws_replayed=total_sources,structural_attempts_replayed=total_sources,
        reference_assays_replayed=sum(len(refs[r]) for r in ['fit','held']),rows=audit_rows,summary=summary,comparisons=comparisons,costs=costs,
        development_gate_passed=gate if len(seeds)==2 else None,new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='Common frozen decoders; source-head held-out conditions from flow training corpus. Paired draw intervals conditional on these decoder checkpoints and compositions. Composition intervals descriptive; no model-seed population claim. No fresh decoder integration replay or absolute output density certification.'))


if __name__=='__main__':main()
