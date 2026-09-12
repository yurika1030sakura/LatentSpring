#!/usr/bin/env python3
"""Attach already-paid source/candidate forces with explicit raw-query provenance.

Candidate force is evaluation/training data and must NEVER enter a forward
pre-query screen. At inference it is available only after the energy query.
"""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/accepted_utility_data_protocol_v1.json';protocol=json.loads(pp.read_text())
    dp=a.project/'runs/accepted_utility_data_v1/dataset';header=json.loads((dp/'results.json').read_text())
    assert header['complete'] and header['protocol_sha256']==sha(pp) and header['data_sha256']==sha(dp/'data.pt')
    rows=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    lookup={(r['index'],r['replica'],r['parent'],r['step']):r for r in rows};assert len(lookup)==len(rows)==1671
    outputs={};checked_states=0;max_force_error=0.;max_score_error=0.;sources=[]
    for source in protocol['source_arms']:
        path=a.project/source['directory']/source['trace']
        assert sha(path)==source['trace_sha256'] and sha(path.parent/'results.json')==source['results_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False);cache={}
        def force(state_id):
            nonlocal checked_states,max_force_error,max_score_error
            if state_id in cache:return cache[state_id]
            state=saved['states'][state_id];q=saved['query_trace'][state['query_batch']];j=state['query_row']
            torch.testing.assert_close(state['positions'],q['positions'][j],atol=0,rtol=0)
            value=(q['raw_force_eV_A'][j]-q['inverted_force_eV_A'][j])/2
            assert torch.isfinite(value).all()
            error=float((value-state['force_eV_A']).abs().max());assert error<1e-10;max_force_error=max(max_force_error,error)
            basis=centered_orthonormal_basis(len(value));target_force=value-.1*state['positions']
            score=(basis.T@(target_force/.025851999786435)).flatten()
            error=float((score-state['score']).abs().max());assert error<1e-7;max_score_error=max(max_score_error,error)
            checked_states+=1;cache[state_id]=(value,dict(state_id=state_id,query_batch=state['query_batch'],query_row=j))
            return cache[state_id]
        for step,transitions in enumerate(saved['transitions']):
            for offset,move in zip(saved['rounds'][step]['active_indices'],transitions):
                if move['kind']!='joint_exchange':continue
                parent=saved['parent_ids'][offset];key=(source['index'],source['replica'],parent,step)
                row=lookup[key];assert key not in outputs and row['valid']==move['valid'] and tuple(row['action'])==tuple(move['action'])
                assert row['source_trace_sha256']==source['trace_sha256']
                old=saved['states'][move['old_state_id']]
                torch.testing.assert_close(row['x'],old['positions'],atol=0,rtol=0)
                assert row['electronic'].tolist()==[old['charge'],old['spin_multiplicity'],.025851999786435]
                source_force,source_provenance=force(move['old_state_id'])
                extra=dict(source_force_eV_A=source_force,source_force_provenance=source_provenance,
                           force_source_trace=source['directory']+'/'+source['trace'])
                if row['valid']:
                    new=saved['states'][move['new_state_id']]
                    torch.testing.assert_close(row['y'],new['positions'],atol=0,rtol=0)
                    assert (old['charge'],old['spin_multiplicity'])==(new['charge'],new['spin_multiplicity'])
                    candidate_force,candidate_provenance=force(move['new_state_id'])
                    delta=row['y']-row['x']
                    extra.update(candidate_force_eV_A=candidate_force,candidate_force_provenance=candidate_provenance,
                        source_linear_work_eV=((source_force-.1*row['x'])*delta).sum(),
                        reverse_linear_work_eV=((candidate_force-.1*row['y'])*(-delta)).sum())
                outputs[key]=dict(row,**extra)
        sources.append(dict(index=source['index'],replica=source['replica'],trace_sha256=sha(path),unique_force_states_checked=len(cache)))
    assert len(outputs)==len(rows)
    final=[outputs[(r['index'],r['replica'],r['parent'],r['step'])] for r in rows]
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);torch.save(final,a.out/'data.pt')
    result=dict(complete=True,source_data_sha256=sha(dp/'data.pt'),source_protocol_sha256=sha(pp),data_sha256=sha(a.out/'data.pt'),
        attempts=len(final),scored=sum(r['valid'] for r in final),source_forces=len(final),candidate_forces=sum(r['valid'] for r in final),
        fit_parents=len({(r['index'],r['parent']) for r in final if r['role']=='fit'}),
        internal_selection_parents=len({(r['index'],r['parent']) for r in final if r['role']=='withheld_parent'}),
        checked_unique_force_states=checked_states,maximum_force_reconstruction_error=max_force_error,
        maximum_projected_score_reconstruction_error=max_score_error,sources=sources,new_physical_queries=0,model_fitted=False,
        candidate_force_available_before_query=False,scientific_submission_ready=False,
        scope='All existing pairs and failures retained, original splits/electronic states unchanged. Source force is cached sampler state. Candidate force may enter training/audit and post-query reverse correction, never the forward pre-query decision.')
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
