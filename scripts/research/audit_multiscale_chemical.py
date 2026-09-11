#!/usr/bin/env python3
"""Replay the independent scale stream and every molecular MH decision."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from audit_chemical_policy import audit_trace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=Path(__file__).resolve().parents[2]
    protocol_path=root/'research/evidence/multiscale_chemical_protocol_v1.json';protocol=json.loads(protocol_path.read_text())
    physical=json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text());kT=physical['kT_eV']
    rows=[];reference='CS(F)(F)(F)(F)F'
    for replica in [0,1]:
        directory=args.run/f'uniform_multiscale_s{replica}'
        r=json.loads((directory/'results.json').read_text());assert r['complete']
        assert r['evaluation_protocol_sha256']==sha(protocol_path) and r['trace_sha256']==sha(directory/'trace.pt')
        d=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        assert r['transition_seed']==protocol['evaluation_seeds'][replica]
        rng=torch.Generator().manual_seed(protocol['scale_seed']+replica)
        chains=len(d['history_state_ids'][0]);steps=len(d['history_state_ids'])-1
        assert steps==protocol['uniform_total_budget_steps']
        assert len(d['scale_choice_history'])==steps
        for step,choices in enumerate(d['scale_choice_history']):
            expected=torch.randint(len(protocol['local_scales']),(chains,),generator=rng)
            assert expected.tolist()==choices
            scales=torch.tensor(protocol['local_scales'],dtype=torch.float64)[expected]*kT**.5
            for chain,row in enumerate(d['transitions'][step*chains:(step+1)*chains]):
                assert row['proposal_std']==float(scales[chain])
        torch.testing.assert_close(rng.get_state(),d['scale_generator_state'],atol=0,rtol=0)
        audit=audit_trace(d,r['condition'],kT,physical['restraint_eV_A2'],.1*kT**.5,
            uniform_local=protocol['fixed_local_probability'],rng_seed=r['transition_seed'])
        assert audit['raw_queries']==r['new_raw_queries']
        local=[q for q in d['transitions'][2::chains] if q['action_index']==0]
        rows.append(dict(replica=replica,**audit,scale_rng_replay=True,
            trace_sha256=sha(directory/'trace.pt'),source_results_sha256=sha(directory/'results.json'),
            total_raw_queries=r['history'][-1]['total_raw_queries'],
            reference_connectivity_first_hit=[next((h['step'] for h in r['history'] if h['smiles'][i]==reference),None) for i in range(chains)],
            final_energy_eV=r['history'][-1]['energy_eV'],
            difficult_parent_local_attempts=len(local),difficult_parent_local_valid=sum(q['valid'] for q in local),
            difficult_parent_local_accepted=sum(q['accepted'] for q in local)))
    report=dict(complete=True,rows=rows,physical_baseline_only=True,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
