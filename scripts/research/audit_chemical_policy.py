#!/usr/bin/env python3
"""Independently reconstruct saved chemical-policy target, proposals and decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii,exchange_terminal_sites,terminal_exchange_actions
from cfm_mol.chemical_policy import ChemicalMovePolicy
from cfm_mol.chemical_sampler import policy_log_probabilities
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_trace(data,condition,kT,restraint,std,policy=None,uniform_local=.5,rng_seed=None):
    states=data['states'];basis=centered_orthonormal_basis(len(condition['numbers']));radii=covalent_radii(condition['numbers'])
    queries=data['query_trace'];raw_count=sum(2*len(q['positions']) for q in queries)
    for i,q in enumerate(queries):
        assert q['raw_queries_after']-q['raw_queries_before']==2*len(q['positions'])
        if i:assert q['raw_queries_before']==queries[i-1]['raw_queries_after']
    for i,s in enumerate(states):
        assert s['state_id']==i
        q=queries[s['query_batch']];j=s['query_row'];x=q['positions'][j]
        torch.testing.assert_close(x,s['positions'],atol=1e-12,rtol=0)
        assert float(x.mean(0).abs().max())<1e-8
        energy=.5*(q['raw_energy_eV'][j]+q['inverted_energy_eV'][j])
        force=.5*(q['raw_force_eV_A'][j]-q['inverted_force_eV_A'][j])
        potential=energy+restraint/2*x.square().sum()
        torch.testing.assert_close(energy,s['energy_eV']);torch.testing.assert_close(force,s['force_eV_A'])
        torch.testing.assert_close(potential,s['potential_eV'])
        score=(basis.T@((force-restraint*x)/kT)).flatten()
        torch.testing.assert_close(score,s['score'])
        graph=infer_chemical_graph(x,condition['numbers'],condition['charge'])
        assert graph['connectivity_smiles']==s['graph']['connectivity_smiles']
        torch.testing.assert_close(graph['bond_orders'],s['graph']['bond_orders'])
        assert terminal_exchange_actions(condition['numbers'],graph['bond_orders'])==s['actions']
    def drift(s,scale):
        score=s['score'];norm=float(score.norm());clipped=score*min(1.,100/kT/max(norm,1e-300))
        return (basis.T@s['positions']).flatten()+.5*scale**2*clipped
    transition_rows=data.get('warm_transitions',data.get('transitions',[]));table=data.get('table',[])
    probabilities=None
    if 'transitions' in data:
        with torch.no_grad():
            # Batching is exact by the model's independent-state contract.
            probabilities=torch.cat([policy_log_probabilities(states[i:i+256],condition['numbers'],kT,policy,uniform_local)
                for i in range(0,len(states),256)])
    for row in transition_rows+table:
        old=states[row['old_state_id']];x=old['positions']
        scale=row.get('proposal_std',std)
        if row['action_index']:
            action=old['actions'][row['action_index']-1];assert action==row['action']
            y,volume,inverse=exchange_terminal_sites(x,radii,action)
            torch.testing.assert_close(volume,row['log_volume']);assert inverse==row['inverse_action']
        else:
            mean=drift(old,scale);y=basis@(mean+scale*row['noise']).reshape(len(condition['numbers'])-1,3)
            torch.testing.assert_close(mean,row['forward_mean'])
        torch.testing.assert_close(y,row['proposal_positions'],atol=1e-10,rtol=0)
        if row['valid']:
            new=states[row['new_state_id']];torch.testing.assert_close(y,new['positions'],atol=1e-10,rtol=0)
            base=-(new['potential_eV']-old['potential_eV'])/kT
            if row['action_index']:
                assert new['actions'][row['reverse_action_index']-1]==inverse
                recovered,rv,_=exchange_terminal_sites(y,radii,inverse)
                torch.testing.assert_close(recovered,x,atol=1e-10,rtol=0)
                torch.testing.assert_close(rv+volume,torch.zeros_like(volume),atol=1e-10,rtol=0)
                base+=volume
            else:
                reverse_mean=drift(new,scale)
                z=(basis.T@x).flatten();w=(basis.T@y).flatten()
                base+=((w-mean).square().sum()-(z-reverse_mean).square().sum())/(2*scale**2)
                torch.testing.assert_close(reverse_mean,row['reverse_mean'])
            torch.testing.assert_close(base,row['base_log_ratio'],atol=1e-7,rtol=1e-9)
        else:
            assert row['new_state_id']==-1 and torch.isneginf(row['base_log_ratio'])
            try:
                graph=infer_chemical_graph(y,condition['numbers'],condition['charge'])
                assert row['action_index'] and row['inverse_action'] not in terminal_exchange_actions(condition['numbers'],graph['bond_orders'])
            except ValueError:pass
        if 'accepted' in row:
            if probabilities is not None:
                pf=float(probabilities[row['old_state_id'],row['action_index']])
                pr=float(probabilities[row['new_state_id'],row['reverse_action_index']]) if row['valid'] else 0.
                assert abs(pf-row['log_forward_policy'])<1e-9
                assert abs(pr-row['log_reverse_policy'])<1e-9
            ratio=float(row['base_log_ratio'])+row['log_reverse_policy']-row['log_forward_policy']
            assert ratio==row['log_acceptance_ratio'] or abs(ratio-row['log_acceptance_ratio'])<1e-7
            assert row['accepted']==(row['valid'] and row['log_uniform']<min(0.,ratio))
    if 'history_state_ids' in data:
        history=data['history_state_ids'];chains=len(history[0])
        assert len(transition_rows)==(len(history)-1)*chains
        for step,(old,new) in enumerate(zip(history,history[1:])):
            for chain,r in enumerate(transition_rows[step*chains:(step+1)*chains]):
                assert old[chain]==r['old_state_id']
                assert new[chain]==(r['new_state_id'] if r['accepted'] else r['old_state_id'])
    else:
        ids=list(data['initial_state_ids']);chains=len(ids)
        assert len(transition_rows)%chains==0
        for begin in range(0,len(transition_rows),chains):
            for chain,r in enumerate(transition_rows[begin:begin+chains]):
                assert r['old_state_id']==ids[chain]
                if r['accepted']:ids[chain]=r['new_state_id']
        assert ids==data['warm_state_ids']
    if rng_seed is not None:
        rng=torch.Generator().manual_seed(rng_seed)
        if 'inversion_signs' in data:
            signs=2*torch.randint(2,(len(data['inversion_signs']),),generator=rng)-1
            torch.testing.assert_close(signs,data['inversion_signs'])
        chains=len(data.get('history_state_ids',[data.get('initial_state_ids',[])])[0])
        for begin in range(0,len(transition_rows),chains):
            group=transition_rows[begin:begin+chains]
            if probabilities is not None:
                p=probabilities[[r['old_state_id'] for r in group]].exp()
                choices=torch.multinomial(p,1,generator=rng)[:,0].tolist()
                assert choices==[r['action_index'] for r in group]
            for r in group:
                if r['action_index']==0:
                    noise=torch.randn(r['noise'].shape,dtype=torch.float64,generator=rng)
                    torch.testing.assert_close(noise,r['noise'],atol=0,rtol=0)
            logu=torch.rand(chains,dtype=torch.float64,generator=rng).log()
            assert logu.tolist()==[r['log_uniform'] for r in group]
        for r in table:
            if r['action_index']==0:
                torch.testing.assert_close(torch.randn(r['noise'].shape,dtype=torch.float64,generator=rng),r['noise'],atol=0,rtol=0)
        torch.testing.assert_close(rng.get_state(),data['generator_state'],atol=0,rtol=0)
    return dict(raw_queries=raw_count,scored_states=len(states),proposals=len(transition_rows)+len(table),
        invalid_proposals=sum(not r['valid'] for r in transition_rows+table),
        saved_raw_pair_reconstruction=True,proposal_and_reverse_ratio_replay=True,
        rng_replay=rng_seed is not None,oracle_requeried=False,independent_physical_accuracy_certified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--table',type=Path,required=True)
    p.add_argument('--evaluation',type=Path);p.add_argument('--policies',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    protocol=json.loads((root/'research/evidence/chemical_policy_protocol_v1.json').read_text())
    header=json.loads((args.table/'results.json').read_text());assert header['complete']
    rows=[]
    for stream_index,stream in enumerate(['training','development']):
        path=args.table/f'{stream}.pt';assert sha(path)==header['artifacts'][stream]
        data=torch.load(path,map_location='cpu',weights_only=False)
        row=audit_trace(data,data['condition'],data['kT_eV'],data['restraint_eV_A2'],protocol['local_scale']*data['kT_eV']**.5,
            rng_seed=protocol['table_seed']+stream_index)
        assert row['raw_queries']==header['streams'][stream]['raw_queries']
        rows.append(dict(stream=stream,artifact_sha256=sha(path),**row))
    condition=data['condition'];kT=data['kT_eV'];restraint=data['restraint_eV_A2']
    if args.evaluation:
        for method in ['uniform','uniform_exchange','learned']:
            for replica in [0,1]:
                directory=args.evaluation/f'{method}_s{replica}';report=json.loads((directory/'results.json').read_text())
                assert report['complete'] and sha(directory/'trace.pt')==report['trace_sha256']
                trace=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False);policy=None
                if method=='learned':
                    path=args.policies/f'replica_{replica}/policy.pt';assert sha(path)==report['policy_sha256']
                    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                    policy=ChemicalMovePolicy(**checkpoint['configuration']).double();policy.load_state_dict(checkpoint['state_dict']);policy.eval()
                row=audit_trace(trace,condition,kT,restraint,protocol['local_scale']*kT**.5,policy,
                    report['uniform_local_probability'] if method!='learned' else .5,
                    rng_seed=protocol['evaluation_seeds'][replica])
                assert row['raw_queries']==report['new_raw_queries']
                for h,ids in zip(report['history'],trace['history_state_ids']):
                    assert h['smiles']==[trace['states'][i]['graph']['connectivity_smiles'] for i in ids]
                    assert h['energy_eV']==[float(trace['states'][i]['energy_eV']) for i in ids]
                rows.append(dict(method=method,replica=replica,artifact_sha256=sha(directory/'trace.pt'),**row))
    result=dict(complete=True,rows=rows,scientific_submission_ready=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
