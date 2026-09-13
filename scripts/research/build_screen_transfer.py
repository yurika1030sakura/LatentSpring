#!/usr/bin/env python3
"""Extract audited physical prefixes for evaluation only; never fitting data."""
import argparse,json,math
from pathlib import Path
import torch
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/screen_transfer_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert protocol['frozen'] and not protocol['model_fitting_allowed']
    fp=a.project/protocol['fit_pair_data'];assert sha(fp)==protocol['fit_pair_data_sha256']
    old=torch.load(fp,map_location='cpu',weights_only=False);excluded={(r['index'],r['parent']) for r in old}
    physical_path=root/protocol['physical_protocol'];assert sha(physical_path)==protocol['physical_protocol_sha256'];physical=json.loads(physical_path.read_text())
    kT=physical['kT_eV'];restraint=physical['restraint_eV_A2'];cap=protocol['reference_cap_per_parent']
    records=[];prefixes=[];sources=[];checked_states=0
    for source in protocol['sources']:
        path=a.project/source['trace'];rp=a.project/source['results'];ap=a.project/source['audit']
        assert sha(path)==source['trace_sha256'] and sha(rp)==source['results_sha256'] and sha(ap)==source['audit_sha256']
        report=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert report['complete'] and audit['complete'] and audit['full_producer_replay'] and audit['producer_results_sha256']==sha(rp)
        trace=torch.load(path,map_location='cpu',weights_only=False);assert trace['parent_ids']==source['parent_ids']
        index,replica=source['index'],source['replica'];condition=report['condition']
        z=torch.tensor(condition['numbers'],dtype=torch.long);electronic=torch.tensor([condition['charge'],condition['spin_multiplicity'],kT],dtype=torch.float64);radii=covalent_radii(z)
        assert not any((index,pid) in excluded for pid in trace['parent_ids'])
        cache={}
        def state(state_id):
            nonlocal checked_states
            if state_id in cache:return cache[state_id]
            s=trace['states'][state_id];q=trace['query_trace'][s['query_batch']];j=s['query_row']
            torch.testing.assert_close(s['positions'],q['positions'][j],atol=0,rtol=0)
            force=(q['raw_force_eV_A'][j]-q['inverted_force_eV_A'][j])/2
            energy=(q['raw_energy_eV'][j]+q['inverted_energy_eV'][j])/2
            torch.testing.assert_close(force,s['force_eV_A'],atol=1e-10,rtol=0)
            torch.testing.assert_close(energy+restraint/2*s['positions'].square().sum(),s['potential_eV'],atol=1e-9,rtol=0)
            assert s['charge']==condition['charge'] and s['spin_multiplicity']==condition['spin_multiplicity']
            cache[state_id]=s;checked_states+=1;return s
        tally={pid:dict(index=index,parent=pid,replica=replica,role='evaluation_only',base_work_eV=0.,base_calls=2,
            nonjoint_work_eV=0.,nonjoint_calls=2,joint_attempts=0) for pid in trace['parent_ids']}
        for step,moves in enumerate(trace['transitions']):
            for offset,move in zip(trace['rounds'][step]['active_indices'],moves):
                if trace['query_count_history'][step][offset]>=cap:continue
                parent=trace['parent_ids'][offset];t=tally[parent];calls=2*int(move['valid']);work=0.
                old_state=state(move['old_state_id'])
                if move['valid']:
                    new=state(move['new_state_id']);delta=new['potential_eV']-old_state['potential_eV']
                    work=-float(delta)*math.exp(min(0.,float(move['log_acceptance_ratio'])))
                t['base_calls']+=calls;t['base_work_eV']+=work
                if move['kind']!='joint_exchange':
                    t['nonjoint_calls']+=calls;t['nonjoint_work_eV']+=work;continue
                t['joint_attempts']+=1
                assert move['decoder']=='arc_site'
                row=dict(index=index,parent=parent,replica=replica,step=step,role='evaluation_only',valid=move['valid'],
                    x=old_state['positions'],bonds=old_state['graph']['bond_orders'],numbers=z,electronic=electronic,radii=radii,
                    action=move.get('action'),source_force_eV_A=old_state['force_eV_A'],source_trace_sha256=source['trace_sha256'],source_state_id=move['old_state_id'])
                if move['valid']:
                    ratio=-delta/kT+move['log_reverse_coordinate']-move['log_forward_coordinate']+move['action_log_ratio']
                    assert abs(float(ratio)-move['log_acceptance_ratio'])<1e-8
                    row.update(y=new['positions'],new_bonds=new['graph']['bond_orders'],inverse_action=move['inverse_action'],
                        candidate_force_eV_A=new['force_eV_A'],candidate_state_id=move['new_state_id'],
                        log_behavior_forward=move['log_forward_coordinate'],log_behavior_reverse=move['log_reverse_coordinate'],
                        target_log_ratio=-delta/kT,action_log_ratio=delta.new_tensor(move['action_log_ratio']),reward_eV=-delta,
                        baseline_expected_utility_eV=work,connectivity_changed=old_state['graph']['connectivity_smiles']!=new['graph']['connectivity_smiles'])
                else:row['failure_reason']=move['rejection_reason']
                records.append(row)
        assert all(v['base_calls']==cap for v in tally.values());prefixes.extend(tally.values())
        sources.append(dict(index=index,replica=replica,trace_sha256=sha(path),results_sha256=sha(rp),audit_sha256=sha(ap),used_force_states=len(cache)))
    assert len(prefixes)==96 and len({(r['index'],r['parent']) for r in prefixes})==48
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);torch.save(records,a.out/'data.pt');(a.out/'prefixes.json').write_text(json.dumps(prefixes,indent=2)+'\n')
    header=dict(complete=True,protocol_sha256=sha(pp),data_sha256=sha(a.out/'data.pt'),prefixes_sha256=sha(a.out/'prefixes.json'),
        parents=48,prefixes=96,attempts=len(records),scored=sum(r['valid'] for r in records),used_force_states=checked_states,
        no_overlap_with_gate_fit_or_internal_selection=True,source_reference_calls=96*cap,new_physical_queries=0,
        model_fitting_allowed=False,model_evaluation_performed=False,sources=sources,scientific_submission_ready=False,
        scope=protocol['scope'])
    (a.out/'results.json').write_text(json.dumps(header,indent=2)+'\n');print(json.dumps({k:v for k,v in header.items() if k not in ['sources','scope']}))


if __name__=='__main__':main()
