#!/usr/bin/env python3
"""Whole recorded-prefix utility/cost constants for joint-only screen fitting.

These are fixed empirical source prefixes, not a replay of the changed chain.
"""
import argparse,json,math
from pathlib import Path
import torch
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/accepted_utility_data_protocol_v1.json';protocol=json.loads(pp.read_text())
    pairs=a.project/'runs/screen_force_pairs_v1';header=json.loads((pairs/'results.json').read_text())
    assert header['complete'] and header['source_protocol_sha256']==sha(pp) and header['data_sha256']==sha(pairs/'data.pt')
    data=torch.load(pairs/'data.pt',map_location='cpu',weights_only=False)
    lookup={(r['index'],r['replica'],r['parent'],r['step']):r for r in data};used=set();records=[];provenance=[]
    for source in protocol['source_arms']:
        path=a.project/source['directory']/source['trace'];assert sha(path)==source['trace_sha256']
        assert sha(path.parent/'results.json')==source['results_sha256']
        ap=a.project/source['audit'];assert sha(ap)==source['audit_sha256'] and json.loads(ap.read_text())['complete']
        trace=torch.load(path,map_location='cpu',weights_only=False);index=source['index'];replica=source['replica']
        by_parent={pid:dict(index=index,replica=replica,parent=pid,
            role='fit' if pid in protocol['splits'][str(index)]['fit_parent_ids'] else 'withheld_parent',
            base_work_eV=0.,base_calls=2,nonjoint_work_eV=0.,nonjoint_calls=2,joint_attempts=0,by_kind={}) for pid in trace['parent_ids']}
        for step,moves in enumerate(trace['transitions']):
            for offset,move in zip(trace['rounds'][step]['active_indices'],moves):
                parent=trace['parent_ids'][offset];row=by_parent[parent];kind=move['kind'];joint=kind=='joint_exchange'
                calls=2*int(move['valid']);work=0.
                if move['valid']:
                    old,new=[trace['states'][move[k]] for k in ['old_state_id','new_state_id']]
                    delta=float(new['potential_eV']-old['potential_eV'])
                    work=-delta*math.exp(min(0.,float(move['log_acceptance_ratio'])))
                row['base_calls']+=calls;row['base_work_eV']+=work
                stats=row['by_kind'].setdefault(kind,dict(attempted=0,scored=0,expected_work_eV=0.,raw_calls=0))
                stats['attempted']+=1;stats['scored']+=int(move['valid']);stats['expected_work_eV']+=work;stats['raw_calls']+=calls
                if joint:
                    row['joint_attempts']+=1;key=(index,replica,parent,step);record=lookup[key]
                    assert key not in used and record['role']==row['role'] and record['valid']==move['valid']
                    assert abs(work-record.get('baseline_expected_utility_eV',0.))<1e-10;used.add(key)
                else:
                    row['nonjoint_calls']+=calls;row['nonjoint_work_eV']+=work
        for offset,parent in enumerate(trace['parent_ids']):
            row=by_parent[parent];assert row['base_calls']==trace['final_queries_per_parent'][offset]==128
            records.append(row)
        provenance.append(dict(index=index,replica=replica,trace_sha256=sha(path),audit_sha256=sha(ap)))
    assert len(used)==len(data)==1671 and len(records)==96
    groups={role:[r for r in records if r['role']==role] for role in ['fit','withheld_parent']}
    summary={}
    for role,rows in groups.items():
        total=sum(r['base_calls'] for r in rows);work=sum(r['base_work_eV'] for r in rows)
        summary[role]=dict(trajectories=len(rows),mean_base_calls=total/len(rows),mean_base_work_eV=work/len(rows),
            baseline_utility_eV_per_raw_call=work/total,mean_nonjoint_calls=sum(r['nonjoint_calls'] for r in rows)/len(rows),
            mean_nonjoint_work_eV=sum(r['nonjoint_work_eV'] for r in rows)/len(rows))
    result=dict(complete=True,pair_data_sha256=sha(pairs/'data.pt'),source_protocol_sha256=sha(pp),
        rows=records,summary=summary,provenance=provenance,new_physical_queries=0,scientific_submission_ready=False,
        scope='Initial query cost and all nonjoint moves retained as fixed constants; joint utilities/costs are replaced only in a fixed-source empirical objective. This does not replay a changed screened chain or predict its endpoint distribution.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,fit=summary['fit'],all_attempts_joined=len(used),new_physical_queries=0)))


if __name__=='__main__':main()
