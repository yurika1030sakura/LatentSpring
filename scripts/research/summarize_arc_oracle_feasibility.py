#!/usr/bin/env python3
"""Reproduce parent-then-composition means of the audited frozen arc endpoint test."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']: parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/arc_oracle_feasibility_protocol_v1.json'
    protocol=json.loads(pp.read_text())
    metrics=['expected_acceptance','expected_potential_change_eV','expected_squared_COM_displacement_A2']
    summaries={}; hashes={}; parents={}; raw=0
    for index in protocol['condition_indices']:
        directory=args.project/f'runs/arc_oracle_feasibility_v1/condition_{index:02d}'
        ap=args.project/f'runs/arc_oracle_feasibility_audit_v1/condition_{index:02d}/results.json'
        audit=json.loads(ap.read_text()); report=json.loads((directory/'results.json').read_text())
        assert audit['complete'] and audit['full_replay'] and report['complete']
        assert sha(directory/'results.json')==audit['results_sha256']
        assert sha(directory/'trace.pt')==audit['trace_sha256']==report['trace_sha256']
        assert report['protocol_sha256']==sha(pp)
        assert audit['raw_queries']==report['new_raw_queries']==report['requested_raw_queries']
        hashes[str(index)]=sha(ap); raw+=report['new_raw_queries']
        ids=sorted({r['parent'] for r in report['rows']}); parents[str(index)]=len(ids)
        summaries[str(index)]={}
        for method in protocol['methods']:
            rows=[r for r in report['rows'] if r['method']==method]
            assert sorted({r['parent'] for r in rows})==ids
            assert len(rows)==len(protocol['fit_context_ids'][str(index)])
            summaries[str(index)][method]={metric:mean(mean(r[metric] for r in rows if r['parent']==parent)
                for parent in ids) for metric in metrics}
    result=dict(complete=True,protocol_sha256=sha(pp),condition_summary=summaries,
        summary={method:{metric:mean(s[method][metric] for s in summaries.values()) for metric in metrics}
            for method in protocol['methods']},parents_by_case=parents,source_audit_sha256=hashes,
        new_raw_queries_in_experiment=raw,new_physical_queries_in_summary=0,scientific_submission_ready=False,
        scope='TRAINING FIT parents only. One frozen root-only proposal per context and method with true paired oracle and MH. Parent means then equal composition means. No trained learner or full-chain advantage follows.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
