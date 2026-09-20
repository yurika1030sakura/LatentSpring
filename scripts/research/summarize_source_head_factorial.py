"""Post-hoc four-cell contrasts of the frozen additional-baseline experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfm_mol.replication_statistics import paired_intervals
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','audit','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());audit=json.loads(a.audit.read_text())
    assert audit['complete'] and audit['protocol_sha256']==sha(a.protocol)
    assert audit['arrays_sha256']==sha(a.audit.with_suffix('.npz'))
    data=dict(np.load(a.audit.with_suffix('.npz')));results={}
    strata=[c['n_atoms']>28 for c in spec['test_rows']]
    for metric in ['graph','joint']:
        def values(name):
            x=data[name+'_graph']
            if metric=='joint':x=x&data[name+'_success']&(data[name+'_force']<=5)
            return x.astype(float)
        g,h,e=[values(k) for k in ['gaussian_fm','harmonic_fm','edm']]
        differences=dict(source_only_vs_neither=h[:,0]-g[:,0],physical_only_vs_neither=g[:,1]-g[:,0],
            both_vs_source_only=h[:,1]-h[:,0],both_vs_physical_only=h[:,1]-g[:,1],
            both_vs_neither=h[:,1]-g[:,0],both_vs_edm=h[:,1]-e[:,0],
            interaction=h[:,1]-h[:,0]-g[:,1]+g[:,0])
        results[metric]={name:paired_intervals(x.mean(-1),strata=strata) for name,x in differences.items()}
    write(a.out,dict(complete=True,protocol_sha256=sha(a.protocol),audit_sha256=sha(a.audit),
        summary=audit['summary'],contrasts=results,new_outputs=0,new_physical_queries=0,
        scope='Post-hoc factorial contrasts requested after the frozen three-target experiment; nominal95 intervals. Both FM parents use one-pass EGNN without self-conditioning, shared training data and budgets. Each fit uses the identical previously trained distance-self-conditioned FM head on both sources. This isolates reuse of one fixed correction, not separately source-trained heads or a factorial of the final self-conditioned generator. Two archived fits on64 compositions. The EDM comparison uses these same two fits; do not combine it with the five-fit GAGA table as one identical experimental setting.'))


if __name__=='__main__':main()
