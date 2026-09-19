"""Independent potential check of all selected TRAIN labels before and after correction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','teacher','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    done=json.loads((a.teacher/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and sha(a.teacher/'bank.pt')==done['bank_sha256']
    bank=torch.load(a.teacher/'bank.pt',map_location='cpu',weights_only=False);rows=[];a.out.mkdir(parents=True,exist_ok=False)
    for method in ['reference','physical']:
        folder=a.out/(method+'_raw');folder.mkdir();report=[]
        for i,row in enumerate(bank['rows']):
            x=row[method][None];c=row['condition'];path=folder/f'c{i}.npz'
            np.savez_compressed(path,raw_positions=x.numpy(),atomic_numbers=np.array(c['numbers']))
            digest=sha(path);report.append(dict(condition=c,attempted=1,file=path.name,sha256=digest))
            rows.append(dict(arm=method,condition=c,raw_sha256=digest,**assess(x,c,[0])))
        write(folder/'generation.json',dict(complete=True,rows=report,artifact_role='TRAIN reference labels, not neural outputs',neural_generation=False))
    write(a.out/'audit.json',dict(complete=True,protocol_sha256=ph,rows=rows,neural_generation=False))
    quality=dict(frozen=True,native_protocol_sha256=ph,methods=['reference','physical'],condition_count=128,samples_per_condition=1,
        xtb_binary=spec['xtb_binary'],xtb_binary_sha256=spec['xtb_binary_sha256'],xtb=spec['xtb'],thresholds=spec['thresholds'],
        scope='All128 TRAIN reference labels and their capped eSEN-relaxed counterparts scored under GFN2 at fixed coordinates. These256 records are not generator outputs or test examples. No labels selected or changed using these GFN2 outcomes.')
    qp=a.out/'protocol.json';write(qp,quality)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,'-u','-m','scripts.research.score_native_minima','--project',str(a.project),
        '--run',str(a.out),'--protocol',str(qp),'--out',str(a.out/'xtb')],env=env,check=True)


if __name__=='__main__':main()
