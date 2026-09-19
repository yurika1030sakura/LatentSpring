"""Independent fixed-coordinate GFN2 assessment of the native closed-set pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from cfm_mol.xtb_singlepoint import parse_singlepoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());assert spec['frozen']
    audit=json.loads((a.run/'audit.json').read_text());assert audit['complete']
    n_conditions=spec.get('condition_count',3);count=spec.get('samples_per_condition',128)
    if 'native_audit_sha256' in spec:assert sha(a.run/'audit.json')==spec['native_audit_sha256']
    else:assert audit['protocol_sha256']==spec['native_protocol_sha256']
    binary=Path(spec['xtb_binary']);assert sha(binary)==spec['xtb_binary_sha256']
    methods=spec.get('methods',['base','proposal_min','full_work_min'])
    tasks=[];graphs={};sources={}
    for arm in methods:
        folder=a.run/(arm+'_raw');report=json.loads((folder/'generation.json').read_text());assert report['complete']
        armrows=[r for r in audit['rows'] if r['arm']==arm]
        for i,row in enumerate(report['rows']):
            file=folder/row['file'];assert sha(file)==row['sha256']==armrows[i]['raw_sha256']
            with np.load(file) as f:positions=f['raw_positions'].copy()
            c=row['condition'];graphs[arm,i]=np.array([r['graph_supported'] for r in armrows[i]['records']]);sources[str(file)]=sha(file)
            for j,x in enumerate(positions):
                task=dict(task_id=f'{arm}_c{i}_s{j}',method=arm,replica=spec.get('replica',0),parent_id=i*count+j,
                    inversion_check=False,positions=x.tolist(),condition_index=i,sample_index=j)
                tasks.append((task,c))
    assert len(tasks)==len(methods)*n_conditions*count;a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'tasks.json',dict(protocol_sha256=sha(a.protocol),tasks=[dict(task=t,condition=c) for t,c in tasks],sources=sources))
    rows=[]
    def work(t,c):
        r=run_task(t,c,binary,a.out,spec['xtb']);r.update(condition_index=t['condition_index'],sample_index=t['sample_index']);return r
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(work,t,c) for t,c in tasks]):
            rows.append(f.result())
            if len(rows)%128==0:write(a.out/'progress.json',dict(complete=False,completed=len(rows)))
    mapping={t['task_id']:t for t,c in tasks};arrays={arm:dict(force=np.full((n_conditions,count),np.inf),success=np.zeros((n_conditions,count),bool)) for arm in methods}
    for r in rows:
        t=mapping[r['task_id']];folder=a.out/'details'/r['task_id']
        assert sha(folder/'input.xyz')==r['input_xyz_sha256'] and sha(folder/'stdout.txt')==r['stdout_sha256'] and sha(folder/'stderr.txt')==r['stderr_sha256']
        np.testing.assert_allclose([[float(v) for v in s.split()[1:]] for s in (folder/'input.xyz').read_text().splitlines()[2:]],t['positions'],atol=1e-12,rtol=0)
        assert '--opt' not in r['command'] and r['command'][-1]=='--grad'
        if r['returncode'] is not None:
            grad=(folder/'gradient').read_text() if (folder/'gradient').exists() else ''
            parsed=parse_singlepoint((folder/'stdout.txt').read_text(),(folder/'stderr.txt').read_text(),r['returncode'],grad,len(t['positions']));assert parsed['success']==r['success']
            if r['success']:np.testing.assert_array_equal(parsed['force_eV_A'],r['force_eV_A'])
        if r['success']:
            i,j=t['condition_index'],t['sample_index'];arrays[t['method']]['success'][i,j]=True
            arrays[t['method']]['force'][i,j]=np.sqrt(np.square(r['force_eV_A']).sum(-1).mean())
    summary={}
    for arm,d in arrays.items():
        graph=np.stack([graphs[arm,i] for i in range(n_conditions)]);valid=graph & d['success']
        summary[arm]=dict(attempted=n_conditions*count,graph_valid=int(graph.sum()),gfn2_failures=int((~d['success']).sum()),
            median_valid_force=float(np.median(d['force'][valid])) if valid.any() else None,
            joint_yield=[dict(threshold=t,mean=float((valid & (d['force']<=t)).mean()),by_composition=(valid & (d['force']<=t)).mean(-1).tolist()) for t in spec['thresholds']])
    result=dict(complete=True,protocol_sha256=sha(a.protocol),tasks_sha256=sha(a.out/'tasks.json'),rows=rows,summary=summary,
        new_gfn2_attempts=len(rows),new_neural_outputs=0,new_esen_queries=0,scope=spec['scope'])
    write(a.out/'audit.json',result);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
