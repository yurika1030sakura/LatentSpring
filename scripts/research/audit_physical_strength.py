"""Independent GFN2 scoring and saved-coordinate audit of selected strengths."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.confirm_gaga_feedback import audit_report,bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','protocol','out']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    spec=json.loads(args.protocol.read_text());assert spec['frozen']
    pp=args.project/spec['generation_protocol'];assert sha(pp)==spec['generation_protocol_sha256']
    campaign=json.loads(pp.read_text());selection=json.loads((args.run/'selection.json').read_text())
    completed=json.loads((args.run/'audit.json').read_text());assert completed['complete']
    assert completed['protocol_sha256']==selection['protocol_sha256']==sha(pp)
    assert completed['selection_sha256']==sha(args.run/'selection.json')
    chosen=selection['alpha'];panel=campaign['test_rows'];tasks=[];reports={};arrays={};hashes={};costs=dict(validation_outputs=0,confirmation_outputs=0,esen_queries=0)
    # Recompute every validation score and the selection from stored raw forces.
    for stage in ['validation','test']:
        count=campaign[stage+'_samples'];rows=campaign[stage+'_rows']
        for seed in campaign['seeds']:
            for family in ['distance','gaga']:
                for alpha in (campaign['alpha'] if stage=='validation' else sorted(set([0.,chosen[family]]))):
                    label=family+'_a'+str(float(alpha)).replace('.','p');folder=args.run/stage/f's{seed}'
                    report=audit_report(folder/(label+'_results.json'),rows,count)
                    path=folder/(label+'_quality.json');quality=json.loads(path.read_text())
                    assert quality['report_sha256']==sha(folder/(label+'_results.json'))
                    assert quality['arrays_sha256']==sha(folder/quality['arrays'])
                    with np.load(folder/quality['arrays']) as a:graph=a['graph'].copy();joint=a['joint'].copy()
                    for i,row in enumerate(report['rows']):
                        np.testing.assert_array_equal(graph[i],[r['graph_supported'] for r in row['records']])
                        artifact=quality['artifacts'][i];assert sha(folder/artifact['path'])==artifact['sha256']
                        saved=torch.load(folder/artifact['path'],map_location='cpu',weights_only=False)
                        assert saved['source_sha256']==row['sample_sha256']
                        idx=saved['raw_indices'];np.testing.assert_array_equal(idx.numpy(),np.where(graph[i])[0])
                        force=np.full(count,np.inf);n=len(idx)
                        if n:
                            f=saved['raw_force_eV_A'];force[idx.numpy()]=((f[:n]-f[n:])/2).square().sum(-1).mean(-1).sqrt().numpy()
                        np.testing.assert_array_equal(force,saved['force_rms'].numpy())
                        np.testing.assert_array_equal(joint[i],graph[i] & (force<=5.))
                        if stage=='test':
                            raw=torch.load(folder/f'{label}_c{i}.pt',map_location='cpu',weights_only=False)
                            for j in np.where(graph[i])[0]:
                                task=dict(task_id=f's{seed}_{label}_c{i}_j{j}',method=label,replica=seed,parent_id=i*count+int(j),
                                    inversion_check=False,positions=raw['positions'][j].tolist(),condition_index=i,sample_index=int(j))
                                tasks.append((task,row['condition']))
                    assert quality['graph']==float(graph.mean()) and quality['joint']==float(joint.mean())
                    assert quality['queries']==2*int(graph.sum())
                    costs['esen_queries']+=quality['queries'];costs['validation_outputs' if stage=='validation' else 'confirmation_outputs']+=graph.size
                    if stage=='validation':assert quality==selection['validation'][f'{seed}/{family}/{alpha}']
                    else:arrays[seed,label]=dict(graph=graph,esen_joint=joint,xtb_joint=np.zeros_like(graph),xtb_success=np.zeros_like(graph))
                    hashes[str(path.relative_to(args.run))]=sha(path)
    for family in ['distance','gaga']:
        candidates=[]
        for alpha in campaign['alpha']:
            scores=[selection['validation'][f'{s}/{family}/{alpha}'] for s in campaign['seeds']]
            if all(v['graph']>=selection['validation'][f'{s}/{family}/0.0']['graph']-.02-1e-12 for s,v in zip(campaign['seeds'],scores)):
                candidates.append((np.mean([v['joint'] for v in scores]),-alpha))
        assert chosen[family]==-max(candidates)[1]
    poolpath=args.project/campaign['source_pool'];assert sha(poolpath)==campaign['source_pool_sha256']
    pool=json.loads(poolpath.read_text())
    for i,c in enumerate(panel):
        reference=pool['rows'][c['pool_index']];assert reference['condition']['atomic_numbers']==c['atomic_numbers']
        for inverse in [False,True]:
            x=np.array(reference['reference_positions'])*(-1 if inverse else 1)
            task=dict(task_id=f'reference_c{i}_'+('minus' if inverse else 'plus'),method='reference',replica=-1,parent_id=i,
                inversion_check=inverse,mirrored_task_id=f'reference_c{i}_plus',positions=x.tolist(),condition_index=i,sample_index=-1)
            tasks.append((task,dict(c,numbers=c['atomic_numbers'])))
    binary=Path(spec['xtb_binary']);assert sha(binary)==spec['xtb_binary_sha256']
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'tasks.json',dict(protocol_sha256=sha(args.protocol),tasks=[dict(task=t,condition=c) for t,c in tasks]))
    result=dict(complete=False,protocol_sha256=sha(args.protocol),selection_sha256=sha(args.run/'selection.json'),
        tasks_sha256=sha(args.out/'tasks.json'),requested_attempts=len(tasks),rows=[])
    def work(t,c):
        r=run_task(t,c,binary,args.out,spec['xtb']);r.update(condition_index=t['condition_index'],sample_index=t['sample_index']);return r
    with ThreadPoolExecutor(max_workers=8) as executor:
        for f in as_completed([executor.submit(work,t,c) for t,c in tasks]):
            result['rows'].append(f.result())
            if len(result['rows'])%64==0:write(args.out/'progress.json',result)
    mapping={t['task_id']:(t,c) for t,c in tasks};references={}
    for row in result['rows']:
        t,c=mapping[row['task_id']];folder=args.out/'details'/row['task_id']
        for file,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(folder/file)==row[key]
        xyz=[[float(v) for v in l.split()[1:]] for l in (folder/'input.xyz').read_text().splitlines()[2:]]
        np.testing.assert_allclose(xyz,t['positions'],atol=1e-12,rtol=0)
        assert '--opt' not in row['command'] and row['command'][-1]=='--grad'
        if row['returncode'] is not None:
            gradient=(folder/'gradient').read_text() if (folder/'gradient').exists() else ''
            parsed=parse_singlepoint((folder/'stdout.txt').read_text(),(folder/'stderr.txt').read_text(),row['returncode'],gradient,len(c['numbers']))
            assert parsed['success']==row['success']
            if row['success']:
                np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A']);assert parsed['energy_eV']==row['energy_eV']
        if t['method']=='reference':references[t['task_id']]=row;continue
        a=arrays[t['replica'],t['method']];i,j=t['condition_index'],t['sample_index']
        if row['success']:
            force=np.sqrt(np.square(row['force_eV_A']).sum(-1).mean())
            a['xtb_success'][i,j]=True;a['xtb_joint'][i,j]=force<=5.
    parity=[]
    for i in range(len(panel)):
        plus,minus=[references[f'reference_c{i}_'+sign] for sign in ['plus','minus']]
        passed=plus['success'] and minus['success']
        if passed:passed=abs(plus['energy_eV']-minus['energy_eV'])<1e-5 and np.max(abs(np.array(plus['force_eV_A'])+np.array(minus['force_eV_A'])))<1e-4
        parity.append(bool(passed))
    summary={};effects={};gates={}
    for oracle in ['esen','xtb']:
        matrix={}
        for family in ['distance','gaga']:
            for role,alpha in [('selected',chosen[family]),('base',0.)]:
                label=family+'_a'+str(float(alpha)).replace('.','p')
                matrix[family+'_'+role]=np.array([arrays[s,label][oracle+'_joint'].mean(-1) for s in campaign['seeds']])
        summary[oracle]={k:dict(mean=float(v.mean()),by_seed=v.mean(-1).tolist()) for k,v in matrix.items()}
        rng=np.random.default_rng(spec['bootstrap_seed']);effects[oracle]={}
        for name,left,right in [('selected_fm_minus_selected_gaga','distance_selected','gaga_selected'),('physical_in_fm','distance_selected','distance_base'),('physical_in_gaga','gaga_selected','gaga_base')]:
            effects[oracle][name]=bootstrap(matrix[left]-matrix[right],rng,20000)
        primary=effects[oracle]['selected_fm_minus_selected_gaga'];gates[oracle]=min(primary['by_seed'])>0 and primary['ci95'][0]>0
    result.update(complete=True,summary=summary,contrasts=effects,selected_alpha=chosen,reference_parity_passed=all(parity),
        primary_gate=bool(all(gates.values()) and all(parity)),raw_quality_hashes=hashes,costs=costs,
        graph_valid_gfn2_failures=sum(not r['success'] for r in result['rows'] if r['method']!='reference'),
        gfn2_attempts=len(tasks),scope='GFN2 scores all graph-valid outputs. All attempts, including invalid outputs and failed calculations, enter joint-yield denominators. No optimization or output selection.')
    write(args.out/'audit.json',result)
    print(json.dumps(dict(complete=True,primary_gate=result['primary_gate'],selected_alpha=chosen,contrasts=effects,costs=costs,gfn2_attempts=len(tasks))),flush=True)


if __name__=='__main__':main()
