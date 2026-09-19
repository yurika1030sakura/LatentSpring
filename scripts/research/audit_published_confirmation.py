"""Audit published-model/noise/short-fit controls and their common-panel contrasts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.audit_generator_output_support import assess
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);s=json.loads(a.protocol.read_text());ph=sha(a.protocol);methods=s['methods']
    common=a.project/'runs/context_confirmation_v1';report=json.loads((common/'audit.json').read_text());assert report['complete'] and report['protocol_sha256']==s['common_protocol_sha256']
    assert sha(common/'audit.npz')==report['arrays_sha256'];spec=json.loads((a.project/s['common_protocol']).read_text())
    with np.load(common/'audit.npz') as f:oldgraph=f['graph'].copy();oldsuccess=f['success'].copy();oldforce=f['force'].copy()
    graph=np.zeros((2,3,16,16),bool);success=graph.copy();force=np.full(graph.shape,np.inf);noise=[];provenance=[]
    for si in [0,1]:
        folder=a.run/f's{si}';done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
        source=s['published_sources'][si]['checkpoint'];assert sha(a.project/source['path'])==source['sha256']
        raw=json.loads((folder/'audit.json').read_text());assert sha(folder/'audit.json')==done['raw_audit_sha256'];positions={}
        short=torch.load(folder/'pair_short.ckpt',map_location='cpu',weights_only=False);parent=torch.load(a.project/s['checkpoint'],map_location='cpu',weights_only=False)
        hsource=s['short_sources'][si];assert sha(a.project/hsource['path'])==hsource['sha256'];head=torch.load(a.project/hsource['path'],map_location='cpu',weights_only=False);assert head['step']==2000
        for key,value in parent['state_dict'].items():assert torch.equal(short['state_dict'][key],value)
        for key,value in head['state_dict'].items():assert torch.equal(short['state_dict']['vector_field.physical_connection.'+key],value)
        for mi,method in enumerate(methods):
            directory=folder/(method+'_raw');r=json.loads((directory/'generation.json').read_text());assert r['complete'] and len(r['rows'])==16 and not r['geometry_refinement'] and not r['energy_filter']
            expected=sha(folder/'pair_short.ckpt') if method=='pair_short' else source['sha256'];assert r['checkpoint_sha256']==expected
            if method=='pair_short':assert r['backbone_calls_per_attempt']==128 and r['connection_calls_per_attempt']==64
            else:assert r['backbone_calls_per_attempt']==128 and r['head_calls_per_attempt']==0 and r['terminal_noise_A']==(.025 if method=='published' else 0.)
            stored=[v for v in raw['rows'] if v['arm']==method];assert len(stored)==16
            for i,row in enumerate(r['rows']):
                c=row['condition'];expected=s['test_rows'][i]
                for key in ['atomic_numbers','charge','spin_multiplicity']:assert c[key]==expected[key]
                path=directory/row['file'];assert sha(path)==row['sha256']==stored[i]['raw_sha256']
                with np.load(path) as f:
                    x=f['raw_positions'].copy();initial=f['source_positions'].copy()
                    if method!='pair_short':zero=f['zero_noise_positions'].copy();delta=f['terminal_displacement'].copy()
                result=assess(torch.from_numpy(x),c,list(range(16)))
                for key in result:assert result[key]==stored[i][key]
                graph[si,mi,i]=[v['graph_supported'] for v in result['records']];positions[method,i]=x
                if method=='pair_short':
                    with np.load(common/f's{si}/pair_long_raw'/row['file']) as f:np.testing.assert_array_equal(initial,f['source_positions'])
                else:
                    native=folder/row['native_file'];assert sha(native)==row['native_sha256'];v=torch.load(native,map_location='cpu',weights_only=False)
                    np.testing.assert_array_equal(delta,v['positions'].numpy()-zero);np.testing.assert_array_equal(v['initial_positions'].numpy(),initial)
                    np.testing.assert_array_equal(x,v['positions'].numpy() if method=='published' else zero)
                    np.testing.assert_allclose(delta.mean(1),0,atol=1e-12,rtol=0)
                    if method=='published':noise.extend(np.sqrt(np.square(delta).sum(-1).mean(-1)).tolist())
            provenance.append(dict(replica=si,method=method,generation_sha256=sha(directory/'generation.json')))
        xp=folder/'xtb';physical=json.loads((xp/'audit.json').read_text());assert physical['complete'] and sha(xp/'audit.json')==done['xtb_audit_sha256'] and sha(xp/'tasks.json')==physical['tasks_sha256']
        tasks={v['task']['task_id']:v['task'] for v in json.loads((xp/'tasks.json').read_text())['tasks']};assert len(physical['rows'])==768
        for row in physical['rows']:
            task=tasks[row['task_id']];mi=methods.index(task['method']);i,j=task['condition_index'],task['sample_index'];d=xp/'details'/row['task_id']
            np.testing.assert_array_equal(task['positions'],positions[task['method'],i][j])
            for name,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(d/name)==row[key]
            np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in (d/'input.xyz').read_text().splitlines()[2:]],task['positions'],atol=1e-12,rtol=0)
            assert row['command'][-1]=='--grad' and '--opt' not in row['command']
            if row['returncode'] is not None:
                gradient=(d/'gradient').read_text() if (d/'gradient').exists() else '';parsed=parse_singlepoint((d/'stdout.txt').read_text(),(d/'stderr.txt').read_text(),row['returncode'],gradient,len(task['positions']));assert parsed['success']==row['success']
                if row['success']:np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A']);assert parsed['energy_eV']==row['energy_eV']
            if row['success']:success[si,mi,i,j]=True;force[si,mi,i,j]=np.sqrt(np.square(row['force_eV_A']).sum(-1).mean())
    allmethods=spec['methods']+methods;g=np.concatenate([oldgraph,graph],1);ok=np.concatenate([oldsuccess,success],1);f=np.concatenate([oldforce,force],1);joint=g&ok&(f<=5.)
    rng=np.random.default_rng(57091);contrasts={}
    for left,right in [('published','gaga'),('published','gaga_full'),('published_zero','published'),('published_zero','gaga'),('published_zero','gaga_full'),('pair_long','pair_short'),('pair_long','published'),('pair_long','published_zero'),('context','published'),('context','published_zero')]:
        contrasts[left+'_minus_'+right]=bootstrap(joint[:,allmethods.index(left)].mean(-1)-joint[:,allmethods.index(right)].mean(-1),rng,20000)
    summary={}
    for mi,method in enumerate(allmethods):
        valid=g[:,mi]&ok[:,mi];summary[method]=dict(graph_rate=float(g[:,mi].mean()),graph_by_seed=g[:,mi].mean((1,2)).tolist(),median_valid_force=float(np.median(f[:,mi][valid])) if valid.any() else None,
            joint5=float(joint[:,mi].mean()),joint5_by_seed=joint[:,mi].mean((1,2)).tolist(),joint2=float((valid&(f[:,mi]<=2.)).mean()))
    array=a.out.with_suffix('.npz');np.savez_compressed(array,graph=g,success=ok,force=f)
    result=dict(complete=True,protocol_sha256=ph,common_audit_sha256=sha(common/'audit.json'),methods=allmethods,summary=summary,contrasts=contrasts,
        arrays_sha256=sha(array),noise_rms_mean=float(np.mean(noise)),provenance=provenance,new_neural_trajectories=1024,additional_derived_outputs=512,total_scored_outputs=1536,new_gfn2_attempts=1536,new_training_steps=0,new_esen_queries=0,
        scope=s['scope'],timing=s['timing'],comparison_scope=s['comparison_scope'])
    write(a.out,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
