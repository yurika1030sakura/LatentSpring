"""Replay head fits, raw full-model outputs and fixed-coordinate GFN2 results."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from cfm_mol.physical_connection import make_physical_connection
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.weighted_endpoints import EndpointDraw
from cfm_mol.weighted_endpoint_fm import checkpoint_source
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.connection_capacity import cache_rows,evaluate
from scripts.research.audit_generator_output_support import assess
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def dictionary_hash(values):
    h=hashlib.sha256()
    for key,value in sorted(values.items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit_audit(project,folder,protocol,kinds,seed):
    spec=json.loads(protocol.read_text());ph=sha(protocol);done=json.loads((folder/'complete.json').read_text())
    assert done['complete'] and done['protocol_sha256']==ph and done['seed']==seed
    bank=project/spec['bank'];assert sha(bank)==spec['bank_sha256'];data=torch.load(bank,map_location='cpu',weights_only=False)
    cache=cache_rows(data['rows'],spec['pair_config'],'cpu');summaries={}
    for kind in kinds:
        logs=[json.loads(line) for line in (folder/kind/'metrics.jsonl').read_text().splitlines()];assert len(logs)==spec['steps'];rng=np.random.default_rng(seed)
        for step,row in enumerate(logs,1):
            index=spec['fit_states'][int(rng.integers(len(spec['fit_states'])))];assert row['step']==step and row['bank_row']==index and np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
        final=None
        for point in done['results'][kind]['history']:
            step=point['step'];file=folder/kind/f'step_{step}.pt';assert sha(file)==done['checkpoints'][f'{kind}/{step}']
            state=torch.load(file,map_location='cpu',weights_only=False);assert state['protocol_sha256']==ph and state['seed']==seed and state['step']==step
            model=make_physical_connection(**state['configuration']).float().eval();model.load_state_dict(state['state_dict'],strict=True)
            for split,indices in [('fit',spec['fit_states']),('validation',spec['validation_states'])]:
                replay=evaluate(model,cache,indices)
                for key,value in replay.items():np.testing.assert_allclose(value,point[split][key],atol=2e-7,rtol=1e-5)
            if step==spec['steps']:final=point
        assert final is not None;summaries[kind]=final
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);methods=spec['methods']
    parentfile=a.project/spec['checkpoint'];assert sha(parentfile)==spec['checkpoint_sha256'];parent=torch.load(parentfile,map_location='cpu',weights_only=False);prior=prior_from_checkpoint(parent)
    assert sha(a.project/spec['teacher_audit'])==spec['teacher_audit_sha256'];shape=(2,len(methods),16,16)
    graph=np.zeros(shape,bool);success=np.zeros(shape,bool);forces=np.full(shape,np.inf);fits={};attempts=0;provenance=[]
    for si,seed in enumerate(spec['training_seeds']):
        folder=a.run/f's{si}';done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['seed']==seed
        fits[str(si)]=dict(original=fit_audit(a.project,a.project/f'runs/connection_capacity_v1/s{si}',a.project/'research/evidence/connection_capacity_v1.json',['pair','context'],seed),
            wide=fit_audit(a.project,folder/'wide_fit',a.project/spec['wide_protocol'],['wide'],seed))
        raw=json.loads((folder/'audit.json').read_text());assert sha(folder/'audit.json')==done['raw_audit_sha256'] and raw['protocol_sha256']==ph;positions={}
        for mi,method in enumerate(methods):
            directory=folder/(method+'_raw');r=json.loads((directory/'generation.json').read_text());assert r['complete'] and len(r['rows'])==16 and not r['geometry_refinement'] and not r['energy_filter'] and r['physical_queries']==0
            info=done['sampling_checkpoints'][method];file=Path(info['file']);file=file if file.is_absolute() else folder/file;assert sha(file)==info['sha256']==r['checkpoint_sha256']
            state=torch.load(file,map_location='cpu',weights_only=False)
            if not method.startswith('gaga'):
                assert r['backbone_calls_per_attempt']==128 and r['connection_calls_per_attempt']==64 and r['primitive_network_calls_per_attempt']==192 and r['terminal_noise_A']==0
                for key,value in parent['state_dict'].items():assert torch.equal(state['state_dict'][key],value),key
                assert state['research_protocol']['direct_endpoint_training']['protocol_sha256']==ph
                if method=='base':
                    for name in ['weight','bias']:assert state['state_dict']['vector_field.physical_connection.pair_network.4.'+name].count_nonzero()==0
                else:
                    headfile=folder/'wide_fit/wide/step_20000.pt' if method=='pair_wide' else a.project/spec['sources'][si]['heads'][method]['path']
                    assert sha(headfile)==info['head_origin'];head=torch.load(headfile,map_location='cpu',weights_only=False)
                    for key,value in head['state_dict'].items():assert torch.equal(state['state_dict']['vector_field.physical_connection.'+key],value),key
                    assert head['configuration']==state['research_protocol']['physical_connection']
            else:
                calls=128 if method=='gaga' else 651;assert r['backbone_calls_per_attempt']==calls and r['connection_calls_per_attempt']==0 and r['primitive_network_calls_per_attempt']==calls
                assert info['sha256']==spec['sources'][si]['gaga']['checkpoint_sha256'] and r['model_state_sha256']==dictionary_hash(state['ema_state_dict'])
            savedrows=[v for v in raw['rows'] if v['arm']==method];assert len(savedrows)==16
            for i,row in enumerate(r['rows']):
                c=row['condition'];expected=spec['test_rows'][i]
                for key in ['atomic_numbers','charge','spin_multiplicity']:assert c[key]==expected[key]
                path=directory/row['file'];assert sha(path)==row['sha256']==savedrows[i]['raw_sha256']
                with np.load(path) as v:x=torch.from_numpy(v['raw_positions'].copy());initial=v['source_positions'].copy()
                result=assess(x,c,list(range(16)))
                for key in result:assert result[key]==savedrows[i][key]
                graph[si,mi,i]=[v['graph_supported'] for v in result['records']];positions[method,i]=x.numpy()
                if not method.startswith('gaga'):
                    for k in [0,8]:
                        draws=[EndpointDraw('replay','none','none',c,np.zeros((c['n_atoms'],3))) for _ in range(8)]
                        x0=checkpoint_source(prior,draws,spec['evaluation_seeds'][si]*1000003+i*100003+k)
                        np.testing.assert_array_equal(x0.reshape(8,c['n_atoms'],3).numpy(),initial[k:k+8])
                else:
                    f=folder/row['native_file'];assert sha(f)==row['native_sha256'];native=torch.load(f,map_location='cpu',weights_only=False)
                    assert native['settings']['calls']==calls and native['settings']['gaga_max_t']==650 and native['settings']['seed']==spec['evaluation_seeds'][si]
                    np.testing.assert_array_equal(native['positions'].numpy(),x.numpy());np.testing.assert_array_equal(native['initial_positions'].numpy(),initial)
            provenance.append(dict(replica=si,method=method,generation_sha256=sha(directory/'generation.json'),checkpoint_sha256=sha(file)))
        xp=folder/'xtb';physical=json.loads((xp/'audit.json').read_text());assert physical['complete'] and sha(xp/'audit.json')==done['xtb_audit_sha256'] and sha(xp/'tasks.json')==physical['tasks_sha256']
        tasks={v['task']['task_id']:v for v in json.loads((xp/'tasks.json').read_text())['tasks']};assert len(physical['rows'])==1536
        for row in physical['rows']:
            task=tasks[row['task_id']]['task'];mi=methods.index(task['method']);i,j=task['condition_index'],task['sample_index'];d=xp/'details'/row['task_id']
            np.testing.assert_array_equal(task['positions'],positions[task['method'],i][j])
            for name,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(d/name)==row[key]
            np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in (d/'input.xyz').read_text().splitlines()[2:]],task['positions'],atol=1e-12,rtol=0)
            assert row['command'][-1]=='--grad' and '--opt' not in row['command']
            if row['returncode'] is not None:
                gradient=(d/'gradient').read_text() if (d/'gradient').exists() else ''
                parsed=parse_singlepoint((d/'stdout.txt').read_text(),(d/'stderr.txt').read_text(),row['returncode'],gradient,len(task['positions']));assert parsed['success']==row['success']
                if row['success']:np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A']);assert parsed['energy_eV']==row['energy_eV']
            if row['success']:success[si,mi,i,j]=True;forces[si,mi,i,j]=np.sqrt(np.square(row['force_eV_A']).sum(-1).mean())
        attempts+=len(physical['rows'])
    joint=graph & success & (forces<=5);rng=np.random.default_rng(57081);contrasts={}
    for right in ['base','pair_long','pair_wide','gaga','gaga_full']:
        contrasts['context_minus_'+right]=bootstrap(joint[:,methods.index('context')].mean(-1)-joint[:,methods.index(right)].mean(-1),rng,20000)
    summary={}
    for mi,method in enumerate(methods):
        valid=graph[:,mi]&success[:,mi]
        summary[method]=dict(graph_rate=float(graph[:,mi].mean()),graph_by_seed=graph[:,mi].mean((1,2)).tolist(),median_valid_force=float(np.median(forces[:,mi][valid])) if valid.any() else None,
            joint_yields=[dict(threshold=t,mean=float((valid & (forces[:,mi]<=t)).mean()),by_seed=(valid & (forces[:,mi]<=t)).mean((1,2)).tolist()) for t in spec['thresholds']])
    array=a.out.with_suffix('.npz');np.savez_compressed(array,graph=graph,success=success,force=forces)
    positive=lambda key:contrasts[key]['ci95'][0]>0 and min(contrasts[key]['by_seed'])>0
    result=dict(complete=True,protocol_sha256=ph,summary=summary,contrasts=contrasts,primary_gate=positive('context_minus_base'),
        contextual_gain_over_matched_size=positive('context_minus_pair_wide'),full_system_gaga_gain=positive('context_minus_gaga') and positive('context_minus_gaga_full'),
        shared_backbone_gaga_superiority_established=False,comparison_scope=spec['comparison_scope'],fits=fits,provenance=provenance,
        arrays_sha256=sha(array),new_neural_outputs=3072,new_gfn2_attempts=attempts,new_optimizer_steps=40000,new_esen_queries=0,
        scope=spec['scope'],interval_scope='Composition bootstrap conditional on the fitted systems. FlowMol uses one shared pretrained parent; GAGA uses two independently initialized EGNN fits.')
    write(a.out,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
