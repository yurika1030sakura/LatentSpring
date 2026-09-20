"""Audit shared physical heads; freeze validation selection or report fresh tests."""
import argparse,datetime,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.matched_physical_connection import endpoint_and_progress
from cfm_mol.physical_connection import make_physical_connection
from cfm_mol.trajectory_physical_teacher import velocity_target
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.run_matched_physical import make_model
from scripts.research.connection_capacity import cache_rows,evaluate
from scripts.research.audit_generator_output_support import assess
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def audit_fit(root,folder,spec,ph,si,name,*,fit_subdir='fit'):
    arm=spec['parents'][si][name];checkpoint=root/arm['checkpoint'];assert sha(checkpoint)==arm['checkpoint_sha256']
    parent=make_model(arm,torch.load(checkpoint,map_location='cpu',weights_only=False)['ema_state_dict'],device='cpu').eval().requires_grad_(False)
    parent_hash=base.state_hash(parent);teacher=folder/name/'teacher';done=json.loads((teacher/'complete.json').read_text())
    assert done['complete'] and done['protocol_sha256']==ph and done['parent_state_sha256']==parent_hash and done['new_esen_queries']==1024
    bankfile=teacher/'bank.pt';assert sha(bankfile)==done['bank_sha256'];bank=torch.load(bankfile,map_location='cpu',weights_only=False)
    assert bank['protocol_sha256']==ph and bank['parent_state_sha256']==parent_hash and len(bank['rows'])==512
    rows=bank['rows'];caps=[];total=0
    for slot,index in enumerate(spec['training_rows']):
        file=teacher/'records'/f'c{slot}.pt';r=torch.load(file,map_location='cpu',weights_only=False)
        assert r['protocol_sha256']==ph and r['parent_state_sha256']==parent_hash and r['training_row']==index
        assert r['source_seed']==spec['trajectory_seeds'][si]*1000003+slot*100003 and r['oracle_queries']==8
        total+=r['oracle_queries'];assert [v['call'] for v in r['observed']]==spec['capture_calls'][name]
        for v in r['observed']:
            h,progress=endpoint_and_progress(parent,v['x'],v['native_t'],v['value'],arm['spec'])
            torch.testing.assert_close(h,v['endpoint'],atol=1e-5,rtol=2e-5);torch.testing.assert_close(progress,v['progress'],atol=1e-7,rtol=1e-6)
        progress=torch.cat([v['progress'] for v in r['observed']]);n=len(progress);raw=r['anchor_raw_force']
        force=(raw[:n]-raw[n:])/2;force-=force.mean(1,keepdim=True)
        torch.testing.assert_close(force,r['centered_even_force'],atol=0,rtol=0)
        conf=spec['target'];norm=force.flatten(1).norm(dim=-1)
        sigma=torch.minimum(torch.full_like(norm,conf['max_sigma']),torch.sqrt(conf['max_shift']*conf['kT']/norm.clamp_min(1e-12)))
        shift=sigma[:,None,None].square()*force/conf['kT']
        target,cap=velocity_target(shift,progress.double(),velocity_scale=spec['head_configuration']['velocity_scale'],gate_power=spec['head_configuration']['gate_power'])
        for x,y in [(sigma,r['sigma']),(shift,r['shift']),(target,r['velocity_target']),(cap,r['cap'])]:torch.testing.assert_close(x,y,atol=0,rtol=0)
        caps.extend(cap.tolist());xs=torch.cat([v['x'] for v in r['observed']]);hs=torch.cat([v['endpoint'] for v in r['observed']])
        for k in range(n):
            row=rows[slot*n+k];assert row['condition']==r['condition'] and row['record_sha256']==sha(file) and row['composition_slot']==slot and row['local_state']==k and row['training_row']==index
            for x,y in [(row['x'],xs[k]),(row['endpoint'],hs[k]),(row['t'],progress[k]),(row['force'],target[k].float())]:torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert total==1024
    fit=folder/name/fit_subdir;done=json.loads((fit/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
    train=done['fit_states'];val=done['validation_states'];assert len(train)==384 and len(val)==128
    assert train==[i for i,r in enumerate(rows) if r['composition_slot'] not in spec['validation_teacher_slots']]
    assert val==[i for i in range(512) if i not in train]
    assert not {rows[i]['condition']['composition_hex'] for i in train}&{rows[i]['condition']['composition_hex'] for i in val}
    logs=[json.loads(line) for line in (fit/'metrics.jsonl').read_text().splitlines()];assert len(logs)==spec['steps'];rng=np.random.default_rng(spec['head_seeds'][si])
    for i,r in enumerate(logs,1):assert r['step']==i and r['bank_row']==train[int(rng.integers(len(train)))] and np.isfinite(r['loss']) and np.isfinite(r['gradient_norm'])
    cache=cache_rows(rows,spec['head_configuration'],'cpu');assert [h['step'] for h in done['history']]==spec['diagnostic_steps']
    for point in done['history']:
        file=fit/f'step_{point["step"]}.pt';saved=torch.load(file,map_location='cpu',weights_only=False)
        assert saved['configuration']==spec['head_configuration'] and saved['protocol_sha256']==ph and saved['seed']==spec['head_seeds'][si] and saved['step']==point['step']
        head=make_physical_connection(**saved['configuration']).eval();head.load_state_dict(saved['state_dict'],strict=True)
        assert sum(p.numel() for p in head.parameters())==done['parameter_count']==7106
        for split,indices in [('fit',train),('validation',val)]:
            repeated=evaluate(head,cache,indices)
            for key,value in repeated.items():np.testing.assert_allclose(value,point[split][key],atol=3e-7,rtol=2e-5)
    final=fit/f'step_{spec["steps"]}.pt';assert sha(final)==done['checkpoint_sha256']
    return dict(path=str(final.relative_to(root)),sha256=sha(final),model_state_sha256=parent_hash,
        head_state_sha256=base.state_hash(head),history=done['history'],capped_targets=sum(v<1 for v in caps))


def audit_outputs(root,folder,spec,ph,si,phase,heads,strengths):
    conditions=spec['validation_rows' if phase=='validation' else 'test_rows'];count=spec['validation_samples' if phase=='validation' else 'test_samples']
    seed=spec['validation_seeds' if phase=='validation' else 'evaluation_seeds'][si]
    graph={};positions={};initial={};success={};forces={};provenance={}
    for name in ['fm','gaga']:
        path=folder/name/'generation/generation.json';report=json.loads(path.read_text());assert report['complete'] and report['protocol_sha256']==ph
        info=heads[str(si)][name];assert report['model_state_sha256']==info['model_state_sha256'] and report['head_state_sha256']==info['head_state_sha256']
        assert len(report['rows'])==len(conditions)*len(strengths[name]);provenance[str(path.relative_to(root))]=sha(path)
        for ai,alpha in enumerate(strengths[name]):
            method=f'{name}_a{ai}';graph[method]=np.zeros((len(conditions),count),bool);success[method]=np.zeros_like(graph[method]);forces[method]=np.full(graph[method].shape,np.inf)
            for i,c in enumerate(conditions):
                r=report['rows'][ai*len(conditions)+i];assert r['method']==method and r['strength']==alpha and r['condition_index']==i
                file=path.parent/r['file'];assert sha(file)==r['sha256'];s=torch.load(file,map_location='cpu',weights_only=False)
                assert s['condition']==c and s['seed']==seed and s['protocol_sha256']==ph and s['strength']==alpha
                assert s['model_state_sha256']==info['model_state_sha256'] and s['head_state_sha256']==info['head_state_sha256']
                batches=(count+spec['evaluation_batch']-1)//spec['evaluation_batch'];headcalls=64 if name=='fm' else 128
                assert s['core_calls']==128*batches and s['head_calls']==headcalls*batches and s['backbone_calls_per_trajectory']==128 and s['head_calls_per_trajectory']==headcalls
                x=s['positions'];assert x.shape==(count,c['n_atoms'],3) and torch.isfinite(x).all()
                quality=assess(x,c,list(range(count)))
                for key,value in quality.items():assert value==r[key],(method,i,key)
                graph[method][i]=[v['graph_supported'] for v in quality['records']];positions[method,i]=x.numpy();initial[method,i]=s['initial_positions'].numpy()
                if ai:np.testing.assert_array_equal(initial[method,i],initial[f'{name}_a0',i])
    xp=folder/'xtb';report=json.loads((xp/'results.json').read_text());assert report['complete'] and report['protocol_sha256']==ph and sha(xp/'tasks.json')==report['tasks_sha256']
    taskfile=json.loads((xp/'tasks.json').read_text());tasks={r['task']['task_id']:r for r in taskfile['tasks']}
    for file,digest in taskfile['sources'].items():assert sha(file)==digest
    expected=len(graph)*len(conditions)*count
    assert len(tasks)==len(report['rows'])==report['new_gfn2_attempts']==expected and len({r['task_id'] for r in report['rows']})==expected
    for row in report['rows']:
        record=tasks[row['task_id']];t=record['task'];method=t['method'];i,j=t['condition_index'],t['sample_index'];c=conditions[i]
        assert record['condition']==c and row['original_charge']==c['charge'] and row['original_spin_multiplicity']==c['spin_multiplicity'] and t['replica']==si
        np.testing.assert_array_equal(t['positions'],positions[method,i][j]);d=xp/'details'/row['task_id']
        for file,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(d/file)==row[key]
        if row['gradient_sha256']:assert sha(d/'gradient')==row['gradient_sha256']
        np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in (d/'input.xyz').read_text().splitlines()[2:]],t['positions'],atol=1e-12,rtol=0)
        assert row['command'][-1]=='--grad' and '--opt' not in row['command'] and row['command'][0]==spec['xtb_binary']
        if row['returncode'] is not None:
            gradient=(d/'gradient').read_text() if (d/'gradient').exists() else ''
            repeated=parse_singlepoint((d/'stdout.txt').read_text(),(d/'stderr.txt').read_text(),row['returncode'],gradient,c['n_atoms']);assert repeated['success']==row['success']
            if row['success']:np.testing.assert_array_equal(repeated['force_eV_A'],row['force_eV_A']);assert repeated['energy_eV']==row['energy_eV']
        if row['success']:success[method][i,j]=True;forces[method][i,j]=np.sqrt(np.square(row['force_eV_A']).sum(-1).mean())
        assert row['graph']==bool(graph[method][i,j]) and row['joint']==bool(graph[method][i,j] and success[method][i,j] and forces[method][i,j]<=5)
    done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['phase']==phase
    assert done['new_neural_outputs']==done['new_gfn2_attempts']==expected
    provenance[str((xp/'results.json').relative_to(root))]=sha(xp/'results.json')
    return dict(graph=graph,success=success,force=forces),provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['select','audit'],required=True);p.add_argument('--selection',type=Path);a=p.parse_args();torch.set_num_threads(2)
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'] and not a.out.exists()
    if a.stage=='select':
        assert not (a.run/'test').exists();heads={str(si):{name:audit_fit(a.project,a.run/f'validation/s{si}',spec,ph,si,name) for name in ['fm','gaga']} for si in [0,1]}
        strengths={name:spec['strengths'] for name in ['fm','gaga']};phase='validation'
    else:
        selection=json.loads(a.selection.read_text());assert selection['complete'] and selection['protocol_sha256']==ph;heads=selection['heads']
        for file,digest in selection['validation_provenance'].items():assert sha(a.project/file)==digest
        strengths={name:[0.,selection['strengths'][name]] for name in ['fm','gaga']};phase='test'
    perseed=[];provenance={}
    for si in [0,1]:
        result,origin=audit_outputs(a.project,a.run/f'{phase}/s{si}',spec,ph,si,phase,heads,strengths);perseed.append(result);provenance.update(origin)
    methods=list(perseed[0]['graph']);arrays={key:np.stack([[r[key][m] for m in methods] for r in perseed]) for key in ['graph','success','force']}
    joint=arrays['graph']&arrays['success']&(arrays['force']<=5)
    summary={m:dict(graph_rate=float(arrays['graph'][:,mi].mean()),joint_rate=float(joint[:,mi].mean()),joint_counts_by_seed=joint[:,mi].sum((1,2)).tolist(),
        joint_by_seed=joint[:,mi].mean((1,2)).tolist(),attempted=int(joint[:,mi].size),failures=int((~arrays['success'][:,mi]).sum())) for mi,m in enumerate(methods)}
    if a.stage=='select':
        chosen={name:max(range(len(spec['strengths'])),key=lambda ai:(summary[f'{name}_a{ai}']['joint_rate'],-spec['strengths'][ai])) for name in ['fm','gaga']}
        value=dict(complete=True,protocol_sha256=ph,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),heads=heads,summary=summary,
            strengths={name:spec['strengths'][i] for name,i in chosen.items()},validation_provenance=provenance,fresh_test_outputs_used=False,
            new_fit_outputs=1024,new_neural_outputs=2048,new_esen_queries=4096,new_optimizer_steps=80000,new_gfn2_attempts=2048)
    else:
        rng=np.random.default_rng(60081);means={m:joint[:,mi].mean(-1) for mi,m in enumerate(methods)};contrasts={}
        for name,left,right in [('fm_improvement','fm_a1','fm_a0'),('gaga_improvement','gaga_a1','gaga_a0'),('base_fm_minus_gaga','fm_a0','gaga_a0'),('corrected_fm_minus_gaga','fm_a1','gaga_a1')]:
            contrasts[name]=bootstrap(means[left]-means[right],rng,20000)
        contrasts['difference_in_improvements']=bootstrap(means['fm_a1']-means['fm_a0']-means['gaga_a1']+means['gaga_a0'],rng,20000)
        positive=lambda key:contrasts[key]['ci95'][0]>0 and min(contrasts[key]['by_seed'])>0
        arrayfile=a.out.with_suffix('.npz');np.savez_compressed(arrayfile,**arrays)
        value=dict(complete=True,protocol_sha256=ph,selection_sha256=sha(a.selection),methods=methods,summary=summary,contrasts=contrasts,
            primary_gate=positive('fm_improvement'),corrected_fm_over_gaga_gate=positive('corrected_fm_minus_gaga'),provenance=provenance,
            arrays_sha256=sha(arrayfile),new_neural_outputs=2048,new_gfn2_attempts=2048,new_esen_queries=0,new_optimizer_steps=0,
            scope=spec['scope'],interval_scope='Paired composition bootstrap conditional on two independently trained parent/head fits; strength chosen on separate validation.')
    write(a.out,value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
