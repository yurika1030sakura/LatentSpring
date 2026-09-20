"""Reconstruct targets, head fits and raw validation before selecting a model."""
import argparse,datetime,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.physical_connection import make_physical_connection
from cfm_mol.atomwise_physical_connection import balanced_force_shift
from cfm_mol.trajectory_physical_teacher import velocity_target
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.connection_capacity import cache_rows,evaluate
from scripts.research.audit_generator_output_support import assess
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def check_targets(root,folder,spec,ph,si,name):
    targetfolder=folder/name/'targets';done=json.loads((targetfolder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
    origin=spec['banks'][si][name];assert sha(root/origin['path'])==done['source_bank_sha256']==origin['sha256']
    source=torch.load(root/origin['path'],map_location='cpu',weights_only=False)['rows'];banks={}
    for kind in ['global','balanced']:
        file=targetfolder/(kind+'.pt');assert sha(file)==done['banks'][kind];saved=torch.load(file,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==ph and saved['source_bank_sha256']==origin['sha256'] and saved['target']==kind and len(saved['rows'])==512
        banks[kind]=saved['rows']
    cfg=spec['head_configuration']
    for slot,index in enumerate(spec['training_rows']):
        file=root/origin['teacher_folder']/'records'/f'c{slot}.pt';digest=sha(file);r=torch.load(file,map_location='cpu',weights_only=False)
        assert r['protocol_sha256']==spec['original_protocol_sha256'] and r['training_row']==index
        shift,mobility,scale=balanced_force_shift(r['centered_even_force'],r['shift'],force_floor=spec['force_floor_eV_A'])
        t=torch.cat([v['progress'] for v in r['observed']]).double();target,cap=velocity_target(shift,t,velocity_scale=cfg['velocity_scale'],gate_power=cfg['gate_power']);assert (cap==1).all()
        torch.testing.assert_close(shift.flatten(1).norm(dim=-1),r['shift'].flatten(1).norm(dim=-1),atol=1e-12,rtol=1e-10)
        for k in range(4):
            i=slot*4+k;s=source[i];diagnostic=done['rows'][i]
            assert diagnostic['source_record_sha256']==digest and diagnostic['composition_slot']==slot and diagnostic['local_state']==k
            np.testing.assert_allclose(diagnostic['mobility'],mobility[k],atol=0,rtol=0);assert diagnostic['scale']==float(scale[k])
            assert diagnostic['force_dot_shift']==float((r['centered_even_force'][k]*shift[k]).sum()) and diagnostic['force_dot_shift']>=-1e-12
            for kind in banks:
                b=banks[kind][i];assert b['condition']==s['condition'] and b['training_row']==index and b['record_sha256']==digest
                for key in ['x','endpoint','t']:torch.testing.assert_close(b[key],s[key],atol=0,rtol=0)
                expected=s['force'] if kind=='global' else target[k].float();torch.testing.assert_close(b['force'],expected,atol=0,rtol=0)
    return banks


def check_fits(root,folder,spec,ph,si,name,heads,banks):
    result={};seed=spec['head_seeds'][si]
    for variant,definition in spec['variants'].items():
        info=heads[name][variant];file=root/info['path'];assert sha(file)==info['sha256'] and info['configuration']==definition['configuration'] and info['target']==definition['target']
        saved=torch.load(file,map_location='cpu',weights_only=False);model=make_physical_connection(**saved['configuration']).float().eval();model.load_state_dict(saved['state_dict'],strict=True)
        assert base.state_hash(model)==info['head_state_sha256'] and sum(p.numel() for p in model.parameters())==7106
        if definition['reuse']:
            assert info['sha256']==spec['reused_heads'][str(si)][name]['sha256'] and saved['protocol_sha256']==spec['original_protocol_sha256'];continue
        directory=folder/name/variant/'fit';done=json.loads((directory/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['seed']==seed
        rows=banks[definition['target']];train=[i for i,r in enumerate(rows) if r['composition_slot'] not in spec['validation_teacher_slots']];val=[i for i in range(512) if i not in train]
        assert done['fit_states']==train and done['validation_states']==val and done['new_optimizer_steps']==20000 and done['checkpoint_sha256']==info['sha256']
        cache=cache_rows(rows,definition['configuration'],'cpu');logs=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()];assert len(logs)==20000;rng=np.random.default_rng(seed)
        for step,row in enumerate(logs,1):assert row['step']==step and row['bank_row']==train[int(rng.integers(len(train)))] and np.isfinite([row['loss'],row['gradient_norm']]).all()
        assert [v['step'] for v in done['history']]==spec['diagnostic_steps']
        for point in done['history']:
            checkpoint=directory/f'step_{point["step"]}.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['configuration']==definition['configuration'] and saved['protocol_sha256']==ph and saved['seed']==seed and saved['step']==point['step']
            model.load_state_dict(saved['state_dict'],strict=True)
            for split,ids in [('fit',train),('validation',val)]:
                repeated=evaluate(model,cache,ids)
                for key,value in repeated.items():np.testing.assert_allclose(value,point[split][key],atol=4e-7,rtol=3e-5)
        result[variant]=done['history'][-1]
    return result


def check_outputs(root,folder,spec,ph,si,phase,heads,strengths):
    rows=spec['validation_rows' if phase=='validation' else 'test_rows'];count=spec['validation_samples' if phase=='validation' else 'test_samples'];seed=spec['validation_seeds' if phase=='validation' else 'evaluation_seeds'][si]
    arrays={};positions={};recipes={};sources={};initial={}
    for name in ['fm','gaga']:
        for variant,definition in spec['variants'].items():
            values=strengths[name][variant];path=folder/name/variant/'generation/generation.json'
            if not values:assert not path.exists();continue
            r=json.loads(path.read_text());assert r['complete'] and r['protocol_sha256']==ph and r['strengths']==values and r['variant']==variant and r['configuration']==definition['configuration']
            info=heads[name][variant];assert r['head_checkpoint_sha256']==info['sha256'] and r['head_state_sha256']==info['head_state_sha256']
            assert r['model_state_sha256']==spec['reused_heads'][str(si)][name]['model_state_sha256'];assert len(r['rows'])==len(values)*len(rows)
            sources[str(path.relative_to(root))]=sha(path)
            for ai,alpha in enumerate(values):
                method=f'{name}_{variant}_a{ai}';recipes[method]=dict(family=name,variant=variant,strength=alpha)
                arrays[method]=dict(graph=np.zeros((len(rows),count),bool),success=np.zeros((len(rows),count),bool),force=np.full((len(rows),count),np.inf))
                for i,c in enumerate(rows):
                    row=r['rows'][ai*len(rows)+i];assert row['method']==method and row['strength']==alpha and row['condition_index']==i
                    file=path.parent/row['file'];assert sha(file)==row['sha256'];saved=torch.load(file,map_location='cpu',weights_only=False)
                    assert saved['protocol_sha256']==ph and saved['strength']==alpha and saved['condition']==c and saved['seed']==seed
                    assert saved['head_state_sha256']==info['head_state_sha256'] and saved['model_state_sha256']==r['model_state_sha256']
                    batches=(count+spec['evaluation_batch']-1)//spec['evaluation_batch'];nc=64 if name=='fm' else 128
                    assert saved['core_calls']==128*batches and saved['head_calls']==nc*batches
                    x=saved['positions'];assert x.shape==(count,c['n_atoms'],3) and torch.isfinite(x).all();quality=assess(x,c,list(range(count)))
                    for key,value in quality.items():assert value==row[key],(method,i,key)
                    arrays[method]['graph'][i]=[v['graph_supported'] for v in quality['records']];positions[method,i]=x.numpy()
                    if (name,i) not in initial:initial[name,i]=saved['initial_positions']
                    else:torch.testing.assert_close(saved['initial_positions'],initial[name,i],atol=0,rtol=0)
    xp=folder/'xtb';r=json.loads((xp/'results.json').read_text());assert r['complete'] and r['protocol_sha256']==ph and sha(xp/'tasks.json')==r['tasks_sha256']
    records=json.loads((xp/'tasks.json').read_text());tasks={v['task']['task_id']:v for v in records['tasks']}
    for file,digest in records['sources'].items():assert sha(file)==digest
    expected=len(arrays)*len(rows)*count;assert len(tasks)==len(r['rows'])==len({v['task_id'] for v in r['rows']})==r['new_gfn2_attempts']==expected
    for row in r['rows']:
        record=tasks[row['task_id']];t=record['task'];i,j=t['condition_index'],t['sample_index'];method=t['method'];c=rows[i]
        assert record['condition']==c and row['original_charge']==c['charge'] and row['original_spin_multiplicity']==c['spin_multiplicity']
        np.testing.assert_array_equal(t['positions'],positions[method,i][j]);d=xp/'details'/row['task_id']
        for filename,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(d/filename)==row[key]
        if row['gradient_sha256']:assert sha(d/'gradient')==row['gradient_sha256']
        np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in (d/'input.xyz').read_text().splitlines()[2:]],t['positions'],atol=1e-12,rtol=0)
        assert row['command'][-1]=='--grad' and '--opt' not in row['command']
        if row['returncode'] is not None:
            gradient=(d/'gradient').read_text() if (d/'gradient').exists() else '';parsed=parse_singlepoint((d/'stdout.txt').read_text(),(d/'stderr.txt').read_text(),row['returncode'],gradient,c['n_atoms'])
            assert parsed['success']==row['success']
            if row['success']:np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A']);assert parsed['energy_eV']==row['energy_eV']
        if row['success']:arrays[method]['success'][i,j]=True;arrays[method]['force'][i,j]=np.sqrt(np.square(row['force_eV_A']).sum(-1).mean())
        assert row['graph']==bool(arrays[method]['graph'][i,j]) and row['joint']==bool(arrays[method]['graph'][i,j] and arrays[method]['success'][i,j] and arrays[method]['force'][i,j]<=5)
    done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['phase']==phase and done['new_neural_outputs']==done['new_gfn2_attempts']==expected
    sources[str((xp/'results.json').relative_to(root))]=sha(xp/'results.json')
    return arrays,recipes,sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['select','audit'],required=True);p.add_argument('--selection',type=Path);a=p.parse_args();a.project=a.project.resolve();a.run=a.run.resolve();torch.set_num_threads(2)
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'] and not a.out.exists();heads={};fits={};perseed=[];provenance={}
    if a.stage=='select':
        assert not (a.run/'test').exists();phase='validation';strengths={name:{v:spec['strengths'] if d['reuse'] else [s for s in spec['strengths'] if s>0] for v,d in spec['variants'].items()} for name in ['fm','gaga']}
    else:
        selection=json.loads(a.selection.read_text());assert selection['complete'] and selection['advance'] and selection['protocol_sha256']==ph
        for file,digest in selection['validation_provenance'].items():assert sha(a.project/file)==digest
        phase='test';strengths=selection['test_strengths'];heads=selection['heads']
    for si in [0,1]:
        folder=a.run/phase/f's{si}';done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
        if a.stage=='select':
            assert done['new_optimizer_steps']==120000;heads[str(si)]=done['heads'];fits[str(si)]={}
            for name in ['fm','gaga']:
                banks=check_targets(a.project,folder,spec,ph,si,name);fits[str(si)][name]=check_fits(a.project,folder,spec,ph,si,name,done['heads'],banks)
        arrays,recipes,origin=check_outputs(a.project,folder,spec,ph,si,phase,heads[str(si)],strengths);perseed.append(arrays);provenance.update(origin)
    methods=list(perseed[0]);array={key:np.stack([[r[m][key] for m in methods] for r in perseed]) for key in ['graph','success','force']};joint=array['graph']&array['success']&(array['force']<=5)
    summary={m:dict(**recipes[m],attempted=int(joint[:,mi].size),graph_rate=float(array['graph'][:,mi].mean()),joint_rate=float(joint[:,mi].mean()),
        graph_by_seed=array['graph'][:,mi].mean((1,2)).tolist(),joint_by_seed=joint[:,mi].mean((1,2)).tolist(),failures=int((~array['success'][:,mi]).sum())) for mi,m in enumerate(methods)}
    if a.stage=='select':
        order=list(spec['variants']);selected={};controls={};test_strengths={}
        key=lambda m:(summary[m]['joint_rate'],summary[m]['graph_rate'],-summary[m]['strength'],-order.index(summary[m]['variant']))
        for name in ['fm','gaga']:
            choices=[m for m in methods if summary[m]['family']==name];chosen=max(choices,key=key);control=max([m for m in choices if summary[m]['variant']=='pair_global'],key=key)
            selected[name]=dict(method=chosen,**summary[chosen]);controls[name]=dict(method=control,**summary[control]);alpha=summary[chosen]['strength']
            test_strengths[name]={v:([alpha] if alpha>0 or v=='pair_global' else []) for v in order}
            test_strengths[name]['pair_global']=sorted(set(test_strengths[name]['pair_global']+[summary[control]['strength']]))
        left,right=selected['fm'],controls['fm'];change=left['joint_rate']-right['joint_rate'];seedchange=np.array(left['joint_by_seed'])-right['joint_by_seed']
        advance=bool(left['variant']!='pair_global' and change>=.02-1e-12 and (seedchange>0).all() and left['graph_rate']>=right['graph_rate']-1e-12)
        value=dict(complete=True,protocol_sha256=ph,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=selected,controls=controls,
            advance=advance,validation_fm_gain=change,validation_fm_gain_by_seed=seedchange.tolist(),heads=heads,fits=fits,summary=summary,test_strengths=test_strengths,
            validation_provenance=provenance,fresh_test_outputs_used=False,new_neural_outputs=spec['expected_new_validation_outputs'],
            new_gfn2_attempts=spec['expected_new_validation_outputs'],new_optimizer_steps=240000,new_esen_queries=0)
    else:
        def matching(name,variant,strength):return next(m for m in methods if recipes[m]==dict(family=name,variant=variant,strength=strength))
        selected={name:matching(name,r['variant'],r['strength']) for name,r in selection['selected'].items()};controls={name:matching(name,'pair_global',r['strength']) for name,r in selection['controls'].items()}
        rng=np.random.default_rng(62081);means={m:joint[:,mi].mean(-1) for mi,m in enumerate(methods)};contrasts={}
        for key,left,right in [('fm_improvement',selected['fm'],controls['fm']),('gaga_improvement',selected['gaga'],controls['gaga']),('fm_minus_gaga',selected['fm'],selected['gaga'])]:contrasts[key]=bootstrap(means[left]-means[right],rng,20000)
        for name in ['fm','gaga']:
            for v in spec['variants']:
                alpha=selection['selected'][name]['strength']
                if alpha==0 and v!='pair_global':continue
                method=matching(name,v,alpha);contrasts[name+'_selected_minus_'+v]=bootstrap(means[selected[name]]-means[method],rng,20000)
        graphdiff=array['graph'][:,methods.index(selected['fm'])].mean(-1)-array['graph'][:,methods.index(selected['gaga'])].mean(-1)
        contrasts['fm_minus_gaga_graph']=bootstrap(graphdiff,rng,20000);positive=lambda k:contrasts[k]['ci95'][0]>0 and min(contrasts[k]['by_seed'])>0
        file=a.out.with_suffix('.npz');np.savez_compressed(file,**array)
        value=dict(complete=True,protocol_sha256=ph,selection_sha256=sha(a.selection),methods=methods,selected=selected,controls=controls,
            summary=summary,contrasts=contrasts,fm_improvement_gate=positive('fm_improvement'),fm_over_gaga_joint_gate=positive('fm_minus_gaga'),
            fm_over_gaga_graph_gate=positive('fm_minus_gaga_graph'),provenance=provenance,arrays_sha256=sha(file),
            new_neural_outputs=int(joint.size),new_gfn2_attempts=int(joint.size),new_optimizer_steps=0,new_esen_queries=0,scope=spec['scope'])
    write(a.out,value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
