"""Collect native TRAIN states, fit bounded heads, and score separate validation."""
import argparse,gc,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base,connectivity_feedback as feedback
from cfm_mol.matched_physical_connection import PhysicalFieldTransform,endpoint_and_progress
from cfm_mol.physical_connection import make_physical_connection
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.trajectory_physical_teacher import velocity_target
from scripts.research.connection_capacity import cache_rows,evaluate as evaluate_head
from scripts.research.run_matched_physical import make_model,make_context
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_fresh_primary_xtb import run_task


def sample(model,arm,source,context,c,seed,batch,calls,transform=None):
    s=arm['spec'];numbers=c['numbers']
    if context is None:return base.sample(model,numbers,s['kind'],s,source,seed,batch,calls,field_transform=transform)
    return feedback.sample(model,numbers,s,source,context,seed,batch,calls,field_transform=transform)


def load_parent(root,arm):
    file=root/arm['checkpoint'];assert sha(file)==arm['checkpoint_sha256']
    state=torch.load(file,map_location='cpu',weights_only=False)['ema_state_dict']
    return make_model(arm,state).eval().requires_grad_(False)


def teacher(root,spec,ph,si,name,model,source,context,data,out):
    out.mkdir(parents=True,exist_ok=False);(out/'records').mkdir();rows=[];arm=spec['parents'][si][name]
    parent_hash=base.state_hash(model);oc=spec['oracle'];worker=Path(__file__).resolve().parent/'oracle_worker.py'
    assert sha(worker)==oc['oracle_worker_sha256'] and sha(oc['oracle_checkpoint'])==oc['oracle_sha256']
    c=data[spec['training_rows'][0]]['condition'];start=time.perf_counter();generation_seconds=0.
    with EnergyOracle(oc['oracle_interpreter'],worker,oc['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],
            spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32'];write(out/'handshake.json',oracle.handshake)
        for slot,index in enumerate(spec['training_rows']):
            c=data[index]['condition'];observed=[];counter=[0];core=[0]
            def observe(x,t,z,value):
                call=counter[0];counter[0]+=1
                if call in spec['capture_calls'][name]:
                    h,progress=endpoint_and_progress(model,x,t,value,arm['spec'])
                    observed.append(dict(call=call,x=x.cpu().clone(),native_t=t.cpu().clone(),value=value.cpu().clone(),endpoint=h.cpu(),progress=progress.cpu()))
                return value
            handle=model.dynamics.egnn.register_forward_hook(lambda *_:core.__setitem__(0,core[0]+1))
            seed=spec['trajectory_seeds'][si]*1000003+slot*100003;tick=time.perf_counter()
            final,initial=sample(model,arm,source,context,c,seed,spec['training_draws'],spec['backbone_calls'],observe)
            torch.cuda.synchronize();generation_seconds+=time.perf_counter()-tick;handle.remove()
            assert core[0]==128 and counter[0]==(64 if name=='fm' else 128) and len(observed)==2
            x=torch.cat([r['x'] for r in observed]);h=torch.cat([r['endpoint'] for r in observed]);progress=torch.cat([r['progress'] for r in observed]);count=len(h)
            oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity']);before=oracle.evaluated
            energy,force=oracle.evaluate_chunked(torch.cat([h.double(),-h.double()]),max_request=16)
            even_force=(force[:count]-force[count:])/2;even_force-=even_force.mean(1,keepdim=True)
            conf=spec['target'];norm=even_force.flatten(1).norm(dim=-1)
            sigma=torch.minimum(torch.full_like(norm,conf['max_sigma']),torch.sqrt(conf['max_shift']*conf['kT']/norm.clamp_min(1e-12)))
            shift=sigma[:,None,None].square()*even_force/conf['kT']
            target,cap=velocity_target(shift,progress.double(),velocity_scale=spec['head_configuration']['velocity_scale'],gate_power=spec['head_configuration']['gate_power'])
            assert torch.isfinite(target).all()
            file=out/'records'/f'c{slot}.pt'
            atomic_save(dict(protocol_sha256=ph,parent_state_sha256=parent_hash,condition=c,training_row=index,source_seed=seed,
                initial_positions=initial.cpu(),final_positions=final.cpu(),observed=observed,anchor_raw_energy=energy,anchor_raw_force=force,
                centered_even_force=even_force,sigma=sigma,shift=shift,velocity_target=target,cap=cap,oracle_queries=oracle.evaluated-before),file)
            digest=sha(file)
            for k in range(count):rows.append(dict(condition=c,training_row=index,composition_slot=slot,local_state=k,x=x[k],endpoint=h[k],t=progress[k],force=target[k].float(),record_sha256=digest))
            report=dict(compositions=slot+1,states=len(rows),queries=oracle.evaluated,seconds=time.perf_counter()-start)
            write(out/'progress.json',report)
            if slot%8==7:print(json.dumps(dict(phase='teacher',arm=name,seed=si,**report)),flush=True)
        queries=oracle.evaluated;assert queries==oracle.requested_evaluations==1024
    assert base.state_hash(model)==parent_hash and not any(p.requires_grad for p in model.parameters())
    atomic_save(dict(protocol_sha256=ph,parent_state_sha256=parent_hash,rows=rows),out/'bank.pt')
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,parent_state_sha256=parent_hash,bank_sha256=sha(out/'bank.pt'),
        new_fit_outputs=256,new_esen_queries=queries,states=len(rows),generation_seconds=generation_seconds,seconds=time.perf_counter()-start))
    return rows


def fit(spec,ph,si,rows,out):
    out.mkdir(parents=True,exist_ok=False);seed=spec['head_seeds'][si];torch.manual_seed(seed)
    config=spec['head_configuration'];head=make_physical_connection(**config).cuda().float().train();cache=cache_rows(rows,config,'cuda')
    val=[i for i,r in enumerate(rows) if r['composition_slot'] in spec['validation_teacher_slots']];train=[i for i in range(len(rows)) if i not in val]
    assert len(train)==384 and len(val)==128 and not {rows[i]['condition']['composition_hex'] for i in train}&{rows[i]['condition']['composition_hex'] for i in val}
    optimizer=torch.optim.AdamW(head.parameters(),lr=spec['learning_rate'],weight_decay=1e-12);rng=np.random.default_rng(seed);history=[];tick=time.perf_counter()
    with (out/'metrics.jsonl').open('w') as f:
        for step in range(1,spec['steps']+1):
            index=train[int(rng.integers(len(train)))];*inputs,target=cache[index];prediction=head(*inputs)
            objective=(prediction-target).square().sum(-1).mean()
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite head loss')
            optimizer.zero_grad(set_to_none=True);objective.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            f.write(json.dumps(dict(step=step,bank_row=index,loss=float(objective.detach()),gradient_norm=float(norm)))+'\n')
            if step in spec['diagnostic_steps']:
                record=dict(step=step,fit=evaluate_head(head,cache,train),validation=evaluate_head(head,cache,val),seconds=time.perf_counter()-tick)
                history.append(record);f.flush();print(json.dumps(dict(phase='head_fit',seed=si,**record)),flush=True)
                atomic_save(dict(state_dict=head.state_dict(),configuration=config,protocol_sha256=ph,seed=seed,step=step),out/f'step_{step}.pt')
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,seed=seed,history=history,fit_states=train,validation_states=val,
        parameter_count=sum(p.numel() for p in head.parameters()),new_optimizer_steps=spec['steps'],checkpoint_sha256=sha(out/f'step_{spec["steps"]}.pt')))
    return head.eval()


def generate(spec,ph,si,name,model,source,context,head,rows,seed,count,strengths,out):
    out.mkdir(parents=True,exist_ok=False);report=[];arm=spec['parents'][si][name];parent_hash=base.state_hash(model)
    for ai,strength in enumerate(strengths):
        label=f'{name}_a{ai}';transform=PhysicalFieldTransform(model,arm['spec'],head,strength)
        for i,c in enumerate(rows):
            positions=[];initial=[];core=[0];before=transform.calls;head_before=head.forward_calls;tick=time.perf_counter()
            handle=model.dynamics.egnn.register_forward_hook(lambda *_:core.__setitem__(0,core[0]+1))
            for begin in range(0,count,spec['evaluation_batch']):
                n=min(spec['evaluation_batch'],count-begin);local_seed=seed*1000003+i*100003+begin
                x,x0=sample(model,arm,source,context,c,local_seed,n,128,transform)
                positions.append(x.cpu().double());initial.append(x0.cpu().double())
            torch.cuda.synchronize();seconds=time.perf_counter()-tick;handle.remove()
            batches=(count+spec['evaluation_batch']-1)//spec['evaluation_batch'];expected=batches*(64 if name=='fm' else 128)
            assert core[0]==128*batches and transform.calls-before==head.forward_calls-head_before==expected
            x=torch.cat(positions);file=out/f'{label}_c{i}.pt'
            atomic_save(dict(positions=x,initial_positions=torch.cat(initial),condition=c,strength=strength,protocol_sha256=ph,
                model_state_sha256=parent_hash,head_state_sha256=base.state_hash(head),seed=seed,core_calls=core[0],head_calls=expected,
                backbone_calls_per_trajectory=128,head_calls_per_trajectory=64 if name=='fm' else 128,generation_seconds=seconds),file)
            quality=assess(x,c,list(range(count)))
            report.append(dict(method=label,strength=strength,condition_index=i,file=file.name,sha256=sha(file),**quality))
            print(json.dumps(dict(phase='generation',arm=name,strength=strength,condition=i,graph=quality['graph_supported'],attempted=count)),flush=True)
    assert base.state_hash(model)==parent_hash
    value=dict(complete=True,protocol_sha256=ph,rows=report,new_neural_outputs=len(rows)*count*len(strengths),
        model_state_sha256=parent_hash,head_state_sha256=base.state_hash(head),new_oracle_queries=0)
    write(out/'generation.json',value);return value


def score(spec,ph,si,reports,out):
    out.mkdir(parents=True,exist_ok=False);binary=Path(spec['xtb_binary']);assert sha(binary)==spec['xtb_binary_sha256'];tasks=[];sources={};graphs={}
    for folder,report in reports:
        assert report['complete'] and report['protocol_sha256']==ph
        for r in report['rows']:
            file=folder/r['file'];assert sha(file)==r['sha256'];sources[str(file)]=r['sha256']
            saved=torch.load(file,map_location='cpu',weights_only=False);c=saved['condition'];i=r['condition_index'];method=r['method']
            graphs[method,i]=[v['graph_supported'] for v in r['records']]
            for j,x in enumerate(saved['positions']):
                task=dict(task_id=f'{method}_c{i}_s{j}',method=method,replica=si,parent_id=i*len(saved['positions'])+j,
                    inversion_check=False,positions=x.tolist(),condition_index=i,sample_index=j)
                tasks.append((task,c))
    write(out/'tasks.json',dict(protocol_sha256=ph,tasks=[dict(task=t,condition=c) for t,c in tasks],sources=sources));results=[]
    settings=dict(accuracy=.1,timeout_seconds=90,restraint_eV_A2=0.)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(run_task,t,c,binary,out,settings) for t,c in tasks]):
            results.append(future.result())
            if len(results)%128==0:
                write(out/'progress.json',dict(completed=len(results),attempted=len(tasks)))
                print(json.dumps(dict(phase='gfn2',completed=len(results),attempted=len(tasks))),flush=True)
    mapping={t['task_id']:t for t,c in tasks};summary={}
    for row in results:
        t=mapping[row['task_id']];method=t['method'];i,j=t['condition_index'],t['sample_index']
        force=float(np.sqrt(np.square(row['force_eV_A']).sum(-1).mean())) if row['success'] else None
        graph=graphs[method,i][j];row.update(condition_index=i,sample_index=j,graph=graph,rms_force=force,joint=bool(graph and row['success'] and force<=5))
    for method in sorted({t['method'] for t,c in tasks}):
        r=[v for v in results if v['method']==method];summary[method]=dict(attempted=len(r),graph=sum(v['graph'] for v in r),
            joint=sum(v['joint'] for v in r),failures=sum(not v['success'] for v in r))
    write(out/'results.json',dict(complete=True,protocol_sha256=ph,tasks_sha256=sha(out/'tasks.json'),rows=sorted(results,key=lambda r:r['task_id']),summary=summary,new_gfn2_attempts=len(results)))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);p.add_argument('--selection',type=Path)
    a=p.parse_args();torch.set_num_threads(2);spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);si=a.seed_index
    assert spec['frozen'];a.out.mkdir(parents=True,exist_ok=False);source=base.HarmonicSource();reports=[];tick=time.perf_counter()
    if a.selection:
        selection=json.loads(a.selection.read_text());assert selection['complete'] and selection['protocol_sha256']==ph
        phase='test';rows=spec['test_rows'];seed=spec['evaluation_seeds'][si];count=spec['test_samples']
    else:
        phase='validation';rows=spec['validation_rows'];seed=spec['validation_seeds'][si];count=spec['validation_samples']
        assert sha(a.project/spec['data'])==spec['data_sha256'];data=torch.load(a.project/spec['data'],map_location='cpu',weights_only=False)['training']
    for name,arm in spec['parents'][si].items():
        model=load_parent(a.project,arm);context=make_context(arm,source)
        if not a.selection:
            bank=teacher(a.project,spec,ph,si,name,model,source,context,data,a.out/name/'teacher')
            head=fit(spec,ph,si,bank,a.out/name/'fit');strengths=spec['strengths']
        else:
            info=selection['heads'][str(si)][name];file=a.project/info['path'];assert sha(file)==info['sha256']
            saved=torch.load(file,map_location='cpu',weights_only=False);assert saved['protocol_sha256']==ph
            head=make_physical_connection(**saved['configuration']).cuda().float();head.load_state_dict(saved['state_dict'],strict=True);head.eval()
            strengths=[0.,selection['strengths'][name]]
        folder=a.out/name/'generation';report=generate(spec,ph,si,name,model,source,context,head,rows,seed,count,strengths,folder);reports.append((folder,report))
        del model,head,context;gc.collect();torch.cuda.empty_cache()
    summary=score(spec,ph,si,reports,a.out/'xtb')
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,phase=phase,seed_index=si,summary=summary,
        generation_reports=[dict(path=str(f/'generation.json'),sha256=sha(f/'generation.json')) for f,r in reports],
        new_neural_outputs=sum(r['new_neural_outputs'] for f,r in reports),new_fit_outputs=0 if a.selection else 512,
        new_esen_queries=0 if a.selection else 2048,new_optimizer_steps=0 if a.selection else 40000,
        new_gfn2_attempts=sum(v['attempted'] for v in summary.values()),seconds=time.perf_counter()-tick,
        selection_sha256=sha(a.selection) if a.selection else None))


if __name__=='__main__':main()
