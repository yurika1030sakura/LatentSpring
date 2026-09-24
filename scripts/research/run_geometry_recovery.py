"""Continue paired FM parents and evaluate every frozen geometry-recovery arm."""
import argparse
import copy
import datetime
import json
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base, connectivity_feedback as feedback
from cfm_mol.geometry_recovery import recovery_loss
from cfm_mol.chemical_geometry_review import molecule_from_coordinates, geometry_diagnostics, contact_diagnostics
from scripts.research.run_matched_connection import load_parent, make_context, generate, score
from scripts.research.run_matched_generators import batches, write
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.train_electronic_fm import sha


def review_geometry(folder, report, out):
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for entry in report['rows']:
            saved = torch.load(folder / entry['file'], map_location='cpu', weights_only=False)
            c = saved['condition']
            for j, (xyz, graph) in enumerate(zip(saved['positions'].numpy(), entry['records'])):
                row = dict(condition=entry['condition_index'],sample=j,graph=bool(graph['graph_supported']),smiles=graph.get('smiles'),
                    record_sha256=entry['sha256'],**contact_diagnostics(xyz,c['numbers']))
                if row['graph']:
                    try:
                        row.update(geometry_diagnostics(molecule_from_coordinates(xyz,c['numbers'],c['charge'])))
                        assert row['smiles']==graph['smiles']
                    except Exception as exc:
                        row.update(closed_shell_geometry_pass=False,error=type(exc).__name__+': '+str(exc))
                else:
                    row['closed_shell_geometry_pass']=False
                records.append(row)
    write(out,dict(complete=True,rows=records,attempted=len(records),
        graph=sum(r['graph'] for r in records),geometry=sum(r['closed_shell_geometry_pass'] for r in records),
        heavy_disconnected=sum(r['heavy_components']>1 for r in records)))
    return records


def evaluate(root, protocol, ph, fit, variant, model, source, context, folder):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        done=json.loads((folder/'complete.json').read_text());assert done['protocol_sha256']==ph
        assert done['model_state_sha256']==base.state_hash(model)
        return done
    spec=json.loads((root/protocol['parent_protocol']).read_text())
    assert sha(root/protocol['parent_protocol'])==protocol['parent_protocol_sha256']
    spec.update(strength_limit=4.,evaluation_batch=protocol['evaluation_batch'])
    head,_=restore_head(root,protocol['physical_heads'][fit]); head_hash=base.state_hash(head)
    conditions=[r['condition'] for r in protocol['conditions']]
    generation=folder/'generation'
    if (generation/'generation.json').exists():
        report=json.loads((generation/'generation.json').read_text())
        assert report['complete'] and report['protocol_sha256']==ph
        assert report['model_state_sha256']==base.state_hash(model) and report['head_state_sha256']==head_hash
        for row in report['rows']:assert sha(generation/row['file'])==row['sha256']
    else:
        report=generate(spec,ph,fit,'fm',model,source,context,head,conditions,
            protocol['evaluation_seeds'][fit],protocol['samples_per_condition'],[4.],generation,method_prefix=variant)
    geom_file=folder/'geometry.json'
    if not geom_file.exists():review_geometry(generation,report,geom_file)
    geometry=json.loads(geom_file.read_text());assert geometry['complete']
    physical=folder/'xtb'
    if not (physical/'results.json').exists():score(spec,ph,fit,[(generation,report)],physical)
    scores=json.loads((physical/'results.json').read_text());assert scores['complete']
    lookup={(r['condition_index'],r['sample_index']):r for r in scores['rows']}
    joint_geometry=sum(r['closed_shell_geometry_pass'] and lookup[r['condition'],r['sample']]['success']
        and lookup[r['condition'],r['sample']]['rms_force']<=5 for r in geometry['rows'])
    assert base.state_hash(head)==head_hash
    done=dict(complete=True,protocol_sha256=ph,fit=fit,variant=variant,
        model_state_sha256=base.state_hash(model),head_state_sha256=head_hash,
        generation_sha256=sha(generation/'generation.json'),geometry_sha256=sha(geom_file),
        physical_sha256=sha(physical/'results.json'),attempted=geometry['attempted'],
        graph=geometry['graph'],geometry=geometry['geometry'],geometry_force=joint_geometry,
        heavy_disconnected=geometry['heavy_disconnected'],physical_summary=scores['summary'],
        new_generation_outputs=geometry['attempted'],new_gfn2_attempts=geometry['attempted'],
        new_esen_queries=0,geometry_optimized=False)
    write(folder/'complete.json',done);print(json.dumps(dict(phase='evaluated',**done)),flush=True)
    return done


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=[0,1],required=True)
    p.add_argument('--variant',choices=['replay','recovery','recovery_local'],required=True)
    p.add_argument('--train-only',action='store_true')
    a=p.parse_args();root=a.project.resolve();out=a.out.resolve();protocol=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    assert protocol['frozen'];torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(protocol['noise_seeds'][a.fit]);out.mkdir(parents=True,exist_ok=True)
    datafile=root/protocol['data'];assert sha(datafile)==protocol['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False)['training']
    assert not {r['condition']['composition_hex'] for r in data}&{r['condition']['composition_hex'] for r in protocol['conditions']}
    arm=protocol['parents'][a.fit];model=load_parent(root,arm);initial=base.state_hash(model)
    source=base.HarmonicSource(arm['spec']['edge_log_width']);context=make_context(arm,source)
    if a.variant=='replay' and not a.train_only:evaluate(root,protocol,ph,a.fit,'frozen',model,source,context,out.parent/'frozen')
    model.train().requires_grad_(True);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],amsgrad=True,weight_decay=1e-12)
    schedule=batches(data,protocol['steps'],protocol['batch_size'],protocol['batch_seeds'][a.fit])
    indices_file=out/'batch_indices.npy'
    if indices_file.exists():np.testing.assert_array_equal(np.load(indices_file),schedule)
    else:np.save(indices_file,schedule)
    initialization=dict(protocol_sha256=ph,initial_state_sha256=initial,batch_schedule_sha256=sha(indices_file),
        fit=a.fit,variant=a.variant,parameters=sum(p.numel() for p in model.parameters()),
        device=torch.cuda.get_device_name(),physical_head=protocol['physical_heads'][a.fit])
    if not (out/'initialization.json').exists():write(out/'initialization.json',initialization)
    else:
        old=json.loads((out/'initialization.json').read_text())
        for key in ['protocol_sha256','initial_state_sha256','batch_schedule_sha256']:assert old[key]==initialization[key]
    start=0;seconds=0.
    if (out/'last.ckpt').exists():
        saved=torch.load(out/'last.ckpt',map_location='cuda',weights_only=False)
        assert saved['protocol_sha256']==ph and saved['initial_state_sha256']==initial
        model.load_state_dict(saved['state_dict']);ema.load_state_dict(saved['ema_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict']);start=saved['global_step'];seconds=saved['training_seconds'];del saved
    attempt=len(list(out.glob('metrics_attempt*.jsonl')))
    with (out/f'metrics_attempt{attempt}.jsonl').open('w') as stream:
        for step,indices in enumerate(schedule[start:],start+1):
            tick=time.perf_counter();rows=[data[int(i)] for i in indices]
            clean=torch.stack([r['positions'] for r in rows]).cuda().float()
            numbers=torch.tensor([r['condition']['numbers'] for r in rows],device='cuda')
            seed=protocol['noise_seeds'][a.fit]*1000003+step
            if step%2:
                objective=feedback.loss(model,clean,numbers,arm['spec'],source,context,seed);extra=dict(kind='original_cfm')
            else:
                objective,extra=recovery_loss(model,clean,numbers,arm['spec'],source,context,seed,
                    corrupt=a.variant!='replay',local_geometry=a.variant=='recovery_local');extra['kind']='endpoint_recovery'
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite recovery training loss')
            optimizer.zero_grad(set_to_none=True);objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            decay=min(protocol['ema_decay'],(1+step)/(10+step))
            with torch.no_grad():
                for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,1-decay)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            if step<=4 or step%100==0:
                entry=dict(step=step,loss=float(objective.detach()),gradient_norm=float(norm),training_seconds=seconds,**extra)
                stream.write(json.dumps(entry)+'\n');stream.flush();print(json.dumps(entry),flush=True)
            if step%protocol['checkpoint_every']==0 or step==protocol['steps']:
                atomic_save(dict(state_dict=model.state_dict(),ema_state_dict=ema.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                    global_step=step,protocol_sha256=ph,initial_state_sha256=initial,training_seconds=seconds),out/'last.ckpt')
    done=dict(complete=True,protocol_sha256=ph,fit=a.fit,variant=a.variant,steps=protocol['steps'],
        initial_state_sha256=initial,ema_state_sha256=base.state_hash(ema),checkpoint_sha256=sha(out/'last.ckpt'),
        training_seconds=seconds,training_forward_examples=protocol['steps']*protocol['batch_size']*2,
        new_optimizer_steps=protocol['steps'],new_esen_queries=0)
    write(out/'training.json',done)
    if a.train_only:return
    evaluate(root,protocol,ph,a.fit,a.variant,ema,source,context,out/'evaluation')
    write(out/'complete.json',dict(**done,evaluation_complete_sha256=sha(out/'evaluation/complete.json')))


if __name__=='__main__':main()
