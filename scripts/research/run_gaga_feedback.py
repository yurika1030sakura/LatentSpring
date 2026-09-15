#!/usr/bin/env python3
"""Train feedback challengers and evaluate fixed, explicitly costed raw samplers."""
import argparse
import copy
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch

from cfm_mol import matched_egnn as base
from cfm_mol import connectivity_feedback as feedback
from scripts.research.audit_generator_output_support import assess
from scripts.research.run_matched_generators import batches,write
from scripts.research.train_electronic_fm import sha
from scripts.research.tree_prior_fm import geometry_counts


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model,source,spec,context,rows,seed,count,out,label):
    out.mkdir(parents=True,exist_ok=True)
    model.eval()
    digest=base.state_hash(model)
    settings=dict(kind=spec['kind'],context=spec.get('context'),two_pass=spec.get('two_pass',False),
        gaga_max_t=spec['gaga_max_t'],calls=128,seed=seed,count=count,batch=spec['evaluation_batch'])
    report_path=out/(label+'_results.json')
    if report_path.exists():
        report=json.loads(report_path.read_text())
        assert report['complete'] and report['settings']==settings and report['model_state_sha256']==digest
        assert [r['condition'] for r in report['rows']]==[dict(c,numbers=c['atomic_numbers']) for c in rows]
        for r in report['rows']:assert sha(out/f'{label}_c{r["condition_index"]}.pt')==r['sample_sha256']
        return report
    results=[]
    for index,c in enumerate(rows):
        c=dict(c,numbers=c['atomic_numbers']);path=out/f'{label}_c{index}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['model_state_sha256']==digest and saved['condition']==c and saved['settings']==settings
        else:
            positions=[];initial=[];tick=time.perf_counter()
            for begin in range(0,count,spec['evaluation_batch']):
                size=min(spec['evaluation_batch'],count-begin)
                rng_seed=seed*1000003+index*100003+begin
                if context is None:
                    x,x0=base.sample(model,c['numbers'],spec['kind'],spec,source,rng_seed,size,128)
                else:
                    x,x0=feedback.sample(model,c['numbers'],spec,source,context,rng_seed,size,128)
                if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite generated coordinates')
                positions.append(x.cpu().double());initial.append(x0.cpu().double())
            torch.cuda.synchronize()
            saved=dict(positions=torch.cat(positions),initial_positions=torch.cat(initial),condition=c,
                model_state_sha256=digest,settings=settings,generation_seconds=time.perf_counter()-tick)
            atomic_save(saved,path)
        x=saved['positions'];assert len(x)==count
        result=dict(condition_index=index,condition=c,sample_sha256=sha(path),
            generation_seconds=saved['generation_seconds'],final_geometry=geometry_counts(x,c['numbers']),
            **assess(x,c,list(range(count))))
        results.append(result)
        print(json.dumps(dict(label=label,index=index,graph=result['graph_supported'],attempted=count)),flush=True)
    report=dict(complete=True,settings=settings,model_state_sha256=digest,rows=results,
        attempted=sum(r['attempted'] for r in results),graph_supported=sum(r['graph_supported'] for r in results),
        raw_unoptimized=True,new_physical_queries=0)
    write(report_path,report)
    return report


def train(args,spec,data,panel):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    spec_hash=sha(args.protocol)
    model=feedback.install(base.initialize(spec,'cuda')).train()
    initial=base.state_hash(model)
    assert sum(p.numel() for p in model.parameters())==2381566
    assert model.norm_values[0]==1.
    original=args.project/f'runs/matched_generators_v1/training/s{args.seed}/harmonic_fm/initialization.json'
    assert json.loads(original.read_text())['state_sha256']==initial
    source=base.HarmonicSource(spec['edge_log_width'])
    context=feedback.GeometryContext(source,spec['context'],spec['tree_regularization'])
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],amsgrad=True,weight_decay=1e-12)
    # Generate the original full schedule before slicing: draws must match its prefix.
    schedule=batches(data['training'],spec['batch_schedule_total_steps'],spec['batch_size'],spec['batch_seed'])[:spec['training_steps']]
    original_schedule=np.load(args.project/f'runs/matched_generators_v1/training/s{args.seed}/harmonic_fm/batch_indices.npy')
    np.testing.assert_array_equal(schedule,original_schedule[:len(schedule)])
    if (out/'batch_indices.npy').exists():np.testing.assert_array_equal(np.load(out/'batch_indices.npy'),schedule)
    else:np.save(out/'batch_indices.npy',schedule)
    initialization=dict(protocol_sha256=spec_hash,state_sha256=initial,parameter_count=2381566,
        batch_schedule_sha256=sha(out/'batch_indices.npy'),batch_schedule_is_original_prefix=True,pretrained=False)
    if (out/'initialization.json').exists():assert json.loads((out/'initialization.json').read_text())==initialization
    else:write(out/'initialization.json',initialization)
    start_step=0;seconds=0.
    if (out/'last.ckpt').exists():
        saved=torch.load(out/'last.ckpt',map_location='cuda',weights_only=False)
        assert saved['protocol_sha256']==spec_hash and saved['initial_state_sha256']==initial
        model.load_state_dict(saved['state_dict'],strict=True);ema.load_state_dict(saved['ema_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start_step=saved['global_step'];seconds=saved['training_seconds']
        del saved
    attempts=out/'attempts';attempts.mkdir(exist_ok=True);attempt=len(list(attempts.glob('*.json')))
    write(attempts/f'{attempt:03d}.json',dict(start_step=start_step,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    metrics=out/f'metrics_attempt{attempt:03d}.jsonl'
    for step,indices in enumerate(schedule[start_step:],start_step+1):
        tick=time.perf_counter();rows=[data['training'][int(i)] for i in indices]
        clean=torch.stack([r['positions'] for r in rows]).float().cuda()
        numbers=torch.tensor([r['condition']['numbers'] for r in rows],device='cuda')
        objective=feedback.loss(model,clean,numbers,spec,source,context,spec['noise_seed']*1000003+step)
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite objective')
        optimizer.zero_grad(set_to_none=True);objective.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        decay=min(spec['ema_decay'],(1+step)/(10+step))
        with torch.no_grad():
            for average,value in zip(ema.parameters(),model.parameters()):average.lerp_(value,1-decay)
        torch.cuda.synchronize();seconds+=time.perf_counter()-tick
        if step==1 or step%250==0:
            record=dict(step=step,loss=float(objective.detach()),gradient_norm=float(norm),training_seconds=seconds,
                examples_seen=step*spec['batch_size'],backbone_example_passes=2*step*spec['batch_size'])
            with metrics.open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%spec['checkpoint_every']==0 or step==spec['training_steps']:
            atomic_save(dict(state_dict=model.state_dict(),ema_state_dict=ema.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),global_step=step,protocol=spec,
                protocol_sha256=spec_hash,initial_state_sha256=initial,training_seconds=seconds),out/'last.ckpt')
    write(out/'training.json',dict(complete=True,steps=spec['training_steps'],training_seconds=seconds,
        examples_seen=spec['training_steps']*spec['batch_size'],backbone_example_passes=2*spec['training_steps']*spec['batch_size'],
        parameter_count=2381566,checkpoint_sha256=sha(out/'last.ckpt'),initial_state_sha256=initial,
        no_equal_wall_time_claim=True))
    evaluate(ema,source,spec,context,panel['validation_rows'],spec['validation_seed'],spec['validation_samples'],out/'validation',spec['context'])
    write(out/'complete.json',dict(complete=True,protocol_sha256=spec_hash,validation_outputs=512,new_physical_queries=0))


def baselines(args,spec,panel):
    source=base.HarmonicSource(spec['edge_log_width'])
    for kind in ['harmonic_fm','gaga']:
        saved_spec=json.loads((args.project/f'research/evidence/matched_generators_{kind}_s{args.seed}_v1.json').read_text())
        path=args.project/spec['baselines'][kind]['path'];assert sha(path)==spec['baselines'][kind]['sha256']
        model=base.initialize(saved_spec,'cuda')
        saved=torch.load(path,map_location='cuda',weights_only=False)
        assert saved['global_step']==30000
        model.load_state_dict(saved['ema_state_dict'],strict=True);del saved
        saved_spec['evaluation_batch']=spec['evaluation_batch']
        for maximum in ([350,500,650] if kind=='gaga' else [650]):
            saved_spec['gaga_max_t']=maximum
            label=f'gaga_{maximum}' if kind=='gaga' else 'harmonic_fm'
            evaluate(model,source,saved_spec,None,panel['validation_rows'],spec['validation_seed'],spec['validation_samples'],args.out,label)
        del model
    write(args.out/'complete.json',dict(complete=True,protocol_sha256=sha(args.protocol),validation_outputs=2048,new_physical_queries=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--stage',choices=['train','baselines'],required=True)
    args=parser.parse_args();spec=json.loads(args.protocol.read_text());assert spec['frozen']
    torch.set_num_threads(2)
    for key in ['data','panel']:assert sha(args.project/spec[key])==spec[key+'_sha256']
    panel=json.loads((args.project/spec['panel']).read_text())
    data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)
    train_keys={r['condition']['composition_hex'] for r in data['training']}
    val_keys={r['condition']['composition_hex'] for r in data['validation']}
    test_keys={r['composition_hex'] for r in panel['test_rows']}
    assert not train_keys&val_keys and not test_keys&(train_keys|val_keys)
    assert {r['composition_hex'] for r in panel['validation_rows']}<=val_keys
    assert len(data['training'])==20000 and len(panel['validation_rows'])==32 and len(test_keys)==32
    assert args.seed in [0,1] and spec['initialization_seed']==40401+args.seed
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'complete.json').exists():
        assert json.loads((args.out/'complete.json').read_text())['protocol_sha256']==sha(args.protocol)
        return
    if args.stage=='train':train(args,spec,data,panel)
    else:baselines(args,spec,panel)


if __name__=='__main__':
    main()
