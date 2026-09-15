#!/usr/bin/env python3
"""Give the selected FM and GAGA parents the same bounded physical adaptation."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch

from cfm_mol import matched_egnn as base
from cfm_mol import connectivity_feedback as feedback
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.escorted_thermal_teacher import make_escorted_teacher
from scripts.research.run_gaga_feedback import evaluate,atomic_save
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def make_model(arm,state):
    model=base.initialize(arm['spec'],'cuda')
    if arm['spec'].get('context')=='distance':feedback.install(model)
    model.load_state_dict(state,strict=True)
    assert sum(p.numel() for p in model.parameters())==2381566 and model.norm_values[0]==1.
    return model


def make_context(arm,source):
    return feedback.GeometryContext(source,'distance') if arm['spec'].get('context')=='distance' else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    assert spec['frozen'] and not args.out.exists()
    for key in ['data','panel','selection','parent_audit']:assert sha(args.project/spec[key])==spec[key+'_sha256']
    data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training']
    panel=json.loads((args.project/spec['panel']).read_text());test=panel['test_rows']
    teacher_conditions=[data[index]['condition'] for index in spec['teacher_rows']]
    assert len(teacher_conditions)==8 and not {c['composition_hex'] for c in teacher_conditions}&{c['composition_hex'] for c in test+panel['validation_rows']}
    assert not {r['condition']['composition_hex'] for r in data}&{c['composition_hex'] for c in test}
    args.out.mkdir(parents=True);source=base.HarmonicSource();worker=Path(__file__).resolve().parent/'oracle_worker.py'
    assert sha(worker)==spec['oracle_worker_sha256'] and sha(spec['oracle_checkpoint'])==spec['oracle_sha256']
    evaluation=args.out/'evaluation';evaluation.mkdir();parent_hashes={};teacher_queries=0;start=time.perf_counter()
    for name,arm in spec['arms'].items():
        checkpoint=args.project/arm['checkpoint'];assert sha(checkpoint)==arm['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);state=saved['ema_state_dict'];del saved
        original_report=args.project/arm['base_report'];assert sha(original_report)==arm['base_report_sha256']
        baseline=json.loads(original_report.read_text());assert baseline['complete']
        model=make_model(arm,state);context=make_context(arm,source);parent_hashes[name]=base.state_hash(model)
        assert baseline['model_state_sha256']==parent_hashes[name]
        for row in baseline['rows']:
            i=row['condition_index'];file=original_report.parent/f'{arm["base_label"]}_c{i}.pt';assert sha(file)==row['sample_sha256']
            (evaluation/f'{name}_base_c{i}.pt').symlink_to(file.resolve())
        (evaluation/f'{name}_base_results.json').symlink_to(original_report.resolve())
        folder=args.out/name;folder.mkdir();raw=folder/'raw_teacher';raw.mkdir()
        fit_report=evaluate(model,source,arm['spec'],context,teacher_conditions,spec['teacher_seed'],spec['teacher_draws'],raw,'fit')
        del model,context;gc.collect();torch.cuda.empty_cache()
        refined=folder/'teacher';refined.mkdir();pools=[];queries=0
        c=teacher_conditions[0]
        with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16,timeout_seconds=180.) as oracle:
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
            for i,row in enumerate(fit_report['rows']):
                file=raw/f'fit_c{i}.pt';draws=torch.load(file,map_location='cpu',weights_only=False);c=draws['condition']
                keep=torch.tensor([r['graph_supported'] for r in row['records']],dtype=torch.bool);anchors=draws['positions'][keep]
                if not len(anchors):continue
                oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
                before=oracle.evaluated
                record=make_escorted_teacher(anchors,c,oracle,**spec['thermal_teacher'],seed=spec['teacher_seed']*100003+i)
                kept=record['eligible'];value=dict(condition=c,raw_positions=anchors[kept],proposals=record['proposal'][kept],uniform_weights=record['uniform_weights'][kept],
                    record=record,raw_source_sha256=sha(file),oracle_queries=oracle.evaluated-before,role='FIT')
                atomic_save(value,refined/f'c{i}.pt')
                if kept.any():pools.append(value)
                print(json.dumps(dict(phase='teacher',method=name,condition=i,valid_starts=len(anchors),retained=int(kept.sum()),queries=oracle.evaluated)),flush=True)
            queries=oracle.evaluated;assert queries==oracle.requested_evaluations
        assert len(pools)>=spec['minimum_teacher_compositions'], 'Insufficient supported teacher compositions; do not replenish.'
        teacher_queries+=queries
        write(folder/'teacher.json',dict(complete=True,protocol_sha256=ph,parent_state_sha256=parent_hashes[name],queries=queries,
            attempted=512,retained_anchors=sum(len(p['raw_positions']) for p in pools),conditions=len(pools),raw_report_sha256=sha(raw/'fit_results.json')))
        states={};names=None
        for role in ['replay','physical']:
            torch.manual_seed(spec['training_seed']);model=make_model(arm,state).train().requires_grad_(True);context=make_context(arm,source)
            optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],amsgrad=True,weight_decay=1e-12)
            names={n for n,_ in model.named_parameters()};rng=torch.Generator().manual_seed(spec['training_seed']+17)
            target_dir=folder/role;target_dir.mkdir();tick=time.perf_counter()
            for step in range(1,arm['student_steps']+1):
                if step%2:
                    index=int(torch.randint(len(data),(1,),generator=rng));entry=data[index];c=entry['condition'];target=entry['positions'];label=dict(kind='reference',row=index)
                else:
                    index=int(torch.randint(len(pools),(1,),generator=rng));entry=pools[index];j=int(torch.randint(len(entry['raw_positions']),(1,),generator=rng));u=float(torch.rand((),generator=rng));c=entry['condition']
                    if role=='replay':target=entry['raw_positions'][j];particle=-1
                    else:
                        weights=entry['uniform_weights'][j];particle=int(torch.searchsorted(weights.cumsum(0),u,right=True).clamp_max(len(weights)-1));assert weights[particle]>0;target=entry['proposals'][j,particle]
                    label=dict(kind='generated_fit',pool=index,row=j,uniform_draw=u,particle=particle)
                clean=target[None].float().cuda();numbers=torch.tensor(c['numbers'],device='cuda')[None];noise_seed=spec['training_seed']*1000003+step
                objective=(feedback.loss(model,clean,numbers,arm['spec'],source,context,noise_seed) if context is not None else
                           base.loss(model,clean,numbers,arm['spec']['kind'],arm['spec'],source,noise_seed))
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite physical-adaptation objective')
                optimizer.zero_grad(set_to_none=True);objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                with (target_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,selection=label,composition=c['composition_hex'],loss=float(objective.detach()),gradient_norm=float(norm)))+'\n')
                if step%250==0:print(json.dumps(dict(phase='fit',method=name,role=role,step=step,loss=float(objective.detach()))),flush=True)
            states[role]={n:v.detach().cpu() for n,v in model.state_dict().items()}
            atomic_save(dict(state_dict=states[role],optimizer_state_dict=optimizer.state_dict(),protocol_sha256=ph,steps=arm['student_steps']),target_dir/'last.ckpt')
            write(target_dir/'training.json',dict(complete=True,steps=arm['student_steps'],examples_seen=arm['student_steps'],
                backbone_example_passes=arm['student_steps']*arm['passes_per_example'],seconds=time.perf_counter()-tick,checkpoint_sha256=sha(target_dir/'last.ckpt')))
            del model,optimizer,context;gc.collect();torch.cuda.empty_cache()
        merged={}
        for key,value in state.items():
            if key in names:merged[key]=value+(states['physical'][key]-states['replay'][key])
            else:assert torch.equal(value,states['physical'][key]) and torch.equal(value,states['replay'][key]);merged[key]=value.clone()
        final=folder/'paired.ckpt';atomic_save(dict(state_dict=merged,protocol_sha256=ph,parent_state_sha256=parent_hashes[name],coefficient=1.),final)
        model=make_model(arm,merged);context=make_context(arm,source)
        evaluate(model,source,arm['spec'],context,test,spec['evaluation_seed'],32,evaluation,name+'_physical')
        write(folder/'complete.json',dict(complete=True,protocol_sha256=ph,paired_checkpoint_sha256=sha(final),teacher_queries=queries))
        del model,context,state,states,merged;gc.collect();torch.cuda.empty_cache()
    # A single persistent oracle reads every raw output; reference geometries add parity checks.
    quality=args.out/'physical_eval';quality.mkdir();records=[];c=test[0]
    poolfile=args.project/panel['source_pool'];assert sha(poolfile)==panel['source_pool_sha256'];pool=json.loads(poolfile.read_text())
    with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['atomic_numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=8,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        for i,c in enumerate(test):
            oracle.condition=dict(numbers=c['atomic_numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
            for label in ['distance_base','distance_physical','gaga_base','gaga_physical','reference']:
                if label=='reference':
                    reference=pool['rows'][c['pool_index']];assert reference['condition']['atomic_numbers']==c['atomic_numbers']
                    x=torch.tensor(reference['reference_positions'],dtype=torch.float64)[None];digest=None
                else:
                    file=evaluation/f'{label}_c{i}.pt';x=torch.load(file,map_location='cpu',weights_only=False)['positions'];digest=sha(file)
                n=len(x);e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=16)
                file=quality/f'{label}_c{i}.pt';atomic_save(dict(positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=(e[:n]+e[n:])/2,even_force_eV_A=(f[:n]-f[n:])/2,source_sample_sha256=digest),file)
                records.append(dict(method=label,condition_index=i,artifact=file.name,artifact_sha256=sha(file),source_sample_sha256=digest))
            write(quality/'results.json',dict(complete=False,protocol_sha256=ph,rows=records,queries=oracle.evaluated))
            print(json.dumps(dict(phase='esen_evaluation',condition=i,queries=oracle.evaluated)),flush=True)
        queries=oracle.evaluated;assert queries==oracle.requested_evaluations==spec['expected_esen_evaluation_queries_per_seed']
    write(quality/'results.json',dict(complete=True,protocol_sha256=ph,rows=records,queries=queries))
    write(args.out/'complete.json',dict(complete=True,protocol_sha256=ph,parent_state_sha256=parent_hashes,teacher_raw_queries=teacher_queries,
        evaluation_raw_queries=queries,new_fit_outputs=1024,new_evaluation_outputs=2048,new_optimizer_steps=6000,
        physical_and_replay_backbone_example_passes=8000,seconds=time.perf_counter()-start))


if __name__=='__main__':main()
