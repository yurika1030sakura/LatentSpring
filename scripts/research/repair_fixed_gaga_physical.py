#!/usr/bin/env python3
"""Repair only the unintended GAGA schedule update; retain the valid FM result."""
import argparse
import gc
import json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.run_matched_physical import make_model
from scripts.research.run_gaga_feedback import evaluate,atomic_save
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    old=args.project/spec['reuse_study_run'];assert sha(old/'complete.json')==spec['reuse_completion_sha256']
    assert not args.out.exists();args.out.mkdir(parents=True)
    (args.out/'distance').symlink_to((old/'distance').resolve(),target_is_directory=True)
    evaluation=args.out/'evaluation';evaluation.mkdir()
    for label in ['distance_base','distance_physical','gaga_base']:
        for path in (old/'evaluation').glob(label+'_*'):(evaluation/path.name).symlink_to(path.resolve())
    folder=args.out/'gaga';folder.mkdir()
    for name in ['raw_teacher','teacher']:(folder/name).symlink_to((old/'gaga'/name).resolve(),target_is_directory=True)
    (folder/'teacher.json').symlink_to((old/'gaga/teacher.json').resolve())
    pools=[]
    for ref in spec['teacher_files']:
        path=args.project/ref['path'];assert sha(path)==ref['sha256'];value=torch.load(path,map_location='cpu',weights_only=False)
        if len(value['raw_positions']):pools.append(value)
    arm=spec['arms']['gaga'];file=args.project/arm['checkpoint'];assert sha(file)==arm['checkpoint_sha256']
    parent=torch.load(file,map_location='cpu',weights_only=False)['ema_state_dict'];source=base.HarmonicSource()
    assert sha(args.project/spec['data'])==spec['data_sha256'];data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training']
    states={};names=None
    for role in ['replay','physical']:
        log_path=old/'gaga'/role/'metrics.jsonl';assert sha(log_path)==spec['selection_log_sha256'][role]
        selections=[json.loads(line) for line in log_path.read_text().splitlines()];assert len(selections)==2000
        torch.manual_seed(spec['training_seed']);model=make_model(arm,parent).train()
        names={n for n,p in model.named_parameters() if p.requires_grad};assert 'gamma.gamma' not in names
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec['learning_rate'],amsgrad=True,weight_decay=1e-12)
        out=folder/role;out.mkdir()
        for step,entry in enumerate(selections,1):
            label=entry['selection'];assert entry['step']==step
            if label['kind']=='reference':row=data[label['row']];condition=row['condition'];target=row['positions']
            else:
                row=pools[label['pool']];condition=row['condition'];j=label['row']
                target=row['raw_positions'][j] if role=='replay' else row['proposals'][j,label['particle']]
            assert condition['composition_hex']==entry['composition']
            x=target[None].float().cuda();z=torch.tensor(condition['numbers'],device='cuda')[None]
            loss=base.loss(model,x,z,'gaga',arm['spec'],source,spec['training_seed']*1000003+step)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite corrected objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();assert model.gamma.gamma.grad is None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            record=dict(step=step,selection=label,composition=entry['composition'],loss=float(loss.detach()),gradient_norm=float(norm))
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if step%250==0:print(json.dumps(dict(role=role,step=step,loss=record['loss'])),flush=True)
        states[role]={n:v.detach().cpu() for n,v in model.state_dict().items()}
        torch.testing.assert_close(states[role]['gamma.gamma'],parent['gamma.gamma'],atol=0,rtol=0)
        checkpoint=out/'last.ckpt';atomic_save(dict(state_dict=states[role],optimizer_state_dict=optimizer.state_dict(),protocol_sha256=ph,steps=2000),checkpoint)
        write(out/'training.json',dict(complete=True,steps=2000,examples_seen=2000,backbone_example_passes=2000,checkpoint_sha256=sha(checkpoint),fixed_noise_schedule_preserved=True))
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    merged={}
    for name,value in parent.items():
        if name in names:merged[name]=value+(states['physical'][name]-states['replay'][name])
        else:
            assert torch.equal(value,states['physical'][name]) and torch.equal(value,states['replay'][name]);merged[name]=value.clone()
    atomic_save(dict(state_dict=merged,protocol_sha256=ph,coefficient=1.,fixed_parameter_names=['gamma.gamma']),folder/'paired.ckpt')
    model=make_model(arm,merged);panel=json.loads((args.project/spec['panel']).read_text())['test_rows']
    evaluate(model,source,arm['spec'],None,panel,spec['evaluation_seed'],32,evaluation,'gaga_physical')
    del model,states,merged;gc.collect();torch.cuda.empty_cache()
    quality=args.out/'physical_eval';quality.mkdir();oldq=json.loads((old/'physical_eval/results.json').read_text());records=[]
    for row in oldq['rows']:
        if row['method']=='gaga_physical':continue
        path=old/'physical_eval'/row['artifact'];assert sha(path)==row['artifact_sha256'];(quality/path.name).symlink_to(path.resolve());records.append(row)
    worker=Path(__file__).resolve().parent/'oracle_worker.py';assert sha(worker)==spec['oracle_worker_sha256'];c=panel[0]
    with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['atomic_numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=8,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        for i,c in enumerate(panel):
            file=evaluation/f'gaga_physical_c{i}.pt';x=torch.load(file,map_location='cpu',weights_only=False)['positions']
            oracle.condition=dict(numbers=c['atomic_numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
            e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=16);target=quality/file.name
            atomic_save(dict(positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=(e[:32]+e[32:])/2,even_force_eV_A=(f[:32]-f[32:])/2,source_sample_sha256=sha(file)),target)
            records.append(dict(method='gaga_physical',condition_index=i,artifact=target.name,artifact_sha256=sha(target),source_sample_sha256=sha(file)))
        assert oracle.evaluated==oracle.requested_evaluations==2048
    write(quality/'results.json',dict(complete=True,protocol_sha256=ph,rows=records,queries=8256,new_queries=2048,reused_queries=6208))
    original=json.loads((old/'complete.json').read_text())
    write(args.out/'complete.json',dict(complete=True,protocol_sha256=ph,parent_state_sha256=original['parent_state_sha256'],teacher_raw_queries=original['teacher_raw_queries'],
        evaluation_raw_queries=8256,new_teacher_queries=0,new_evaluation_queries=2048,new_fit_outputs=0,new_evaluation_outputs=1024,new_optimizer_steps=4000,
        correction='Fixed GAGA schedule only; same parent, cached teachers, target choices, learning rate and coefficient. FM negative outcomes retained unchanged.'))


if __name__=='__main__':main()
