#!/usr/bin/env python3
"""Transfer fixed-budget complete-work teachers into the original molecular FM."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from flowmol.model_utils.load import read_config_file

from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import restore_model,sample_source,evaluate
from scripts.research.run_tree_manifold import make_graph
from scripts.research.evaluate_generator_quality import write
from scripts.research.train_electronic_fm import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    assert spec['frozen'] and not args.out.exists()
    for key in ['config','data','warm_checkpoint','replay_checkpoint','selection_log','teacher_protocol','condition_manifest']:
        assert sha(args.project/spec[key])==spec[key+'_sha256']
    teacher_spec=json.loads((args.project/spec['teacher_protocol']).read_text())
    folder=args.project/spec['teacher_run'];teacher_result=json.loads((folder/'results.json').read_text())
    assert teacher_result['complete'] and teacher_result['protocol_sha256']==spec['teacher_protocol_sha256']
    lookup={(r['method'],r['condition_index'],r['anchor_index']):r for r in teacher_result['rows']}
    pools=[];artifacts=[]
    for ref in teacher_spec['teacher_files']:
        file=args.project/ref['path'];assert sha(file)==ref['sha256']
        old=torch.load(file,map_location='cpu',weights_only=False)
        expected=torch.where(old['record']['eligible'])[0].tolist();assert expected==ref['anchor_indices']
        assert torch.equal(old['raw_positions'],old['record']['anchor'][old['record']['eligible']])
        pool=dict(condition=old['condition'],anchors=old['raw_positions'],particles={m:[] for m in spec['methods']},weights={m:[] for m in spec['methods']})
        for j in expected:
            for m in ['translation','secant']:
                row=lookup[m,ref['condition_index'],j];path=folder/row['artifact'];assert sha(path)==row['artifact_sha256']
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['protocol_sha256']==spec['teacher_protocol_sha256']
                assert torch.equal(saved['anchor'],old['record']['anchor'][j])
                assert saved['valid'].any(), 'New teacher lost all particles for an original replay anchor'
                assert torch.isfinite(saved['weights']).all() and abs(float(saved['weights'].sum())-1)<1e-12
                if m=='translation':
                    pool['particles']['force24'].append(saved['proposal'])
                    pool['weights']['force24'].append(saved['valid'].double()/saved['valid'].sum())
                    label='work24'
                else:label='curvature_work16'
                pool['particles'][label].append(saved['proposal']);pool['weights'][label].append(saved['weights'])
                artifacts.append(dict(artifact=row['artifact'],sha256=sha(path)))
        pools.append(pool)
    assert sum(len(p['anchors']) for p in pools)==spec['teacher_anchors']
    excluded={r['composition_hex'] for r in json.loads((args.project/spec['condition_manifest']).read_text())['rows']}
    assert not excluded & {p['condition']['composition_hex'] for p in pools}
    data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training']
    warm=torch.load(args.project/spec['warm_checkpoint'],map_location='cpu',weights_only=False)
    replay=torch.load(args.project/spec['replay_checkpoint'],map_location='cpu',weights_only=False)
    selection=[json.loads(line) for line in (args.project/spec['selection_log']).read_text().splitlines()]
    assert len(selection)==spec['training_steps']==1000
    cfg=read_config_file(args.project/spec['config']);cfg['mol_fm'].pop('bgfm',None)
    assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    args.out.mkdir(parents=True);out=args.out/'evaluation';out.mkdir()
    write(args.out/'teacher_inputs.json',dict(complete=True,teacher_result_sha256=sha(folder/'results.json'),
        artifacts=artifacts,retained_anchors=spec['teacher_anchors'],replay_targets_identical=True,protocol_sha256=ph))
    prior=prior_from_checkpoint(warm);start=time.perf_counter()
    for method in spec['methods']:
        torch.manual_seed(spec['training_seed']);model=restore_model(cfg,warm).train().requires_grad_(True)
        extra=list(model.vector_field.self_conditioning_residual_layer.parameters())+list(model.vector_field.to_edge_logits.parameters())
        extra_ids={id(p) for p in extra}
        optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in extra_ids],lr=spec['fm_lr']),
            dict(params=extra,lr=spec['feedback_lr'])],weight_decay=1e-12)
        directory=args.out/method;directory.mkdir();tick=time.perf_counter()
        for step,selected in enumerate(selection,1):
            assert selected['step']==step;old_label=selected['selection'];label=dict(old_label)
            if label['kind']=='reference':
                row=data[label['row']];c=row['condition'];target=row['positions']
            else:
                row=pools[label['pool']];c=row['condition'];j=label['row'];weights=row['weights'][method][j]
                particle=int(torch.searchsorted(weights.cumsum(0),label['uniform_draw'],right=True).clamp_max(len(weights)-1))
                assert weights[particle]>0;target=row['particles'][method][j][particle];label['particle']=particle
            assert selected['composition']==c['composition_hex']
            g,nbi,uem=make_graph(c,cfg);g.ndata['x_1_true']=target.cuda().float()
            g.ndata['has_reference_geometry']=torch.ones(c['n_atoms'],1,dtype=torch.bool,device='cuda')
            x0,_=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['training_seed']*3000017+step)
            objective=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',prior_positions=x0,
                pairing='typed_rotation',pairing_radii=covalent_radii(c['numbers'],device='cuda',dtype=torch.float32),
                generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*1000003+step),
                pairing_generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*2000003+step))
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite curvature-distillation loss')
            optimizer.zero_grad(set_to_none=True);objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with (directory/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,selection=label,composition=c['composition_hex'],loss=float(objective.detach()),gradient_norm=float(norm)))+'\n')
            if step%100==0:print(json.dumps(dict(method=method,step=step,loss=float(objective.detach()))),flush=True)
        trained={n:v.detach().cpu() for n,v in model.state_dict().items()}
        torch.save(dict(state_dict=trained,optimizer_state_dict=optimizer.state_dict(),protocol_sha256=ph),directory/'student.ckpt')
        names={n for n,_ in model.named_parameters()};merged={}
        for name,value in warm['state_dict'].items():
            if name in names:merged[name]=value+(trained[name]-replay['state_dict'][name])
            else:
                assert torch.equal(value,trained[name]) and torch.equal(value,replay['state_dict'][name]);merged[name]=value.clone()
        recipe=dict(warm['research_protocol'],curvature_distillation_protocol_sha256=ph,curvature_distillation_method=method)
        checkpoint=directory/'last.ckpt';torch.save(dict(state_dict=merged,research_protocol=recipe,source_prior=warm['source_prior']),checkpoint)
        write(directory/'training.json',dict(complete=True,steps=1000,seconds=time.perf_counter()-tick,
            checkpoint_sha256=sha(checkpoint),student_sha256=sha(directory/'student.ckpt'),selection_log_sha256=spec['selection_log_sha256'],
            replay_checkpoint_sha256=spec['replay_checkpoint_sha256'],coefficient=1.))
        model.load_state_dict(merged,strict=True)
        evaluate(model,prior,method,cfg,spec,ph,out,sha(checkpoint),args.project/spec['condition_manifest'])
        del model,optimizer,trained,merged,g,nbi,uem;gc.collect();torch.cuda.empty_cache()
    worker=args.project/spec['oracle_worker'];assert sha(worker)==spec['oracle_worker_sha256']
    assert sha(spec['oracle_checkpoint'])==spec['oracle_sha256']
    quality=args.out/'physical_eval';quality.mkdir();rows=[]
    c=pools[0]['condition']
    with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],
            spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=8,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        for i in spec['conditions']:
            for method in spec['methods']:
                file=out/f'{method}_c{i}.pt';saved=torch.load(file,map_location='cpu',weights_only=False);c=saved['condition'];x=saved['positions'];n=len(x)
                oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
                e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=16);target=quality/f'{method}_c{i}.pt'
                torch.save(dict(positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=(e[:n]+e[n:])/2,
                    even_force_eV_A=(f[:n]-f[n:])/2,source_sample_sha256=sha(file)),target)
                rows.append(dict(method=method,condition_index=i,artifact=target.name,artifact_sha256=sha(target),source_sample_sha256=sha(file)))
            write(quality/'results.json',dict(complete=False,protocol_sha256=ph,rows=rows,raw_queries=oracle.evaluated))
        queries=oracle.evaluated;assert queries==oracle.requested_evaluations==spec['esen_queries_per_seed']
    write(quality/'results.json',dict(complete=True,protocol_sha256=ph,rows=rows,raw_queries=queries))
    write(args.out/'complete.json',dict(complete=True,protocol_sha256=ph,methods=spec['methods'],new_training_steps=3000,
        new_neural_outputs=spec['new_neural_outputs'],evaluation_raw_queries=queries,teacher_raw_queries=teacher_result['new_raw_queries'],seconds=time.perf_counter()-start))


if __name__=='__main__':main()
