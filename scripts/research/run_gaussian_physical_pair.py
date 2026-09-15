#!/usr/bin/env python3
"""Complete the missing Gaussian-source cell with the original physical recipe."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file

from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.escorted_thermal_teacher import make_escorted_teacher
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_chemical_policy import write
from scripts.research.run_tree_manifold import make_graph
from scripts.research.train_electronic_fm import sha
from scripts.research.tree_prior_fm import restore_model,sample_source,evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    assert spec['frozen'] and not args.out.exists()
    for key in ['config','data','warm_checkpoint','condition_manifest','original_teacher_protocol']:
        assert sha(args.project/spec[key])==spec[key+'_sha256']
    original=json.loads((args.project/spec['original_teacher_protocol']).read_text())
    for key in ['teacher_training_rows','teacher_draws_per_composition','teacher_seed','training_seed','training_steps','fm_lr','feedback_lr','thermal_teacher']:
        assert spec[key]==original[key],key
    cfg=read_config_file(args.project/spec['config']);cfg['mol_fm'].pop('bgfm',None)
    assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)
    warm=torch.load(args.project/spec['warm_checkpoint'],map_location='cpu',weights_only=False)
    assert warm['research_protocol']['source_prior_kind']=='gaussian'
    prior=prior_from_checkpoint(warm);assert prior is None
    rows=[data['training'][i] for i in spec['teacher_training_rows']]
    excluded={r['composition_hex'] for r in json.loads((args.project/spec['condition_manifest']).read_text())['rows']}
    assert not excluded & {r['condition']['composition_hex'] for r in rows}
    args.out.mkdir(parents=True);teacher_dir=args.out/'teacher';teacher_dir.mkdir();start=time.perf_counter()
    model=restore_model(cfg,warm).eval().requires_grad_(False);raw=[]
    for i,row in enumerate(rows):
        c=row['condition'];batch=spec['teacher_draws_per_composition'];g,nbi,uem=make_graph(c,cfg,batch)
        x0=torch.stack([sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['teacher_seed']*1000003+i*100003+j)[0] for j in range(batch)])
        with torch.no_grad():
            y=sample_clamped_flow(model,g,nbi,uem,x0=x0.reshape(-1,3).cuda().float(),n_ode_steps=32,terminal_time=1.,parameterization='displacement').reshape(batch,c['n_atoms'],3).cpu().double()
        noise=torch.randn(y.shape,dtype=torch.float64,generator=torch.Generator().manual_seed(spec['teacher_seed']+i));noise-=noise.mean(1,keepdim=True)
        y+=spec['terminal_noise_std_A']*noise;y-=y.mean(1,keepdim=True)
        assessment=assess(y,c,list(range(batch)));keep=torch.tensor([r['graph_supported'] for r in assessment['records']],dtype=torch.bool)
        item=dict(condition=c,raw_positions=y,source_positions=x0,graph_supported=keep,assessment=assessment,role='FIT',training_processed_index=c['processed_index'])
        torch.save(item,teacher_dir/f'raw_c{i}.pt');raw.append(item)
        print(json.dumps(dict(phase='teacher_generation',condition=i,valid=int(keep.sum()),attempted=batch)),flush=True)
    del model,g,nbi,uem;gc.collect();torch.cuda.empty_cache()
    worker=Path(__file__).resolve().parent/'oracle_worker.py'
    assert sha(worker)==spec['oracle_worker_sha256'] and sha(spec['oracle_checkpoint'])==spec['oracle_sha256']
    pools=[];teacher_queries=0
    for i,item in enumerate(raw):
        c=item['condition'];x=item['raw_positions'][item['graph_supported']]
        if not len(x):continue
        with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16) as oracle:
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
            record=make_escorted_teacher(x,c,oracle,**spec['thermal_teacher'],seed=spec['teacher_seed']*100003+i)
            assert oracle.evaluated==oracle.requested_evaluations;teacher_queries+=oracle.evaluated
            saved=dict(condition=c,raw_positions=x[record['eligible']],proposals=record['proposal'][record['eligible']],
                weights=record['weights'][record['eligible']],uniform_weights=record['uniform_weights'][record['eligible']],
                role='FIT',record=record,raw_source_sha256=sha(teacher_dir/f'raw_c{i}.pt'),oracle_queries=oracle.evaluated)
            torch.save(saved,teacher_dir/f'refined_c{i}.pt')
            if len(saved['raw_positions']):pools.append(saved)
        print(json.dumps(dict(phase='teacher',condition=i,retained=len(saved['raw_positions']),queries=teacher_queries)),flush=True)
    assert len(pools)>=spec['minimum_teacher_compositions']
    write(args.out/'teacher_progress.json',dict(complete=True,protocol_sha256=ph,queries=teacher_queries,conditions=len(pools),
        retained_anchors=sum(len(p['raw_positions']) for p in pools),new_neural_outputs=len(rows)*spec['teacher_draws_per_composition'],
        scope='Original eight-particle preparation, including stored work diagnostics; students use uniform force targets only.'))
    states={};names=None
    for method in ['replay','escort']:
        torch.manual_seed(spec['training_seed']);model=restore_model(cfg,warm).train().requires_grad_(True)
        names={n for n,_ in model.named_parameters()}
        extra=list(model.vector_field.self_conditioning_residual_layer.parameters())+list(model.vector_field.to_edge_logits.parameters());ids={id(p) for p in extra}
        optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=spec['fm_lr']),dict(params=extra,lr=spec['feedback_lr'])],weight_decay=1e-12)
        rng=torch.Generator().manual_seed(spec['training_seed']+17);directory=args.out/method;directory.mkdir();tick=time.perf_counter()
        for step in range(1,spec['training_steps']+1):
            if step%2:
                idx=int(torch.randint(len(data['training']),(1,),generator=rng));entry=data['training'][idx];c=entry['condition'];target=entry['positions'];label=dict(kind='reference',row=idx)
            else:
                idx=int(torch.randint(len(pools),(1,),generator=rng));entry=pools[idx];j=int(torch.randint(len(entry['raw_positions']),(1,),generator=rng));c=entry['condition'];u=float(torch.rand((),generator=rng))
                if method=='replay':target=entry['raw_positions'][j];particle=-1
                else:
                    weights=entry['uniform_weights'][j];particle=int(torch.searchsorted(weights.cumsum(0),u,right=True).clamp_max(len(weights)-1));target=entry['proposals'][j,particle];assert weights[particle]>0
                label=dict(kind='generated_fit',pool=idx,row=j,particle=particle,uniform_draw=u)
            g,nbi,uem=make_graph(c,cfg);g.ndata['x_1_true']=target.cuda().float();g.ndata['has_reference_geometry']=torch.ones(c['n_atoms'],1,dtype=torch.bool,device='cuda')
            x0,_=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['training_seed']*3000017+step)
            objective=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',prior_positions=x0,pairing='typed_rotation',
                pairing_radii=covalent_radii(c['numbers'],device='cuda',dtype=torch.float32),generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*1000003+step),
                pairing_generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*2000003+step))
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite Gaussian physical objective')
            optimizer.zero_grad(set_to_none=True);objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,selection=label,composition=c['composition_hex'],loss=float(objective.detach()),gradient_norm=float(norm)))+'\n')
            if step%100==0:print(json.dumps(dict(method=method,step=step,loss=float(objective.detach()))),flush=True)
        states[method]={n:v.detach().cpu() for n,v in model.state_dict().items()}
        checkpoint=directory/'last.ckpt'
        torch.save(dict(state_dict=states[method],research_protocol=dict(warm['research_protocol'],source_factorial_protocol_sha256=ph,physical_method=method),source_prior=None,global_step=1000,optimizer_state_dict=optimizer.state_dict()),checkpoint)
        write(directory/'training.json',dict(complete=True,steps=1000,seconds=time.perf_counter()-tick,checkpoint_sha256=sha(checkpoint)))
        del model,optimizer,g,nbi,uem;gc.collect();torch.cuda.empty_cache()
    merged={}
    for name,value in warm['state_dict'].items():
        if name in names:merged[name]=value+(states['escort'][name]-states['replay'][name])
        else:
            assert torch.equal(value,states['escort'][name]) and torch.equal(value,states['replay'][name]);merged[name]=value.clone()
    directory=args.out/'gaussian_physical';directory.mkdir();checkpoint=directory/'last.ckpt'
    final=dict(state_dict=merged,research_protocol=dict(warm['research_protocol'],source_factorial_protocol_sha256=ph,physical_method='paired_force'),source_prior=None)
    torch.save(final,checkpoint)
    model=restore_model(cfg,final);out=args.out/'evaluation';out.mkdir()
    evaluate(model,prior,'gaussian_physical',cfg,spec,ph,out,sha(checkpoint),args.project/spec['condition_manifest'])
    del model,states,merged,final;gc.collect();torch.cuda.empty_cache()
    quality=args.out/'physical_eval';quality.mkdir();records=[]
    c=pools[0]['condition']
    with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=8,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        for i in spec['conditions']:
            file=out/f'gaussian_physical_c{i}.pt';saved=torch.load(file,map_location='cpu',weights_only=False);c=saved['condition'];x=saved['positions'];n=len(x)
            oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
            e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=16);target=quality/f'gaussian_physical_c{i}.pt'
            torch.save(dict(positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=(e[:n]+e[n:])/2,even_force_eV_A=(f[:n]-f[n:])/2,source_sample_sha256=sha(file)),target)
            records.append(dict(method='gaussian_physical',condition_index=i,artifact=target.name,artifact_sha256=sha(target),source_sample_sha256=sha(file)))
            write(quality/'results.json',dict(complete=False,protocol_sha256=ph,rows=records,raw_queries=oracle.evaluated))
        queries=oracle.evaluated;assert queries==oracle.requested_evaluations==spec['esen_queries_per_seed']
    write(quality/'results.json',dict(complete=True,protocol_sha256=ph,rows=records,raw_queries=queries))
    write(args.out/'complete.json',dict(complete=True,protocol_sha256=ph,new_training_steps=2000,new_fit_outputs=256,new_evaluation_outputs=768,
        teacher_raw_queries=teacher_queries,evaluation_raw_queries=queries,checkpoint_sha256=sha(checkpoint),seconds=time.perf_counter()-start))


if __name__=='__main__':main()
