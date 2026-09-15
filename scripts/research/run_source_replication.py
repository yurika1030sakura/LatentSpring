#!/usr/bin/env python3
"""Three prespecified extra continuations with retained training-budget curves."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.geometry_self_conditioning import patch_geometry_self_conditioning
from scripts.research.evaluate_chemical_policy import write
from scripts.research.run_tree_manifold import make_graph
from scripts.research.train_electronic_fm import sha
from scripts.research.tree_prior_fm import restore_model,sample_source,evaluate,load_prior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    assert spec['frozen'] and not args.out.exists()
    for key in ['data','config','warm_checkpoint','condition_manifest']:assert sha(args.project/spec[key])==spec[key+'_sha256']
    data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training'];assert len(data)==3000
    panel=json.loads((args.project/spec['condition_manifest']).read_text())
    assert not {r['condition']['composition_hex'] for r in data}&{c['composition_hex'] for c in panel['rows']}
    cfg=read_config_file(args.project/spec['config']);cfg['mol_fm'].pop('bgfm',None)
    assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    warm=torch.load(args.project/spec['warm_checkpoint'],map_location='cpu',weights_only=False)
    args.out.mkdir(parents=True);completed=[];new_outputs=0
    for method in ['gaussian','harmonic_tree']:
        prior=load_prior(method,None,spec,ph);torch.manual_seed(spec['fm_seed'])
        model=restore_model(cfg,warm).train();model._research_prior_kind=method
        patch_geometry_self_conditioning(model,edge_feedback='clamped',**spec['self_conditioning'])
        directory=args.out/method;directory.mkdir()
        extra=list(model.vector_field.self_conditioning_residual_layer.parameters())+list(model.vector_field.to_edge_logits.parameters());ids={id(p) for p in extra}
        optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=spec['fm_lr']),dict(params=extra,lr=spec['feedback_lr'])],weight_decay=1e-12)
        trainable={n:p.requires_grad for n,p in model.named_parameters()}
        torch.manual_seed(spec['fm_seed']+17);seconds=0.;milestones=[]
        for step in range(1,spec['fm_steps']+1):
            tick=time.perf_counter();index=(step-1)%len(data);row=data[index];c=row['condition'];g,nbi,uem=make_graph(c,cfg)
            g.ndata['x_1_true']=row['positions'].cuda().float();g.ndata['has_reference_geometry']=torch.ones(c['n_atoms'],1,dtype=torch.bool,device='cuda')
            x0,_=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['fm_seed']*3000017+step)
            objective=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',prior_positions=x0,pairing='typed_rotation',
                pairing_radii=covalent_radii(c['numbers'],device='cuda',dtype=torch.float32),generator=torch.Generator(device='cuda').manual_seed(spec['fm_seed']*1000003+step),
                pairing_generator=torch.Generator(device='cuda').manual_seed(spec['fm_seed']*2000003+step))
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite source replication objective')
            optimizer.zero_grad(set_to_none=True);objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            with (directory/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,training_row=index,processed_index=c['processed_index'],loss=float(objective.detach()),gradient_norm=float(norm),training_seconds=seconds))+'\n')
            if step%250==0:print(json.dumps(dict(method=method,step=step,loss=float(objective.detach()),training_seconds=seconds)),flush=True)
            if step in spec['budget_steps']:
                recipe={**warm['research_protocol'],**spec,'format':'source_replication_v1','source_prior_kind':method,'model_variant':method,
                    'source_replication_protocol_sha256':ph,'position_parameterization':'displacement',
                    'geometry_self_conditioning':dict(edge_feedback='clamped',**spec['self_conditioning'])}
                checkpoint=directory/f'step{step}.ckpt'
                torch.save(dict(state_dict=model.state_dict(),research_protocol=recipe,global_step=step,optimizer_state_dict=optimizer.state_dict(),
                    source_prior=None if prior is None else dict(configuration=prior.configuration,state_dict=prior.state_dict())),checkpoint)
                milestone=dict(step=step,checkpoint_sha256=sha(checkpoint),training_seconds=seconds,examples_seen=step,distinct_training_rows=min(step,len(data)),primitive_denoiser_training_forwards=2*step)
                milestones.append(milestone);write(directory/'milestones.json',dict(complete=False,protocol_sha256=ph,rows=milestones))
                evaluation=directory/f'evaluation_step{step}';evaluation.mkdir()
                evaluation_spec=dict(spec,samples_per_condition=spec['samples_at_budget'][str(step)])
                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                evaluate(model,prior,method,cfg,evaluation_spec,ph,evaluation,sha(checkpoint),args.project/spec['condition_manifest'])
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
                model.train()
                for name,param in model.named_parameters():param.requires_grad_(trainable[name])
                new_outputs+=len(spec['conditions'])*evaluation_spec['samples_per_condition']
        write(directory/'milestones.json',dict(complete=True,protocol_sha256=ph,rows=milestones))
        write(directory/'training.json',dict(complete=True,steps=spec['fm_steps'],training_seconds=seconds,
            budget_steps=spec['budget_steps'],gpu=torch.cuda.get_device_name(),protocol_sha256=ph))
        completed.append(method);write(args.out/'progress.json',dict(complete=len(completed)==2,protocol_sha256=ph,completed=completed,new_outputs=new_outputs))
        del model,optimizer,g,nbi,uem;gc.collect();torch.cuda.empty_cache()
    write(args.out/'complete.json',dict(complete=True,protocol_sha256=ph,new_training_steps=2*spec['fm_steps'],
        new_neural_outputs=new_outputs,new_physical_queries=0,source_law_and_budget_not_selected_from_outcomes=True))


if __name__=='__main__':main()
