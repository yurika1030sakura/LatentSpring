"""Matched direct FM fits to original or physically corrected TRAIN references."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.weighted_endpoints import EndpointDraw
from cfm_mol.weighted_endpoint_fm import dgl_batch,endpoint_fm_loss
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from scripts.research.run_gaga_feedback import atomic_save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','bank','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--method',choices=['reference_ft','relaxed_ft'],required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'] and a.seed in spec['training_seeds']
    completion=json.loads((a.bank.parent/'complete.json').read_text());assert completion['complete'] and completion['protocol_sha256']==ph and completion['bank_sha256']==sha(a.bank)
    data=torch.load(a.bank,map_location='cpu',weights_only=False);assert data['protocol_sha256']==ph and len(data['rows'])==128
    cfg=read_config_file(a.project/spec['config']);cfg['mol_fm'].pop('bgfm',None)
    assert sha(a.project/spec['config'])==spec['config_sha256'] and cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    path=a.project/spec['checkpoint'];assert sha(path)==spec['checkpoint_sha256']
    parent=torch.load(path,map_location='cpu',weights_only=False);recipe=parent['research_protocol']
    assert recipe['position_parameterization']=='displacement' and recipe['data_endpoint_time']==1.
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(parent['state_dict'],strict=True)
    patch_smooth_geometry(model,recipe.get('geometry_softening',0.));model.cuda().float().train()
    trainable=[p for p in model.parameters() if p.requires_grad];frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    extra=set()
    for name in ['self_conditioning_residual_layer','to_edge_logits']:
        module=getattr(model.vector_field,name,None)
        if module:extra|={id(p) for p in module.parameters() if p.requires_grad}
    groups=[dict(params=[p for p in trainable if (id(p) in extra)==flag],lr=lr) for flag,lr in [(False,spec['learning_rate']),(True,spec['feedback_learning_rate'])]]
    optimizer=torch.optim.AdamW([g for g in groups if g['params']],weight_decay=1e-12)
    prior=prior_from_checkpoint(parent);rng=np.random.default_rng(a.seed);a.out.mkdir(parents=True,exist_ok=False)
    key='reference' if a.method=='reference_ft' else 'physical';tick=time.perf_counter()
    with (a.out/'metrics.jsonl').open('w') as stream:
        for step in range(1,spec['training_steps']+1):
            index=int(rng.integers(len(data['rows'])));row=data['rows'][index];c=row['condition']
            x=row[key].numpy().copy();noise=rng.normal(size=x.shape);noise-=noise.mean(0);x+=spec['endpoint_smoothing_A']*noise
            draws=[EndpointDraw(str(index),str(step),'empirical_endpoint',c,x)]
            g,nbi,uem=dgl_batch(draws,cfg,torch.device('cuda'),model_kT_eV=1.)
            loss=endpoint_fm_loss(model,g,nbi,uem,draws,prior,a.seed*3000017+step)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite FM loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
            value=dict(step=step,bank_row=index,training_row=row['training_row'],loss=float(loss.detach()),gradient_norm=float(norm))
            stream.write(json.dumps(value)+'\n')
            if step%100==0:stream.flush();print(json.dumps(dict(method=a.method,seed=a.seed,**value)),flush=True)
    for n,p in model.named_parameters():
        if n in frozen:assert torch.equal(p.detach().cpu(),frozen[n]),n
    target=dict(mode='empirical_physical_endpoints',model_kT_eV=1.,teacher_temperature_K=None,
        terminal_noise_std_A=0.,protocol_sha256=ph,bank_sha256=sha(a.bank),method=a.method,
        target=spec['target'],thermal_distribution_claim=False)
    path=a.out/'last.ckpt';atomic_save(dict(state_dict=model.state_dict(),source_prior=parent['source_prior'],
        research_protocol={**recipe,'direct_endpoint_training':target},global_step=spec['training_steps'],
        optimizer_state_dict=optimizer.state_dict(),training_seed=a.seed),path)
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,bank_sha256=sha(a.bank),
        checkpoint_sha256=sha(path),seed=a.seed,method=a.method,steps=spec['training_steps'],
        primitive_training_forwards=2*spec['training_steps'],seconds=time.perf_counter()-tick,oracle_queries=0))


if __name__=='__main__':main()
