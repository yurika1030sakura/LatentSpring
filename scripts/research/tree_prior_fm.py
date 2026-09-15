#!/usr/bin/env python3
"""Matched main-FM continuations and fresh generation with declared spatial priors."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import dgl
from flowmol.model_utils.load import model_from_config
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.condition_systems import load_condition,graph_from_condition
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from scripts.research.train_tree_mixture_prior import load_training
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_chemical_policy import write


def geometry_counts(x,numbers):
    radii=covalent_radii(numbers,dtype=x.dtype,device=x.device);n=len(radii)
    if n<2:return dict(disconnected=0,overlap=0)
    distance=torch.cdist(x,x);cutoff=radii[:,None]+radii[None,:]
    off=~torch.eye(n,dtype=torch.bool,device=x.device)
    overlap=((distance<.6*cutoff)&off).any((1,2))
    adjacent=distance<=1.25*cutoff
    reached=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device);reached[:,0]=True
    for _ in range(n-1):reached=reached|(adjacent&reached[:,:,None]).any(1)
    return dict(disconnected=int((~reached.all(1)).sum()),overlap=int(overlap.sum()))


def load_prior(method,prior_run,protocol,protocol_hash):
    if method in ['gaussian','warm']:return None
    if method in ['fixed','tree_actual','tree_sham']:return TreeMixturePrior('fixed',width=protocol['edge_log_width']).double()
    if method in ['harmonic_tree','covariance_gaussian']:
        from cfm_mol.tree_prior_controls import TreePriorControl
        return TreePriorControl(method,width=protocol['edge_log_width'],**protocol['covariance_control']).double()
    report=json.loads((prior_run/'results.json').read_text())
    assert report['complete'] and report['protocol_sha256']==protocol_hash
    path=prior_run/f'{method}.pt';assert sha(path)==report['models'][method]['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_sha256']==protocol_hash and state['data_sha256']==report['data_sha256']
    model=TreeMixturePrior(**state['configuration']).double();model.load_state_dict(state['state_dict'],strict=True)
    model.eval();model.requires_grad_(False);return model


def sample_source(prior,numbers,charge,spin,seed):
    generator=torch.Generator().manual_seed(seed)
    if prior is None:
        x=torch.randn((len(numbers),3),dtype=torch.float64,generator=generator)
        return x-x.mean(0),[]
    return prior.sample(numbers,charge,spin,rng=np.random.default_rng(seed+700000001),generator=generator)


def restore_model(cfg,state):
    model=model_from_config(cfg);recipe=state['research_protocol']
    prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True)
    patch_smooth_geometry(model,recipe.get('geometry_softening',0.))
    return model.cuda().float()


def evaluate(model,prior,method,cfg,protocol,protocol_hash,out,checkpoint_hash,manifest,sampler=None):
    model.eval();model.requires_grad_(False);rows=[]
    context_mode=protocol.get('context_modes',{}).get(method)
    if context_mode:
        from cfm_mol.latent_tree_context import context_tree,tree_features,attach_context
    for index in protocol['conditions']:
        condition=load_condition(manifest,index);condition['numbers']=condition['atomic_numbers'];condition['requested_kT_eV']=1.
        numbers=condition['numbers'];n=len(numbers);batch=protocol['evaluation_batch']
        graph=dgl.batch([graph_from_condition(condition,cfg['dataset']['atom_map'],
            n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4)]*batch).to('cuda')
        nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph);basis=centered_orthonormal_basis(n,device='cuda')
        positions=[];initial=[];seeds=[];trees=[];context_trees=[];start=time.perf_counter()
        for begin in range(0,protocol['samples_per_condition'],batch):
            init=[];matrices=[]
            for j in range(batch):
                seed=protocol['evaluation_seed']*1000003+index*100003+begin+j
                x,tree=sample_source(prior,numbers,condition['charge'],condition['spin_multiplicity'],seed)
                init.append(x);seeds.append(seed);trees.append(tree)
                if context_mode:
                    selected=context_tree(prior,numbers,condition['charge'],condition['spin_multiplicity'],tree,
                        seed+protocol['context_seed_offset'],context_mode)
                    context_trees.append(selected);matrices.append(tree_features(n,selected))
            init=torch.stack(init);initial.append(init)
            if context_mode:attach_context(graph,matrices,nbi)
            with torch.no_grad():
                final=(sample_clamped_flow if sampler is None else sampler)(model,graph,nbi,uem,x0=init.reshape(-1,3).cuda().float(),
                    n_ode_steps=protocol['midpoint_steps'],terminal_time=1.,parameterization='displacement')
            final=final.reshape(batch,n,3).double();final=final-final.mean(1,keepdim=True)
            noise_seed=protocol['evaluation_seed']*2000003+index*100003+begin
            noise=torch.randn((batch,n-1,3),device='cuda',dtype=torch.float64,generator=torch.Generator(device='cuda').manual_seed(noise_seed))
            final+=protocol['terminal_noise_std_A']*torch.einsum('nk,bkd->bnd',basis,noise)
            if not torch.isfinite(final).all():raise FloatingPointError('Nonfinite generated structures')
            positions.append(final.cpu())
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        x=torch.cat(positions);x0=torch.cat(initial)
        file=out/f'{method}_c{index}.pt'
        saved=dict(positions=x,initial_positions=x0,auxiliary_tree_edges=trees,seeds=seeds,condition=condition)
        if context_mode:saved.update(context_tree_edges=context_trees,context_mode=context_mode)
        torch.save(saved,file)
        row=dict(method=method,condition_index=index,condition=condition,sample_sha256=sha(file),checkpoint_sha256=checkpoint_hash,
            generation_seconds=seconds,initial_geometry=geometry_counts(x0,numbers),final_geometry=geometry_counts(x,numbers),
            **assess(x,condition,list(range(len(x)))))
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['method','condition_index','graph_supported','geometrically_supported','initial_geometry','final_geometry','generation_seconds']}),flush=True)
    write(out/f'{method}_results.json',dict(complete=True,protocol_sha256=protocol_hash,checkpoint_sha256=checkpoint_hash,rows=rows,
        new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='Nominal source laws are explicitly sampled; latent trees are not chemical bonds. Finite midpoint integration plus noise output has no qualified density or thermal-law claim.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','prior-run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--methods',nargs='+',required=True);p.add_argument('--include-warm',action='store_true')
    p.add_argument('--reuse-fixed-run',type=Path)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());protocol_hash=sha(a.protocol)
    assert protocol['frozen'] and all(m in protocol['methods'] for m in a.methods)
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);a.out.mkdir(parents=True)
    cfg,ds,metadata,order=load_training(a.project,protocol)
    if metadata.selection_info is not None:write(a.out/'selection.json',metadata.selection_info)
    warm_path=a.project/protocol['warm_checkpoint'];assert sha(warm_path)==protocol['warm_checkpoint_sha256']
    warm=torch.load(warm_path,map_location='cpu',weights_only=False)
    assert warm['research_protocol']['position_parameterization']=='displacement' and warm['research_protocol']['data_endpoint_time']==1.
    manifest=a.project/protocol['condition_manifest'];assert sha(manifest)==protocol['condition_manifest_sha256']
    radii=covalent_radii(metadata.atomic_numbers.tolist(),dtype=torch.float32,device='cuda')
    evaluation=a.out/'evaluation';evaluation.mkdir()
    if a.include_warm:
        model=restore_model(cfg,warm)
        evaluate(model,None,'warm',cfg,protocol,protocol_hash,evaluation,protocol['warm_checkpoint_sha256'],manifest)
        del model;torch.cuda.empty_cache()
    completed=[]
    if a.reuse_fixed_run is not None:
        assert 'fixed' in a.methods
        previous=a.reuse_fixed_run.resolve()
        old_selection=json.loads((previous/'selection.json').read_text())
        assert old_selection['selected']==order.tolist()
        old_validation=json.loads((previous/'fixed/validation.json').read_text())
        old_report=json.loads((previous/'evaluation/fixed_results.json').read_text())
        assert old_validation['complete'] and old_report['complete'] and old_report['protocol_sha256']==protocol_hash
        assert sha(previous/'fixed/last.ckpt')==old_validation['checkpoint_sha256']==old_report['checkpoint_sha256']
        assert len(old_report['rows'])==len(protocol['conditions'])
        (a.out/'fixed').symlink_to(previous/'fixed',target_is_directory=True)
        for row in old_report['rows']:
            name=f'fixed_c{row["condition_index"]}.pt'
            assert sha(previous/'evaluation'/name)==row['sample_sha256']
            (evaluation/name).symlink_to(previous/'evaluation'/name)
        (evaluation/'fixed_results.json').symlink_to(previous/'evaluation/fixed_results.json')
        write(a.out/'baseline_reuse.json',dict(source=str(previous),protocol_sha256=protocol_hash,
            checkpoint_sha256=old_validation['checkpoint_sha256'],report_sha256=sha(previous/'evaluation/fixed_results.json'),
            reason='No-block training does not use dynamic attention; preserve its complete identical-protocol training/evaluation after an independent attention-gradient implementation fix.'))
        completed=['fixed']
    for method in a.methods:
        if method in completed:continue
        torch.manual_seed(protocol['fm_seed']);model=restore_model(cfg,warm).train()
        dynamic_mode=protocol.get('dynamic_modes',{}).get(method)
        prior=load_prior('fixed' if dynamic_mode else method,a.prior_run,protocol,protocol_hash)
        if dynamic_mode:
            from cfm_mol.dynamic_tree_attention import patch_dynamic_tree_attention
            torch.manual_seed(protocol['dynamic_initialization_seed'])
            patch_dynamic_tree_attention(model,mode=dynamic_mode,**protocol['dynamic_adapter'])
        context_mode=protocol.get('context_modes',{}).get(method)
        if context_mode:
            from cfm_mol.latent_tree_context import patch_latent_tree_context,context_tree,tree_features
            torch.manual_seed(protocol['context_initialization_seed'])
            patch_latent_tree_context(model,**protocol['context_adapter'])
        assert not (context_mode and dynamic_mode)
        source_kind='fixed' if context_mode or dynamic_mode else method
        model._research_prior_kind=source_kind
        directory=a.out/method;directory.mkdir()
        recipe={**warm['research_protocol'],**protocol,'format':'tree_prior_fm_v1','source_prior_kind':source_kind,
            'pairing_protocol_sha256':None,'tree_protocol_sha256':protocol_hash,'seed':protocol['fm_seed'],
            'warm_sha256':protocol['warm_checkpoint_sha256'],'data_order_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest(),
            'source_prior_configuration':None if prior is None else prior.configuration,'prior_std':1.,
            'purpose':'Matched main-generator source-prior comparison; FM-only, no bond/energy supervision.'}
        if context_mode:
            recipe.update(latent_tree_context=protocol['context_adapter'],context_mode=context_mode,model_variant=method,
                purpose='Actual versus independent latent tree conditioning; same shell source and training data, no bonds or oracle labels.')
            torch.save(model.vector_field.latent_tree_adapter.state_dict(),directory/'adapter_initial.pt')
        if dynamic_mode:
            recipe.update(dynamic_tree_attention=dict(mode=dynamic_mode,**protocol['dynamic_adapter']),model_variant=method,
                purpose='State-dependent global tree attention versus local mass and fixed global affinities; same main FM objective and source, no supplied bonds.')
            torch.save(model.vector_field.dynamic_tree_attention.state_dict(),directory/'dynamic_initial.pt')
        write(directory/'protocol.json',recipe)
        if context_mode:
            adapter=list(model.vector_field.latent_tree_adapter.parameters());ids={id(p) for p in adapter}
            groups=[dict(params=[p for p in model.parameters() if id(p) not in ids],lr=protocol['fm_lr']),
                    dict(params=adapter,lr=protocol['context_lr'])]
            optimizer=torch.optim.AdamW(groups,weight_decay=1e-12)
        elif dynamic_mode:
            adapter=list(model.vector_field.dynamic_tree_attention.parameters());ids={id(p) for p in adapter}
            optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=protocol['fm_lr']),
                dict(params=[p for p in adapter if p.requires_grad],lr=protocol['dynamic_lr'])],weight_decay=1e-12)
        else:optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['fm_lr'],weight_decay=1e-12)
        torch.manual_seed(protocol['fm_seed']+17);start=time.perf_counter()
        for step,index in enumerate(order[:protocol['fm_steps']].tolist(),1):
            graph=ds[index];metadata.attach(graph,[index],1.)
            numbers=metadata.atomic_numbers[graph.ndata['a_1_true'].argmax(-1)].tolist()
            electronic=graph.ndata['electronic_state'][0];q=int(electronic[0]);spin=int(electronic[1])
            x0,tree=sample_source(prior,numbers,q,spin,protocol['fm_seed']*3000017+step)
            matrices=None
            if context_mode:
                selected=context_tree(prior,numbers,q,spin,tree,protocol['fm_seed']*3000017+step+protocol['context_seed_offset'],context_mode)
                matrices=[tree_features(len(numbers),selected)]
            graph=graph.to('cuda');nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
            records=[]
            loss=clamped_fm_loss(model,graph,nbi,uem,terminal_time=1.,parameterization='displacement',prior_positions=x0,prior_edge_context=matrices,
                generator=torch.Generator(device='cuda').manual_seed(protocol['fm_seed']*1000003+step),
                pairing='typed_rotation',pairing_radii=radii[graph.ndata['a_1_true'].argmax(-1)],
                pairing_generator=torch.Generator(device='cuda').manual_seed(protocol['fm_seed']*2000003+step),pairing_diagnostics=records)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite FM objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            row=dict(step=step,processed_index=index,n_atoms=len(numbers),fm_loss=float(loss),gradient_norm=float(norm),
                initial_geometry=geometry_counts(x0[None],numbers),pairing=records,seconds=time.perf_counter()-start)
            if context_mode:
                grads=[p.grad for p in model.vector_field.latent_tree_adapter.parameters() if p.grad is not None]
                row['adapter_gradient_norm']=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%100==0 or step==1:print(json.dumps(dict(method=method,**row)),flush=True)
            if step%1000==0 or step==protocol['fm_steps']:
                if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):raise FloatingPointError('Nonfinite model weights')
                checkpoint=dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),global_step=step,research_protocol=recipe,
                    source_prior=None if prior is None else dict(configuration=prior.configuration,state_dict=prior.state_dict()))
                temporary=directory/'last.ckpt.tmp';torch.save(checkpoint,temporary);temporary.replace(directory/'last.ckpt')
                del checkpoint
        write(directory/'validation.json',dict(complete=True,global_step=protocol['fm_steps'],seconds=time.perf_counter()-start,checkpoint_sha256=sha(directory/'last.ckpt')))
        evaluate(model,prior,method,cfg,protocol,protocol_hash,evaluation,sha(directory/'last.ckpt'),manifest)
        completed.append(method);write(a.out/'progress.json',dict(complete=len(completed)==len(a.methods),protocol_sha256=protocol_hash,completed=completed))
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
