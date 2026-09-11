#!/usr/bin/env python3
"""Matched-source EACF joint-entropy control with a frozen architecture profile.

This is not standalone FAB: the FM source has no evaluated marginal density.
Joint KL and physical marginal KL remain distinct. Every failed update/query is
retained, and physical evaluation always uses the projected target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from eacf.flow.aug_flow_dist import FullGraphSample
from cfm_mol.eacf_reference import recipe_for,build_reference_flow,build_reference_optimizer
from cfm_mol.jax_energy_oracle import make_even_log_target,make_raw_training_log_energy
from cfm_mol.numpy_energy_oracle import NumpyEnergyOracle

jax.config.update('jax_enable_x64',True)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,report):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def statistics(x):
    x=np.asarray(x);return {'mean':float(x.mean()),'row_sem':float(x.std(ddof=1)/len(x)**.5),'parents':len(x)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replica',type=int,choices=[0,1],default=0)
    p.add_argument('--profile',choices=['published','compact'],default='published')
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle-checkpoint',type=Path,required=True)
    p.add_argument('--engineering-smoke',action='store_true');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    root=Path(__file__).resolve().parents[2]
    experiment_name=('eacf_joint_refinement_protocol_v1.json' if args.profile=='published'
                     else 'eacf_compact_refinement_protocol_v1.json')
    experiment_path=root/'research/evidence'/experiment_name
    experiment=json.loads(experiment_path.read_text())
    manifest_path=args.source/'manifest.json';manifest=json.loads(manifest_path.read_text())
    array_path=args.source/'arrays.npz'
    if not manifest['complete'] or sha(array_path)!=manifest['arrays_sha256']:raise ValueError('Unqualified source export')
    if not experiment['frozen'] or sha(manifest_path)!=experiment['source_export_sha256']:
        raise ValueError('The declared source export changed')
    physical=root/'research/evidence/parity_training_protocol_v1.json'
    if sha(physical)!=experiment['physical_protocol_sha256'] or manifest['target']!=json.loads(physical.read_text()):
        raise ValueError('The physical target differs from the frozen comparison')
    if manifest['condition_index']!=experiment['condition_index']:raise ValueError('Wrong baseline condition')
    protocol=manifest['target'];condition=manifest['condition'];n=len(condition['numbers'])
    if sha(args.oracle_checkpoint)!=protocol['raw_oracle_sha256']:raise ValueError('Wrong physical oracle')
    recipe=recipe_for(n,engineering=False,profile=args.profile)
    recipe_keys={'n_layers':'layers','n_blocks':'gnn_blocks','mlp_units':'mlp_units',
                 'n_invariant_feat_hidden':'invariant_hidden','spline_num_bins':'spline_bins'}
    if any(recipe[k]!=experiment[v] for k,v in recipe_keys.items()):
        raise ValueError('Architecture differs from the frozen experiment')
    if 'embedding_dim' in experiment and recipe['embedding_dim']!=experiment['embedding_dim']:
        raise ValueError('Embedding differs from the frozen experiment')
    if experiment['batch']!=16:raise ValueError('Source streams prescribe16 parents per update')
    steps,count=((experiment['engineering_steps'],experiment['engineering_evaluation_parents'])
                 if args.engineering_smoke else (experiment['steps'],experiment['evaluation_parents']))
    expected_queries=16*steps+4*count
    arrays=np.load(array_path);replica=args.replica
    xtrain=arrays['training_positions'];xdev=arrays['development_positions'][:count]*arrays['evaluation_signs'][:count,None,None]
    source_files=[Path(__file__).resolve(),root/'cfm_mol/eacf_reference.py',root/'cfm_mol/jax_energy_oracle.py',
        root/'cfm_mol/numpy_energy_oracle.py',root/'scripts/research/oracle_worker.py']
    report={'complete':False,'scope':__doc__,'kind':'eacf_spherical_joint','recipe':recipe,'condition':condition,
        'architecture_profile':args.profile,
        'experiment_protocol_sha256':sha(experiment_path),
        'replica':replica,'steps':steps,'batch':16,'evaluation_parents':count,'engineering_only':args.engineering_smoke,
        'source_export_sha256':sha(manifest_path),'source_results_sha256':manifest['source_results_sha256'],
        'source_checkpoint_sha256':manifest['source_checkpoint_sha256'],'training_sha256':manifest['training_sha256'],
        'evaluation_sha256':manifest['evaluation_sha256'],'refinement_protocol_sha256':manifest['refinement_protocol_sha256'],
        'source_protocol_sha256':manifest['source_protocol_sha256'],'source_kind':protocol['source_kind'],
        'target_kind':protocol['target_kind'],'kT_eV':protocol['kT_eV'],'restraint_eV_A2':protocol['restraint_eV_A2'],
        'source_sha256':{str(f.relative_to(root)):sha(f) for f in source_files},
        'init_seed':9601+replica,'aux_training_seed':9605+replica,
        'aux_evaluation_seed':9607+replica+100*manifest['condition_index'],
        'expected_oracle_evaluations':expected_queries,'oracle_evaluations_so_far':0,'history':[],
        'optimizer':'Pinned upstream warmup-cosine Adam:2e-5 ->2e-4 ->2e-5 over1000 attempts, warmup30; dynamic median clipping2/ignore10/window100',
        'scientific_submission_ready':False,
        'limitations':['Joint change upper-bounds physical marginal change; do not rank unlike KLs as equal.',
            'Recorded EACF architecture profile, matched-source adaptation; not standalone FAB.',
            'GPU oracle passed selected CPU-agreement checks. Legacy CPU-oracle wall times are not a matched timing baseline.',
            'One source condition/seed or a smoke does not establish broad performance.']}
    write(output,report);start=time.perf_counter()
    flow=build_reference_flow(recipe)
    features=jnp.asarray(condition['numbers'],dtype=int)[None,:,None]
    initial=jnp.asarray(xtrain[:16],dtype=jnp.float64)
    params=flow.init(jax.random.PRNGKey(report['init_seed']),FullGraphSample(positions=initial,features=jnp.repeat(features,16,axis=0)))
    optimizer,learning_rate=build_reference_optimizer(1000)
    opt_state=optimizer.init(params.bijector);theta=params.bijector
    report['parameters']=sum(p.size for p in jax.tree_util.tree_leaves(theta))
    if 'parameters' in experiment and report['parameters']!=experiment['parameters']:
        report['failure']='Parameter count differs from the frozen architecture'
        write(output,report);raise ValueError(report['failure'])
    def joint(x,a):
        return FullGraphSample(positions=jnp.stack([x,a],axis=-2),features=jnp.repeat(features[...,None],len(x),axis=0))
    def aux_log_prob(x,a):
        sample=FullGraphSample(positions=x,features=jnp.repeat(features,len(x),axis=0))
        return flow.aux_target_log_prob_apply(params.aux_target,sample,a[:,:,None,:])
    transport=jax.jit(flow.bijector_forward_and_log_det_with_extra_apply)
    oracle=NumpyEnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
        numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],
        device='cuda',batch_size=32,stderr_path=args.out/'oracle.log')
    report['oracle_runtime']=oracle.handshake
    if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype')!='torch.float32':
        oracle.close();raise ValueError('Baseline requires qualified full-precision oracle settings')
    raw=make_raw_training_log_energy(oracle,kT=protocol['kT_eV'],restraint=protocol['restraint_eV_A2'],energy_zero_eV=manifest['energy_zero_eV'])
    even=make_even_log_target(oracle,kT=protocol['kT_eV'],restraint=protocol['restraint_eV_A2'],energy_zero_eV=manifest['energy_zero_eV'])
    def objective(theta,x,a):
        y,volume,extra=flow.bijector_forward_and_log_det_with_extra_apply(theta,joint(x,a))
        physical,aux=y.positions[:,:,0,:],y.positions[:,:,1,:]
        component=jnp.mean(-raw(physical)-aux_log_prob(physical,aux)-volume)
        regularizer=jnp.mean(extra.aux_loss)
        return component+recipe['aux_regularizer_weight']*regularizer,(component,regularizer)
    @jax.jit
    def update(theta,state,x,a):
        (loss,parts),gradient=jax.value_and_grad(objective,has_aux=True)(theta,x,a)
        norm=optax.global_norm(gradient)
        changes,state=optimizer.update(gradient,state,theta)
        return optax.apply_updates(theta,changes),state,loss,parts,norm
    try:
        for step in range(steps):
            x=jnp.asarray(xtrain[arrays[f'indices_s{replica}'][step]]*arrays[f'signs_s{replica}'][step,:,None,None])
            noise=jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(report['aux_training_seed']),step),x.shape,dtype=jnp.float64)
            a=x+recipe['aux_scale']*noise
            theta,opt_state,loss,parts,norm=update(theta,opt_state,x,a);loss.block_until_ready()
            if not np.isfinite(float(loss)):raise FloatingPointError('Nonfinite EACF objective')
            if oracle.evaluated!=16*(step+1):raise RuntimeError('Unexpected raw training query count')
            if step==0 or (step+1)%20==0 or step+1==steps:
                report['history'].append({'step':step+1,'objective':float(loss),'joint_component':float(parts[0]),
                    'basis_regularizer':float(parts[1]),'gradient_norm':float(norm) if np.isfinite(float(norm)) else None,
                    'ignored_updates':int(opt_state.ignored_grads_count),'learning_rate':float(learning_rate(step)),
                    'seconds':time.perf_counter()-start})
                report['oracle_evaluations_so_far']=oracle.evaluated;write(output,report)
                print(json.dumps(report['history'][-1]),flush=True)
        x=jnp.asarray(xdev,dtype=jnp.float64)
        a=x+recipe['aux_scale']*jax.random.normal(jax.random.PRNGKey(report['aux_evaluation_seed']),x.shape,dtype=jnp.float64)
        position_parts=[];aux_parts=[];volumes=[]
        for begin in range(0,count,32):
            y,v,_=transport(theta,joint(x[begin:begin+32],a[begin:begin+32]))
            position_parts.append(y.positions[:,:,0,:]);aux_parts.append(y.positions[:,:,1,:]);volumes.append(v)
        y=jnp.concatenate(position_parts);b=jnp.concatenate(aux_parts);volume=jnp.concatenate(volumes)
        base_logp=np.asarray(even(x));adapted_logp=np.asarray(even(y))
        base_aux=np.asarray(aux_log_prob(x,a));adapted_aux=np.asarray(aux_log_prob(y,b))
        change=-adapted_logp+base_logp-adapted_aux+base_aux-np.asarray(volume)
        np.savez(args.out/'samples.npz',base_positions=np.asarray(x),base_auxiliary=np.asarray(a),
            positions=np.asarray(y),auxiliary=np.asarray(b),log_volume=np.asarray(volume),
            base_log_target=base_logp,adapted_log_target=adapted_logp,base_aux_log_prob=base_aux,
            adapted_aux_log_prob=adapted_aux,paired_joint_kl_change=change,
            parent_ids=arrays['development_parent_ids'][:count],inversion_signs=arrays['evaluation_signs'][:count])
        sample=joint(x[:4],a[:4]);mapped,vl,_=transport(theta,sample)
        reconstructed,iv,_=flow.bijector_inverse_and_log_det_with_extra_apply(theta,mapped)
        np.testing.assert_allclose(reconstructed.positions,sample.positions,atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(vl+iv,0,atol=1e-7,rtol=0)
        mirror,mv,_=transport(theta,sample._replace(positions=-sample.positions))
        np.testing.assert_allclose(mirror.positions,-mapped.positions,atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(mv,vl,atol=1e-7,rtol=1e-7)
        order=jnp.arange(n-1,-1,-1)
        permuted,pv,_=transport(theta,sample._replace(positions=sample.positions[:,order],features=sample.features[:,order]))
        np.testing.assert_allclose(permuted.positions,mapped.positions[:,order],atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(pv,vl,atol=1e-7,rtol=1e-7)
        checkpoint=args.out/'adapter.pkl'
        with checkpoint.open('wb') as f:pickle.dump({'recipe':recipe,'theta':jax.device_get(theta),
            'optimizer':jax.device_get(opt_state),'condition':condition},f)
        with checkpoint.open('rb') as f:saved=pickle.load(f)
        rebuilt=build_reference_flow(saved['recipe'])
        replay,rv,_=rebuilt.bijector_forward_and_log_det_with_extra_apply(saved['theta'],sample)
        np.testing.assert_allclose(replay.positions,mapped.positions,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(rv,vl,atol=1e-8,rtol=1e-8)
        if oracle.evaluated!=expected_queries:raise RuntimeError('Final physical query count differs')
        if float(jnp.max(jnp.abs(y.mean(1))))>1e-8:raise RuntimeError('Physical COM constraint failed')
        if args.engineering_smoke and int(opt_state.ignored_grads_count)==steps:
            raise RuntimeError('Engineering smoke made no accepted optimizer update')
        report.update(complete=True,paired_joint_kl_change=statistics(change),
            physical_COM_error=float(jnp.max(jnp.abs(y.mean(1)))),inverse_error=float(jnp.max(jnp.abs(reconstructed.positions-sample.positions))),
            checkpoint_replay_passed=True,oracle_evaluations=oracle.evaluated,
            ignored_updates=int(opt_state.ignored_grads_count),seconds=time.perf_counter()-start,
            artifacts={p.name:sha(p) for p in [checkpoint,args.out/'samples.npz']})
        write(output,report)
    except Exception as exc:
        report.update(complete=False,failure=f'{type(exc).__name__}: {exc}',oracle_evaluations=oracle.evaluated,
            requested_oracle_evaluations=oracle.requested_evaluations)
        write(output,report);raise
    finally:oracle.close()


if __name__=='__main__':main()
