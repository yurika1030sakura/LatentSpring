#!/usr/bin/env python3
"""Qualify real JAX energy/force callbacks and one upstream EACF optimizer step.

Prespecified saved parents from conditions0 and7. This is interface engineering,
not an energy-trained external baseline or a scientific performance comparison.
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

from cfm_mol.numpy_energy_oracle import NumpyEnergyOracle
from cfm_mol.jax_energy_oracle import make_even_log_target

jax.config.update('jax_enable_x64',True)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,report):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def finite_tree(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))


def eacf_step(log_target,x,numbers,out):
    from eacf.flow.aug_flow_dist import FullGraphSample
    from eacf.flow.build_flow import FlowDistConfig,build_flow
    from eacf.utils.testing import get_minimal_nets_config
    n=x.shape[1]
    nets=get_minimal_nets_config('egnn')._replace(num_discrete_feat=119)
    config=FlowDistConfig(dim=3,n_aug=1,nodes=n,n_layers=1,nets_config=nets,
        type='spherical',identity_init=True,scaling_layer=False,
        kwargs={'spherical':{'reflection_invariant':True,'n_inner_transforms':1}})
    flow=build_flow(config)
    features=jnp.broadcast_to(jnp.asarray(numbers,dtype=int)[None,:,None],(len(x),n,1))
    params=flow.init(jax.random.PRNGKey(9471),FullGraphSample(positions=x,features=features))
    auxiliary=jax.random.normal(jax.random.PRNGKey(9472),x.shape,dtype=jnp.float64)
    joint=FullGraphSample(positions=jnp.stack([x,auxiliary],axis=-2),features=features[...,None])
    def loss(bijector):
        y,volume,_=flow.bijector_forward_and_log_det_with_extra_apply(bijector,joint)
        physical,aux=y.positions[:,:,0,:],y.positions[:,:,1,:]
        # Target auxiliary is an independent unit Gaussian in3N dimensions.
        # Initial auxiliary/source constants cancel for the fixed-source joint change.
        return (-log_target(physical)+.5*jnp.square(aux).sum((1,2))-volume).mean()
    value,gradient=jax.jit(jax.value_and_grad(loss))(params.bijector)
    value.block_until_ready()
    if not finite_tree((value,gradient)):raise ValueError('Nonfinite EACF energy gradient')
    optimizer=optax.adam(1e-5);state=optimizer.init(params.bijector)
    updates,state=optimizer.update(gradient,state,params.bijector)
    changed=optax.apply_updates(params.bijector,updates)
    after=jax.jit(loss)(changed);after.block_until_ready()
    if not finite_tree(after):raise ValueError('Nonfinite updated EACF objective')
    forward=jax.jit(flow.bijector_forward_and_log_det_with_extra_apply)
    y,volume,_=forward(changed,joint)
    restored,back_volume,_=flow.bijector_inverse_and_log_det_with_extra_apply(changed,y)
    np.testing.assert_allclose(restored.positions,joint.positions,atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(volume+back_volume,0,atol=2e-8,rtol=0)
    inverted,iv,_=forward(changed,joint._replace(positions=-joint.positions))
    np.testing.assert_allclose(inverted.positions,-y.positions,atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(iv,volume,atol=2e-8,rtol=2e-8)
    order=jnp.arange(n-1,-1,-1)
    permuted,pv,_=forward(changed,joint._replace(positions=joint.positions[:,order],features=joint.features[:,order]))
    np.testing.assert_allclose(permuted.positions,y.positions[:,order],atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(pv,volume,atol=2e-8,rtol=2e-8)
    checkpoint=out/'eacf_engineering_step.pkl'
    with checkpoint.open('wb') as f:pickle.dump({'config':config,'bijector':jax.device_get(changed),'optimizer':jax.device_get(state)},f)
    return {'complete':True,'layers':1,'configuration':repr(config),'parameters':sum(v.size for v in jax.tree_util.tree_leaves(changed)),
        'gradient_norm':float(optax.global_norm(gradient)),'before_objective':float(value),'after_objective':float(after),
        'inverse_max_error':float(jnp.max(jnp.abs(restored.positions-joint.positions))),
        'inversion_and_joint_permutation_checked':True,'checkpoint_sha256':sha(checkpoint),
        'objective_scope':'fixed-source JOINT relative-KL training component; not exact marginal KL',
        'engineering_only':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle-checkpoint',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((args.panel/'manifest.json').read_text());protocol=manifest['protocol']
    if not manifest['complete'] or [r['index'] for r in manifest['conditions']]!=[0,7]:raise ValueError('Wrong fixed input panel')
    if sha(args.oracle_checkpoint)!=protocol['raw_oracle_sha256']:raise ValueError('Oracle checkpoint differs')
    root=Path(__file__).resolve().parents[2]
    worker=root/'scripts/research/oracle_worker.py'
    report={'complete':False,'scope':__doc__,'rows':[],'engineering_only':True,'scientific_submission_ready':False,
        'panel_manifest_sha256':sha(args.panel/'manifest.json'),'protocol_sha256':manifest['protocol_sha256'],
        'fd_steps_A':[.003,.0015],'fd_absolute_tolerance_eV_A':.005,'fd_relative_tolerance':.02,
        'cached_energy_tolerance_eV':1e-4,'expected_raw_queries':64,'actual_raw_queries':0,
        'limitations':['Selected geometry checks do not establish global potential accuracy or a qualified external benchmark.',
            'Pure callbacks may be repeated/elided; report actual acknowledged worker calls after blocking.',
            'EACF engineering uses a one-layer minimal network, not the full production recipe.',
            'The frozen-source augmented objective is joint KL; a conditional auxiliary gap separates it from marginal KL.']}
    write(output,report)
    start=time.perf_counter()
    for item in manifest['conditions']:
        panel=args.panel/item['panel']
        if sha(panel)!=item['panel_sha256']:raise ValueError('Input panel changed')
        data=np.load(panel);x=jnp.asarray(data['positions'],dtype=jnp.float64);x=x-x.mean(1,keepdims=True)
        condition=item['condition'];row={'index':item['index'],'condition':condition,'complete':False}
        report['rows'].append(row)
        with NumpyEnergyOracle(args.oracle_python,worker,args.oracle_checkpoint,
                numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],
                stderr_path=args.out/f'oracle_{item["index"]:02d}.log') as oracle:
            try:
                fn=make_even_log_target(oracle,kT=protocol['kT_eV'],restraint=protocol['restraint_eV_A2'])
                weights=jnp.array([.7,-1.3],dtype=jnp.float64)
                value,gradient=jax.jit(jax.value_and_grad(lambda z:jnp.sum(weights*fn(z))))(x)
                value.block_until_ready()
                cached=-(data['energy_eV']+.5*protocol['restraint_eV_A2']*np.square(np.asarray(x)).sum((1,2)))/protocol['kT_eV']
                values,individual_gradient=jax.jit(jax.vmap(jax.value_and_grad(fn)))(x)
                values.block_until_ready()
                np.testing.assert_allclose(gradient,weights[:,None,None]*individual_gradient,atol=1e-7,rtol=1e-7)
                error=float(np.max(np.abs(np.asarray(values)-cached))*protocol['kT_eV'])
                if error>report['cached_energy_tolerance_eV']:raise ValueError('Projected energy fails saved-source replay')
                mirror=jax.jit(fn)(-x);mirror.block_until_ready()
                np.testing.assert_allclose(mirror,values,atol=1e-7,rtol=1e-10)
                direction=jax.random.normal(jax.random.PRNGKey(9470+item['index']),x.shape,dtype=jnp.float64)
                direction-=direction.mean(1,keepdims=True)
                direction/=jnp.linalg.norm(direction,axis=(1,2),keepdims=True)
                slope=np.sum(np.asarray(individual_gradient*direction),axis=(1,2))*protocol['kT_eV']
                ladder=[]
                for h in report['fd_steps_A']:
                    plus=np.asarray(jax.jit(fn)(x+h*direction));minus=np.asarray(jax.jit(fn)(x-h*direction))
                    finite_difference=(plus-minus)/(2*h)*protocol['kT_eV']
                    residual=np.abs(finite_difference-slope)
                    if np.any(residual>report['fd_absolute_tolerance_eV_A']+report['fd_relative_tolerance']*np.abs(slope)):
                        raise ValueError('Real callback force fails physical finite-difference ladder')
                    ladder.append({'step_A':h,'analytic_log_target_slope_times_kT':slope.tolist(),
                        'finite_difference_times_kT':finite_difference.tolist(),'absolute_error_eV_A':residual.tolist()})
                row.update(cached_energy_error_eV=error,finite_differences=ladder,
                    batch_vjp_checked=True,gradient_COM_error=float(jnp.max(jnp.abs(gradient.mean(1)))))
                if item['index']==0:row['eacf_step']=eacf_step(fn,x,condition['numbers'],args.out)
                row.update(complete=True,acknowledged_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
            except Exception as exc:
                row.update(failure=f'{type(exc).__name__}: {exc}',acknowledged_raw_queries=oracle.evaluated,
                    requested_raw_queries=oracle.requested_evaluations)
                report['actual_raw_queries']=sum(r.get('acknowledged_raw_queries',0) for r in report['rows'])
                write(output,report);raise
        report['actual_raw_queries']=sum(r['acknowledged_raw_queries'] for r in report['rows'])
        write(output,report);print(json.dumps({'index':item['index'],'queries':row['acknowledged_raw_queries'],'complete':True}),flush=True)
    if report['actual_raw_queries']!=report['expected_raw_queries']:raise ValueError('Observed callback execution count differs from qualified recipe')
    report.update(complete=True,seconds=time.perf_counter()-start);write(output,report)


if __name__=='__main__':main()
