#!/usr/bin/env python3
"""Known-probability COM test of learned coupling, not molecular validation."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol.latent_mass_coupling import (ResidualSurrogate,ShapeTwist,GaussianCoupling,rotate,
    reference_flow,toy_energy,log_weight,analytic_angle,exact_targets,joint_reverse_kl)
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def draw(n,g):return torch.randn(n,2,3,dtype=torch.float64,generator=g)


def train_surrogates(scenario,seed,protocol):
    models=[];data=[];history=[]
    for component in [0,1]:
        torch.manual_seed(seed+component);g=torch.Generator().manual_seed(seed+101+component)
        z=draw(protocol['labels_per_component'],g);x,logq=reference_flow(z,component)
        energy=toy_energy(x,component,scenario);labels=-energy-logq
        model=ResidualSurrogate(protocol['hidden']).double();optimizer=torch.optim.Adam(model.parameters(),lr=protocol['surrogate_lr'])
        trace=[]
        for step in range(protocol['surrogate_steps']):
            idx=torch.randint(len(z),(protocol['batch_size'],),generator=g)
            loss=(model(z[idx])-labels[idx]).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();trace.append(float(loss))
        model.eval();model.requires_grad_(False)
        data.append(dict(z=z,x=x,logq=logq,energy=energy,labels=labels));models.append(model);history.append(trace)
    return models,data,history


def score(surrogates,z,y,normalizers):
    a=surrogates[0](z).clamp(-8,8).exp()/normalizers[0]
    b=surrogates[1](y).clamp(-8,8).exp()/normalizers[1]
    return (a*b).mean()


def fit_couplings(surrogates,seed,protocol):
    g=torch.Generator().manual_seed(seed+301);bank=draw(protocol['coupling_bank'],g);noise=draw(len(bank),g)
    with torch.no_grad():normalizers=torch.stack([m(bank).clamp(-8,8).exp().mean() for m in surrogates])
    best=None;grid=[]
    with torch.no_grad():
        for reflection in [False,True]:
            for j in range(protocol['angle_grid']):
                angle=-math.pi/2+(j+.5)*math.pi/protocol['angle_grid']
                value=float(score(surrogates,bank,rotate(bank,bank.new_tensor(angle),reflection),normalizers))
                grid.append([angle,reflection,value])
                if best is None or value>best[2]:best=(angle,reflection,value)
    fitted={};traces={}
    for kind in ['gaussian','nonlinear']:
        torch.manual_seed(seed+401)
        model=(GaussianCoupling(initial_angle=best[0],reflection=best[1]) if kind=='gaussian' else
               ShapeTwist(hidden=protocol['hidden'],initial_angle=best[0],reflection=best[1])).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=protocol['coupling_lr']);g=torch.Generator().manual_seed(seed+501);trace=[]
        for step in range(protocol['coupling_steps']):
            idx=torch.randint(len(bank),(protocol['batch_size'],),generator=g);z=bank[idx];eps=noise[idx]
            loss=-score(surrogates,z,model(z,eps),normalizers)
            if not torch.isfinite(loss):raise ValueError('Nonfinite emulated coupling objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
            trace.append(dict(step=step,objective=float(loss),gradient_norm=float(norm)))
        model.eval();model.requires_grad_(False);fitted[kind]=model;traces[kind]=trace
    return fitted,dict(best_constant=dict(angle=best[0],reflection=best[1],surrogate_cross_moment=best[2]),
        grid=grid,normalizers=normalizers.tolist(),traces=traces,bank_seed=seed+301)


@torch.no_grad()
def evaluate(surrogates,couplings,fit,scenario,seed,protocol):
    truth=exact_targets();rows={};arrays={};reference_calls=0
    g=torch.Generator().manual_seed(seed+701);check=draw(protocol['surrogate_check_samples'],g)
    surrogate_error={str(i):float((surrogates[i](check)-log_weight(check,i,scenario)).square().mean().sqrt()) for i in [0,1]}
    for n in protocol['calibration_sizes']:
        g=torch.Generator().manual_seed(seed+10000+n)
        z=draw(protocol['repetitions']*n,g);eps=draw(len(z),g);a=log_weight(z,0,scenario).reshape(-1,n)
        reference_calls+=len(z)
        methods={};stored={}
        for name in ['independent','identity','constant','gaussian','nonlinear','oracle']:
            if name=='independent':y=eps
            elif name=='identity':y=z
            elif name=='constant':y=rotate(z,z.new_tensor(fit['best_constant']['angle']),fit['best_constant']['reflection'])
            elif name=='oracle':y=rotate(z,analytic_angle(z,scenario))
            else:y=couplings[name](z,eps)
            b=log_weight(y,1,scenario).reshape(-1,n)
            if name=='identity':b_identity=b
            logz0=torch.logsumexp(a,1)-math.log(n);logz1=torch.logsumexp(b,1)-math.log(n)
            ratio=logz1-logz0;mass=ratio.sigmoid();err=mass-truth['component_mass'][1]
            ratio_err=ratio-truth['log_normalizer_ratio']
            kl=[joint_reverse_kl([1-float(p),float(p)],truth) for p in mass]
            methods[name]=dict(mass_RMSE=float(err.square().mean().sqrt()),mass_bias=float(err.mean()),
                log_ratio_RMSE=float(ratio_err.square().mean().sqrt()),log_ratio_variance=float(ratio.var(unbiased=False)),
                normalizer_means=[float(logz0.exp().mean()),float(logz1.exp().mean())],mean_joint_reverse_KL=float(np.mean(kl)),
                target_calls_per_calibration=2*n)
            stored[name]=dict(component_mass=mass,log_ratio=ratio,normalizers=torch.stack([logz0.exp(),logz1.exp()],1))
            reference_calls+=len(z)
        # Mean-work/variational weighting is a diagnostic, not an IS estimator.
        mean_work_mass=(b_identity.mean(1)-a.mean(1)).sigmoid()
        methods['mean_work']=dict(mass_RMSE=float((mean_work_mass-truth['component_mass'][1]).square().mean().sqrt()),
            mass_bias=float(mean_work_mass.mean()-truth['component_mass'][1]),target_calls_per_calibration=2*n)
        comparisons={}
        for control in ['independent','identity','constant','gaussian']:
            new_error=(stored['nonlinear']['component_mass']-truth['component_mass'][1]).square()
            old_error=(stored[control]['component_mass']-truth['component_mass'][1]).square()
            difference=new_error-old_error;se=float(difference.std(unbiased=True)/math.sqrt(len(difference)))
            comparisons['nonlinear minus '+control]=dict(mean_squared_mass_error_difference=float(difference.mean()),
                paired_MC_normal95=[float(difference.mean())-1.96*se,float(difference.mean())+1.96*se],
                RMSE_ratio=methods['nonlinear']['mass_RMSE']/methods[control]['mass_RMSE'])
        rows[str(n)]=dict(methods=methods,comparisons=comparisons);arrays[str(n)]=stored
    # One explicit generator sample check. Within-component shapes stay fixed.
    chosen_n=protocol['generation_calibration_size'];generation={}
    for name in ['identity','constant','gaussian','nonlinear']:
        mass=float(arrays[str(chosen_n)][name]['component_mass'][0]);g=torch.Generator().manual_seed(seed+20001)
        labels=(torch.rand(protocol['generation_samples'],generator=g,dtype=torch.float64)<mass).long()
        z=draw(len(labels),g);x=torch.zeros(len(labels),3,3,dtype=torch.float64)
        for component in [0,1]:
            mask=labels==component
            if mask.any():x[mask]=reference_flow(z[mask],component)[0]
        radius=x.square().sum((1,2)).sqrt();classified=(radius>2.5).long()
        assert torch.equal(classified,labels) and float(x.mean(1).abs().max())<1e-12
        generation[name]=dict(calibrated_mass=mass,empirical_mass=float(labels.double().mean()),
            samples=len(labels),support_retention=1.,max_COM_error=float(x.mean(1).abs().max()),
            conditional_shapes_corrected=False,joint_reverse_KL=joint_reverse_kl([1-mass,mass],truth))
    return dict(exact=truth,surrogate_log_weight_RMSE=surrogate_error,calibration=rows,generation=generation,
        analytic_evaluation_calls=reference_calls+2*protocol['surrogate_check_samples'],
        mean_work_infinite_data_mass_bias=truth['fixed_shape_reverse_KL_optimal_mass'][1]-truth['component_mass'][1],
        exact_mass_fixed_shape_joint_reverse_KL=joint_reverse_kl(truth['component_mass'],truth),
        variational_optimal_fixed_shape_joint_reverse_KL=joint_reverse_kl(truth['fixed_shape_reverse_KL_optimal_mass'],truth)),arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--replica',type=int,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());assert protocol['frozen'] and a.replica in [0,1,2]
    assert protocol['target']['kappas']==[1.,2.5] and protocol['target']['relative_scale']==.4
    assert protocol['target']['shells']==[[1,2],[3,4]] and protocol['target']['constant_angle']==.4
    assert protocol['target']['nonlinear_angle']=='1.1*tanh(1.5*log(sum(z^2)/6))'
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'results.json').exists():raise FileExistsError(a.out)
    for scenario_index,scenario in enumerate(protocol['scenarios']):
        directory=a.out/scenario
        if directory.exists():raise FileExistsError(directory)
        directory.mkdir();seed=protocol['seeds'][a.replica]+100000*scenario_index;start=time.monotonic()
        surrogates,data,history=train_surrogates(scenario,seed,protocol)
        couplings,fit=fit_couplings(surrogates,seed,protocol)
        evaluated,arrays=evaluate(surrogates,couplings,fit,scenario,seed,protocol)
        saved=dict(scenario=scenario,seed=seed,protocol_sha256=sha(a.protocol),fit=fit,data=data,surrogate_training=history,
            surrogate_states=[m.state_dict() for m in surrogates],
            couplings={name:dict(configuration=m.configuration,state_dict=m.state_dict()) for name,m in couplings.items()})
        torch.save(saved,directory/'models.pt');torch.save(arrays,directory/'calibration.pt')
        report=dict(complete=True,replica=a.replica,scenario=scenario,seed=seed,protocol_sha256=sha(a.protocol),models_sha256=sha(directory/'models.pt'),
            calibration_sha256=sha(directory/'calibration.pt'),analytic_training_calls=2*protocol['labels_per_component'],
            new_molecular_oracle_calls=0,elapsed_seconds=time.monotonic()-start,scientific_submission_ready=False,**evaluated)
        write(directory/'results.json',report)
        print(json.dumps(dict(complete=True,scenario=scenario,replica=a.replica,seconds=report['elapsed_seconds'],surrogate_error=evaluated['surrogate_log_weight_RMSE'],
            mass_RMSE64={k:v['mass_RMSE'] for k,v in evaluated['calibration']['64']['methods'].items()})),flush=True)
    write(a.out/'results.json',dict(complete=True,replica=a.replica,protocol_sha256=sha(a.protocol),new_molecular_oracle_calls=0,scientific_submission_ready=False))


if __name__=='__main__':main()
