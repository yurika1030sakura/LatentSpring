#!/usr/bin/env python3
"""Equal-total-query online coupling test; every target observation is retained."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol.latent_mass_coupling import ResidualSurrogate,ShapeTwist,GaussianCoupling,rotate,log_weight,exact_targets
from scripts.research.latent_mass_calibration import draw,score
from scripts.research.check_binned_mass_coupling import fit as fit_bins,transform as apply_bins
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def update_surrogates(models,zs,labels,seed,protocol):
    for component in [0,1]:
        optimizer=torch.optim.Adam(models[component].parameters(),lr=protocol['surrogate_lr'])
        models[component].requires_grad_(True);models[component].train()
        g=torch.Generator().manual_seed(seed+component)
        z=torch.cat(zs[component]);y=torch.cat(labels[component])
        for step in range(protocol['surrogate_steps']):
            idx=torch.randint(len(z),(protocol['batch_size'],),generator=g)
            loss=(models[component](z[idx])-y[idx]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite online surrogate loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        models[component].eval();models[component].requires_grad_(False)


def update_coupling(models,kind,previous,seed,protocol):
    bank=draw(protocol['coupling_bank'],torch.Generator().manual_seed(seed+301))
    noise=draw(len(bank),torch.Generator().manual_seed(seed+302))
    with torch.no_grad():normalizers=torch.stack([m(bank).clamp(-8,8).exp().mean() for m in models])
    if kind=='binned':
        return fit_bins(models,dict(seed=seed,fit=dict(normalizers=normalizers.tolist())),protocol,protocol['bins'])
    best=None
    with torch.no_grad():
        for reflection in [False,True]:
            for j in range(protocol['angle_grid']):
                angle=-math.pi/2+(j+.5)*math.pi/protocol['angle_grid']
                value=float(score(models,bank,rotate(bank,bank.new_tensor(angle),reflection),normalizers))
                if best is None or value>best[0]:best=(value,angle,reflection)
    if kind=='constant':return dict(angle=best[1],reflection=best[2])
    if previous is None:
        torch.manual_seed(seed+401)
        model=(GaussianCoupling(best[1],best[2]) if kind=='gaussian' else ShapeTwist(protocol['hidden'],best[1],best[2])).double()
    else:model=previous
    model.requires_grad_(True);optimizer=torch.optim.Adam(model.parameters(),lr=protocol['coupling_lr'])
    g=torch.Generator().manual_seed(seed+501)
    for step in range(protocol['coupling_steps']):
        idx=torch.randint(len(bank),(protocol['batch_size'],),generator=g)
        loss=-score(models,bank[idx],model(bank[idx],noise[idx]),normalizers)
        if not torch.isfinite(loss):raise ValueError('Nonfinite online coupling objective')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
    model.eval();model.requires_grad_(False);return model


def run_history(method,scenario,seed,protocol):
    torch.manual_seed(seed+11)
    models=[ResidualSurrogate(protocol['hidden']).double() for _ in [0,1]]
    zs=[[],[]];labels=[[],[]];coupling=None;checkpoints={};batch_trace=[];start=time.monotonic()
    for stage in range(protocol['total_pairs']//protocol['pairs_per_batch']):
        n=protocol['pairs_per_batch'];g=torch.Generator().manual_seed(seed+1001+stage*100)
        z=draw(n,g);eps=draw(n,g)
        # Current coupling depends only on completed earlier batches.
        with torch.no_grad():
            if method=='independent':y=eps
            elif method=='identity' or coupling is None:y=z
            elif method=='constant':y=rotate(z,z.new_tensor(coupling['angle']),coupling['reflection'])
            elif method=='binned':y=apply_bins(z,coupling)
            else:y=coupling(z,eps)
            a=log_weight(z,0,scenario);b=log_weight(y,1,scenario)
        zs[0].append(z);zs[1].append(y);labels[0].append(a);labels[1].append(b)
        completed=(stage+1)*n
        # Never re-evaluate old points under a newly fitted coupling.
        logz=[float(torch.logsumexp(torch.cat(v),0)-math.log(completed)) for v in labels]
        checkpoints[str(completed)]=dict(target_calls=2*completed,log_normalizers=logz,
            component_mass=float(torch.tensor(logz[1]-logz[0],dtype=torch.float64).sigmoid()))
        batch_trace.append(dict(stage=stage,seed=seed+1001+stage*100,learned_from_previous_pairs=stage*n,
            input_z=z,second_z=y,log_weights=torch.stack([a,b],1)))
        if completed<protocol['total_pairs'] and method not in ['independent','identity']:
            fit_seed=seed+2001+stage*100
            update_surrogates(models,zs,labels,fit_seed,protocol)
            coupling=update_coupling(models,method,coupling,fit_seed,protocol)
    return dict(method=method,scenario=scenario,seed=seed,checkpoints=checkpoints,batches=batch_trace,seconds=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--chunk',type=int,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());assert protocol['frozen'];torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);rows=[];data=[]
    for scenario_index,scenario in enumerate(protocol['scenarios']):
        for history in range(a.chunk*protocol['histories_per_chunk'],(a.chunk+1)*protocol['histories_per_chunk']):
            seed=protocol['base_seed']+1000000*scenario_index+10000*history
            for method in protocol['methods']:
                result=run_history(method,scenario,seed,protocol);data.append(result)
                rows.append({k:v for k,v in result.items() if k!='batches'})
            print(json.dumps(dict(scenario=scenario,history=history,finished_methods=len(protocol['methods']))),flush=True)
    torch.save(data,a.out/'trace.pt')
    write(a.out/'results.json',dict(complete=True,chunk=a.chunk,protocol_sha256=sha(a.protocol),trace_sha256=sha(a.out/'trace.pt'),rows=rows,
        total_analytic_target_calls=len(rows)*2*protocol['total_pairs'],new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='Adaptive couplings are fitted only on earlier batches; every queried value is retained unchanged in normalizer estimates. Conditional Gaussian marginals imply unbiased normalizer sums via tower expectation. Finite ratios/masses need not be unbiased. Constructed-target test only.'))


if __name__=='__main__':main()
