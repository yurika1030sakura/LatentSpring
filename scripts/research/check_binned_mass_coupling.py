#!/usr/bin/env python3
"""Strong low-cost nonlinear control, using the same frozen surrogate labels."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.special import gammaincinv
from cfm_mol.latent_mass_coupling import ResidualSurrogate,rotate,log_weight,exact_targets
from scripts.research.latent_mass_calibration import draw,score
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def fit(surrogates,saved,protocol,bins):
    bank=draw(protocol['coupling_bank'],torch.Generator().manual_seed(saved['seed']+301))
    boundaries=torch.tensor(2*gammaincinv(3.,np.arange(1,bins)/bins),dtype=torch.float64)
    ids=torch.bucketize(bank.square().sum((-2,-1)),boundaries);normalizers=bank.new_tensor(saved['fit']['normalizers'])
    angles=[];reflections=[]
    with torch.no_grad():
        for i in range(bins):
            z=bank[ids==i];assert len(z)>0;best=None
            for reflection in [False,True]:
                for j in range(protocol['angle_grid']):
                    angle=-math.pi/2+(j+.5)*math.pi/protocol['angle_grid']
                    value=float(score(surrogates,z,rotate(z,z.new_tensor(angle),reflection),normalizers))
                    if best is None or value>best[0]:best=(value,angle,reflection)
            angles.append(best[1]);reflections.append(best[2])
    return dict(boundaries=boundaries,angles=torch.tensor(angles,dtype=torch.float64),reflection=torch.tensor(reflections,dtype=torch.bool))


def transform(z,spec,inverse=False):
    ids=torch.bucketize(z.square().sum((-2,-1)),spec['boundaries'])
    angle=spec['angles'][ids];flip=torch.where(spec['reflection'][ids],-1.,1.).to(z)
    if inverse:
        x=rotate(z,-angle);return torch.stack([x[...,0,:],flip[...,None]*x[...,1,:]],-2)
    x=torch.stack([z[...,0,:],flip[...,None]*z[...,1,:]],-2)
    return rotate(x,angle)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replica',type=int,required=True);a=p.parse_args();protocol=json.loads(a.protocol.read_text());torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);results=[];truth=exact_targets()
    for scenario in protocol['scenarios']:
        directory=a.run/scenario;header=json.loads((directory/'results.json').read_text())
        assert header['complete'] and header['protocol_sha256']==sha(a.protocol) and sha(directory/'models.pt')==header['models_sha256']
        assert sha(directory/'calibration.pt')==header['calibration_sha256']
        saved=torch.load(directory/'models.pt',map_location='cpu',weights_only=False);original=torch.load(directory/'calibration.pt',map_location='cpu',weights_only=False)
        surrogates=[]
        for state in saved['surrogate_states']:
            model=ResidualSurrogate(protocol['hidden']).double();model.load_state_dict(state);model.eval();model.requires_grad_(False);surrogates.append(model)
        spec=fit(surrogates,saved,protocol,8)
        check=draw(128,torch.Generator().manual_seed(saved['seed']+30001))
        torch.testing.assert_close(transform(transform(check,spec),spec,True),check,atol=1e-11,rtol=0)
        torch.testing.assert_close(transform(check,spec).square().sum((-2,-1)),check.square().sum((-2,-1)),atol=1e-11,rtol=0)
        rows={};arrays={};calls=0
        for n in protocol['calibration_sizes']:
            z=draw(protocol['repetitions']*n,torch.Generator().manual_seed(saved['seed']+10000+n))
            b=log_weight(transform(z,spec),1,scenario).reshape(-1,n);calls+=len(z)
            logz1=torch.logsumexp(b,1)-math.log(n)
            logz0=original[str(n)]['identity']['normalizers'][:,0].log()
            mass=(logz1-logz0).sigmoid();err=mass-truth['component_mass'][1]
            neural=original[str(n)]['nonlinear']['component_mass']-truth['component_mass'][1]
            delta=neural.square()-err.square();se=float(delta.std(unbiased=True)/math.sqrt(len(delta)))
            rows[str(n)]=dict(binned_mass_RMSE=float(err.square().mean().sqrt()),binned_mass_bias=float(err.mean()),
                nonlinear_mass_RMSE=float(neural.square().mean().sqrt()),neural_to_binned_RMSE_ratio=float(neural.square().mean().sqrt()/err.square().mean().sqrt()),
                paired_squared_error_difference=float(delta.mean()),paired_MC_normal95=[float(delta.mean())-1.96*se,float(delta.mean())+1.96*se])
            arrays[str(n)]=mass
        torch.save(dict(spec=spec,mass=arrays),a.out/f'{scenario}.pt')
        results.append(dict(scenario=scenario,source_results_sha256=sha(directory/'results.json'),output_sha256=sha(a.out/f'{scenario}.pt'),calibration=rows,new_analytic_target_calls=calls))
        print(json.dumps(dict(scenario=scenario,replica=a.replica,at64=rows['64'])),flush=True)
    write(a.out/'results.json',dict(complete=True,replica=a.replica,protocol_sha256=sha(a.protocol),cases=results,new_molecular_oracle_calls=0,
        scope='Supplementary8-bin nonlinear control added after the first toy result. Fits angles/reflections from the same frozen surrogates, without target queries or calibration outcomes. Same held evaluation latents. Discontinuities at radial bin boundaries have Gaussian measure zero; the map preserves Gaussian measure conditionally on radius.',scientific_submission_ready=False))


if __name__=='__main__':main()
