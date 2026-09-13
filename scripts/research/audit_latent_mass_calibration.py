#!/usr/bin/env python3
"""Replay the constructed calibration test and check trained marginal preservation."""
import argparse,json,math
from pathlib import Path
import torch
from cfm_mol.latent_mass_coupling import ResidualSurrogate,ShapeTwist,GaussianCoupling,reference_flow,toy_energy,log_weight
from scripts.research.latent_mass_calibration import evaluate,draw
from scripts.research.audit_masked_angular import sha,equal
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replica',type=int,required=True);a=p.parse_args();protocol=json.loads(a.protocol.read_text())
    torch.set_num_threads(2);report=[]
    assert json.loads((a.run/'results.json').read_text())['complete']
    for scenario_index,scenario in enumerate(protocol['scenarios']):
        directory=a.run/scenario;header=json.loads((directory/'results.json').read_text())
        assert header['complete'] and header['protocol_sha256']==sha(a.protocol)
        assert sha(directory/'models.pt')==header['models_sha256'] and sha(directory/'calibration.pt')==header['calibration_sha256']
        saved=torch.load(directory/'models.pt',map_location='cpu',weights_only=False)
        assert saved['scenario']==scenario and saved['seed']==protocol['seeds'][a.replica]+100000*scenario_index
        surrogates=[]
        for component in [0,1]:
            model=ResidualSurrogate(protocol['hidden']).double();model.load_state_dict(saved['surrogate_states'][component]);model.eval();model.requires_grad_(False);surrogates.append(model)
            expected=draw(protocol['labels_per_component'],torch.Generator().manual_seed(saved['seed']+101+component))
            row=saved['data'][component];torch.testing.assert_close(expected,row['z'],atol=0,rtol=0)
            x,logq=reference_flow(expected,component);torch.testing.assert_close(x,row['x'],atol=1e-12,rtol=0);torch.testing.assert_close(logq,row['logq'],atol=1e-12,rtol=0)
            energy=toy_energy(x,component,scenario);torch.testing.assert_close(energy,row['energy'],atol=1e-12,rtol=0)
            torch.testing.assert_close(-energy-logq,row['labels'],atol=1e-12,rtol=0)
        couplings={}
        for kind,cls in [('gaussian',GaussianCoupling),('nonlinear',ShapeTwist)]:
            spec=saved['couplings'][kind];model=cls(**spec['configuration']).double();model.load_state_dict(spec['state_dict']);model.eval();model.requires_grad_(False);couplings[kind]=model
        check=draw(128,torch.Generator().manual_seed(saved['seed']+7001));twist=couplings['nonlinear']
        torch.testing.assert_close(twist(check).square().sum((-2,-1)),check.square().sum((-2,-1)),atol=1e-11,rtol=0)
        torch.testing.assert_close(twist.inverse(twist(check)),check,atol=1e-11,rtol=0)
        for point in check[:4]:
            jac=torch.autograd.functional.jacobian(lambda z:twist(z.reshape(2,3)).flatten(),point.flatten())
            assert abs(float(torch.linalg.slogdet(jac)[1]))<1e-10
        cross,noise=couplings['gaussian'].matrices()
        torch.testing.assert_close(cross@cross.T+noise@noise.T,torch.eye(2,dtype=torch.float64),atol=1e-12,rtol=0)
        computed,arrays=evaluate(surrogates,couplings,saved['fit'],scenario,saved['seed'],protocol)
        for key,value in computed.items():equal(value,header[key])
        equal(arrays,torch.load(directory/'calibration.pt',map_location='cpu',weights_only=False))
        report.append(dict(scenario=scenario,complete=True,full_evaluation_replay=True,all_training_labels_replayed=True,
            trained_twist_inverse_checks=128,trained_twist_Jacobian_checks=4,gaussian_marginal_covariance_checked=True,
            source_results_sha256=sha(directory/'results.json'),models_sha256=header['models_sha256'],calibration_sha256=header['calibration_sha256'],
            analytic_evaluation_calls_replayed=header['analytic_evaluation_calls']))
        print(json.dumps(report[-1]),flush=True)
    if a.out.exists():raise FileExistsError(a.out)
    write(a.out,dict(complete=True,replica=a.replica,protocol_sha256=sha(a.protocol),cases=report,new_molecular_oracle_calls=0,scientific_submission_ready=False,
        caveat='The oracle map is optimal for fixed-marginal asymptotic log-ratio variance, not a proven finite-budget mass-MSE optimum. Ratios/logs of unbiased normalizer estimates need not be unbiased. The manufactured COM target is not a chemical molecule.'))


if __name__=='__main__':main()
