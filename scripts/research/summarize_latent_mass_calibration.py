#!/usr/bin/env python3
"""Audited calibration summary including the simple nonlinear counterexample."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from cfm_mol.latent_mass_coupling import ResidualSurrogate,log_weight,exact_targets
from scripts.research.latent_mass_calibration import draw
from scripts.research.check_binned_mass_coupling import fit,transform
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','audit','binned','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());torch.set_num_threads(2)
    rows=[];sources={};binned_checks=0
    for replica in [0,1,2]:
        audit_path=a.audit/f's{replica}.json';audit=json.loads(audit_path.read_text())
        assert audit['complete'] and audit['protocol_sha256']==sha(a.protocol)
        br_path=a.binned/f's{replica}/results.json';br=json.loads(br_path.read_text());assert br['complete'] and br['protocol_sha256']==sha(a.protocol)
        for scenario in protocol['scenarios']:
            directory=a.run/f's{replica}'/scenario;rp=directory/'results.json';report=json.loads(rp.read_text())
            checked=next(x for x in audit['cases'] if x['scenario']==scenario);bs=next(x for x in br['cases'] if x['scenario']==scenario)
            assert report['complete'] and checked['full_evaluation_replay'] and checked['source_results_sha256']==sha(rp)==bs['source_results_sha256']
            assert sha(directory/'models.pt')==report['models_sha256']==checked['models_sha256']
            assert sha(directory/'calibration.pt')==report['calibration_sha256']==checked['calibration_sha256']
            bp=a.binned/f's{replica}/{scenario}.pt';assert sha(bp)==bs['output_sha256']
            original=torch.load(directory/'calibration.pt',map_location='cpu',weights_only=False)
            model_data=torch.load(directory/'models.pt',map_location='cpu',weights_only=False)
            binned=torch.load(bp,map_location='cpu',weights_only=False);surrogates=[]
            for state in model_data['surrogate_states']:
                m=ResidualSurrogate(protocol['hidden']).double();m.load_state_dict(state);m.eval();m.requires_grad_(False);surrogates.append(m)
            rebuilt=fit(surrogates,model_data,protocol,8)
            for key in rebuilt:torch.testing.assert_close(rebuilt[key],binned['spec'][key],atol=1e-12,rtol=0)
            for n in protocol['calibration_sizes']:
                z=draw(protocol['repetitions']*n,torch.Generator().manual_seed(model_data['seed']+10000+n))
                b=log_weight(transform(z,rebuilt),1,scenario).reshape(-1,n)
                logz1=torch.logsumexp(b,1)-math.log(n);logz0=original[str(n)]['identity']['normalizers'][:,0].log()
                mass=(logz1-logz0).sigmoid();torch.testing.assert_close(mass,binned['mass'][str(n)],atol=1e-10,rtol=0)
                binned_checks+=len(mass)
                rmse=float((mass-exact_targets()['component_mass'][1]).square().mean().sqrt())
                assert abs(rmse-bs['calibration'][str(n)]['binned_mass_RMSE'])<1e-12
                data=report['calibration'][str(n)];methods={k:dict(v) for k,v in data['methods'].items()}
                methods['binned']=dict(mass_RMSE=rmse,mass_bias=float(mass.mean()-exact_targets()['component_mass'][1]),target_calls_per_calibration=2*n)
                neural=methods['nonlinear']['mass_RMSE'];best=min(methods[k]['mass_RMSE'] for k in ['independent','identity','constant','gaussian'])
                rows.append(dict(scenario=scenario,replica=replica,pairs=n,methods=methods,
                    neural_to_best_observed_linear_control_RMSE=neural/best,neural_to_binned_RMSE=neural/rmse,
                    binned_comparison=bs['calibration'][str(n)],comparisons=data['comparisons']))
            sources[f'{scenario}_s{replica}']=dict(results_sha256=sha(rp),audit_sha256=sha(audit_path),binned_results_sha256=sha(br_path),binned_data_sha256=sha(bp))
    if a.out.exists():raise FileExistsError(a.out)
    write(a.out,dict(complete=True,protocol_sha256=sha(a.protocol),sources=sources,rows=rows,exact=exact_targets(),
        primary_cases=6,fit_seed_runs=3,scenarios=protocol['scenarios'],primary_full_replay=True,binned_full_replay=True,binned_mass_values_checked=binned_checks,
        new_molecular_oracle_calls=0,analytic_training_labels=6*2*protocol['labels_per_component'],scientific_submission_ready=False,
        interpretation='The constructed nonlinear case benefits from nonlinear marginal-preserving coupling versus linear/Gaussian choices. Eight radial bins perform similarly to the neural map, so a distinctive neural advantage is not established. The constant-angle case gives no substantive neural advantage. Coarse mass calibration does not correct conditional shapes; joint reverse KL can remain worse than variational mass weighting.',
        limits=['The target was constructed with a known radius-dependent twist; this is not molecular performance.','GenSNIS and Gaussian-preserving flow rearrangements are direct prior art.','The oracle coupling is an asymptotic log-ratio-variance reference, not a finite-sample MSE lower bound.','Normalizer estimates preserve unbiasedness under fixed independent coupling training; normalized masses and log ratios generally have finite-sample bias.','Three separately reported fit seeds are not a new-condition benchmark. Calibration repeats are independent of fitting, not independent model replications.']))
    print(json.dumps([dict(scenario=row['scenario'],replica=row['replica'],rmse={k:v['mass_RMSE'] for k,v in row['methods'].items()}) for row in rows if row['pairs']==64]))


if __name__=='__main__':main()
