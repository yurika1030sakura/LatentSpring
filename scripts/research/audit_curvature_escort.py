#!/usr/bin/env python3
"""Independently recover affine-proposal density ratios and pilot contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from cfm_mol.curvature_escort import centered_basis
from cfm_mol.escorted_thermal_teacher import graph_key
from cfm_mol.chemical_moves import infer_chemical_graph
from scripts.research.evaluate_generator_quality import write
from scripts.research.train_electronic_fm import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ['project','run','out']:
        parser.add_argument('--'+k,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);assert not args.out.exists()
    methods=['translation','isotropic','secant'];ess=np.zeros((2,8,4,3));retention=np.zeros_like(ess)
    artifacts=[];queries=0;max_error=0.;checked=0
    for seed in [0,1]:
        protocol=args.project/f'research/evidence/curvature_escort_s{seed}_v1.json';spec=json.loads(protocol.read_text())
        folder=args.run/f's{seed}';resultfile=folder/'results.json';result=json.loads(resultfile.read_text())
        assert result['complete'] and result['protocol_sha256']==sha(protocol)
        assert result['new_raw_queries']==spec['maximum_new_esen_queries']
        assert len(result['rows'])==96
        lookup={(r['method'],r['condition_index'],r['anchor_index']):r for r in result['rows']}
        for ref in spec['teacher_files']:
            source=args.project/ref['path'];assert sha(source)==ref['sha256']
            old=torch.load(source,weights_only=False,map_location='cpu');c=old['condition'];basis=centered_basis(c['n_atoms'])
            d=basis.shape[1];i=ref['condition_index']
            for aj,j in enumerate(ref['anchor_indices']):
                for mi,m in enumerate(methods):
                    row=lookup[m,i,j];file=folder/row['artifact'];assert sha(file)==row['artifact_sha256']
                    saved=torch.load(file,weights_only=False,map_location='cpu')
                    assert saved['protocol_sha256']==sha(protocol) and saved['source_teacher_sha256']==ref['sha256']
                    anchor=old['record']['anchor'][j];sigma=old['record']['sigma'][j]
                    assert torch.equal(anchor,saved['anchor']) and torch.equal(sigma,saved['sigma'])
                    y=saved['proposal'];n=len(y);assert n==spec['production_particles'][m]
                    key=graph_key(infer_chemical_graph(anchor,c['numbers'],c['charge']))
                    flags=[]
                    for x in y:
                        try:flags.append(graph_key(infer_chemical_graph(x,c['numbers'],c['charge']))==key)
                        except (ValueError,RuntimeError,IndexError):flags.append(False)
                    valid=torch.tensor(flags);assert torch.equal(valid,saved['valid'])
                    py=(y-anchor).reshape(n,-1)@basis
                    sx=(saved['source']-anchor).reshape(n,-1)@basis
                    torch.testing.assert_close(py,sx@saved['matrix'].T+saved['shift'],atol=1e-12,rtol=1e-12)
                    covariance=sigma**2*saved['matrix']@saved['matrix'].T
                    proposal=torch.distributions.MultivariateNormal(saved['shift'],covariance_matrix=covariance)
                    reference=torch.distributions.MultivariateNormal(torch.zeros(d,dtype=basis.dtype),covariance_matrix=sigma**2*torch.eye(d,dtype=basis.dtype))
                    energy=(saved['raw_energy_eV'][:n]+saved['raw_energy_eV'][n:])/2
                    assert torch.equal(energy,saved['even_energy_eV'])
                    expected=reference.log_prob(py)-(energy-saved['anchor_energy_eV'])/spec['kT']-proposal.log_prob(py)
                    if valid.any():
                        error=float((expected[valid]-saved['logw'][valid]).abs().max());max_error=max(max_error,error)
                        torch.testing.assert_close(expected[valid],saved['logw'][valid],atol=1e-10,rtol=1e-10)
                        weights=expected[valid].softmax(0);value=float(1/weights.square().sum())
                    else:value=0.
                    assert np.isclose(value,row['ess'],atol=1e-9,rtol=1e-9)
                    assert row['same_graph']==int(valid.sum())
                    ess[seed,i,aj,mi]=value;retention[seed,i,aj,mi]=float(valid.double().mean())
                    assert row['pilot_raw_queries']+row['production_raw_queries']<=48
                    checked+=n
        queries+=result['new_raw_queries'];artifacts.append(dict(seed=seed,protocol_sha256=sha(protocol),results_sha256=sha(resultfile)))
    rng=np.random.default_rng(41491);comparisons={}
    for a,b in [(2,0),(2,1),(1,0)]:
        diff=ess[...,a]-ess[...,b]
        flat=diff.ravel();boot=flat[rng.integers(len(flat),size=(10000,len(flat)))].mean(-1)
        cells=diff.mean((0,2));cboot=cells[rng.integers(len(cells),size=(10000,len(cells)))].mean(-1)
        comparisons[methods[a]+' minus '+methods[b]]=dict(mean_ess_difference=float(diff.mean()),
            paired_anchor95=np.quantile(boot,[.025,.975]).tolist(),composition95=np.quantile(cboot,[.025,.975]).tolist(),
            seed_mean_differences=diff.mean((1,2)).tolist(),positive_anchors=int((diff>0).sum()),anchors=diff.size,
            graph_retention_difference=float((retention[...,a]-retention[...,b]).mean()))
    primary=comparisons['secant minus translation'];gate=bool(min(primary['seed_mean_differences'])>0 and primary['paired_anchor95'][0]>0 and primary['graph_retention_difference']>=-.02)
    arrayfile=args.out.with_suffix('.npz');assert not arrayfile.exists();np.savez_compressed(arrayfile,ess=ess,retention=retention)
    report=dict(complete=True,methods=methods,mean_ess={m:float(ess[...,i].mean()) for i,m in enumerate(methods)},
        by_seed={s:{m:float(ess[s,...,i].mean()) for i,m in enumerate(methods)} for s in [0,1]},
        mean_graph_retention={m:float(retention[...,i].mean()) for i,m in enumerate(methods)},
        comparisons=comparisons,primary_gate=gate,new_raw_esen_queries=queries,proposal_graph_checks_replayed=checked,
        maximum_independent_log_density_error=max_error,artifacts=artifacts,arrays_sha256=sha(arrayfile),
        new_neural_outputs=0,new_training_steps=0,
        scope='Equal maximum query-budget FIT-only molecular escort pilot. Full work correction verified against independent multivariate Gaussian densities. Local finite-sample ESS improvement does not establish neural-output or target-distribution accuracy.')
    write(args.out,report);print(json.dumps({k:report[k] for k in ['mean_ess','by_seed','mean_graph_retention','comparisons','primary_gate']},indent=2))


if __name__=='__main__':main()
