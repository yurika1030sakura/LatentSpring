#!/usr/bin/env python3
"""Compare local curvature escorts at equal maximum physical-query budgets."""
import argparse
import json
from pathlib import Path
import time
import torch

from cfm_mol.curvature_escort import (centered_basis, fit_curvature, affine_parameters,
    affine_candidates, complete_log_weights)
from cfm_mol.escorted_thermal_teacher import graph_key
from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.evaluate_generator_quality import write
from scripts.research.train_electronic_fm import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    assert spec['frozen'] and not args.out.exists()
    worker=args.project/spec['oracle_worker']
    assert sha(worker)==spec['oracle_worker_sha256'] and sha(spec['oracle_checkpoint'])==spec['oracle_sha256']
    args.out.mkdir(parents=True)
    start=time.perf_counter();rows=[];charged_pilot=0;charged_anchors=0
    first=torch.load(args.project/spec['teacher_files'][0]['path'],weights_only=False,map_location='cpu')['condition']
    with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=first['numbers'],
            charge=first['charge'],spin_multiplicity=first['spin_multiplicity'],device='cuda',
            batch_size=8,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        for ref in spec['teacher_files']:
            file=args.project/ref['path'];assert sha(file)==ref['sha256']
            saved=torch.load(file,weights_only=False,map_location='cpu');record=saved['record'];c=saved['condition']
            assert c['composition_hex']==ref['composition_hex']
            oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
            basis=centered_basis(c['n_atoms']);identity=torch.eye(basis.shape[1],dtype=basis.dtype)
            all_valid=record['valid'];count=int(all_valid.sum())
            pilot_force=torch.zeros_like(record['proposal'])
            pilot_force[all_valid]=(record['proposal_raw_force_eV_A'][:count]-record['proposal_raw_force_eV_A'][count:])/2
            for j in ref['anchor_indices']:
                anchor=record['anchor'][j];force=record['anchor_force_eV_A'][j]
                ea=record['anchor_energy_eV'][j];sigma=record['sigma'][j];valid=all_valid[j]
                hessian,kappa=fit_curvature(record['proposal'][j][valid]-anchor,
                    force[None]-pilot_force[j][valid],basis,ridge_fraction=spec['ridge_fraction'])
                charged_pilot+=2*int(valid.sum());charged_anchors+=2
                key=graph_key(infer_chemical_graph(anchor,c['numbers'],c['charge']))
                rng=torch.Generator().manual_seed(spec['production_seed']*1000003+ref['condition_index']*100003+j)
                noise=torch.randn((24,c['n_atoms'],3),dtype=torch.float64,generator=rng)
                noise-=noise.mean(1,keepdim=True)
                for method in spec['methods']:
                    h=identity*0 if method=='translation' else (identity*kappa if method=='isotropic' else hessian)
                    matrix,shift,logdet=affine_parameters(force,sigma,spec['kT'],h,basis)
                    n=spec['production_particles'][method]
                    source,proposal=affine_candidates(anchor,noise[:n],sigma,matrix,shift,basis)
                    valid_new=[]
                    for y in proposal:
                        try:
                            valid_new.append(graph_key(infer_chemical_graph(y,c['numbers'],c['charge']))==key)
                        except (ValueError,RuntimeError,IndexError):
                            valid_new.append(False)
                    valid_new=torch.tensor(valid_new,dtype=torch.bool)
                    energy,forces=oracle.evaluate_chunked(torch.cat([proposal,-proposal]),max_request=16)
                    even=(energy[:n]+energy[n:])/2
                    logw=complete_log_weights(anchor,source,proposal,ea,even,sigma,spec['kT'],logdet,valid_new)
                    weights=logw.softmax(0) if valid_new.any() else torch.zeros_like(logw)
                    ess=float(1/weights.square().sum()) if valid_new.any() else 0.
                    name=f'{method}_c{ref["condition_index"]}_a{j}.pt';target=args.out/name
                    torch.save(dict(anchor=anchor,source=source,proposal=proposal,sigma=sigma,force=force,
                        anchor_energy_eV=ea,raw_energy_eV=energy,raw_force_eV_A=forces,even_energy_eV=even,
                        matrix=matrix,shift=shift,logdet=logdet,hessian=h,kappa=kappa,valid=valid_new,
                        logw=logw,weights=weights,source_teacher_sha256=ref['sha256'],protocol_sha256=ph),target)
                    rows.append(dict(method=method,condition_index=ref['condition_index'],anchor_index=j,
                        artifact=name,artifact_sha256=sha(target),production_particles=n,
                        same_graph=int(valid_new.sum()),ess=ess,log_normalizer=float(torch.logsumexp(logw,0)-torch.log(logw.new_tensor(n))) if valid_new.any() else None,
                        pilot_raw_queries=0 if method=='translation' else 2*int(valid.sum()),
                        production_raw_queries=2*n,anchor_raw_queries=2))
            write(args.out/'results.json',dict(complete=False,protocol_sha256=ph,rows=rows,
                new_raw_queries=oracle.evaluated,reused_pilot_raw_queries=charged_pilot,reused_anchor_raw_queries=charged_anchors))
            print(json.dumps(dict(condition=ref['condition_index'],new_raw_queries=oracle.evaluated,
                mean_ess={m:sum(r['ess'] for r in rows if r['method']==m)/sum(r['method']==m for r in rows) for m in spec['methods']})),flush=True)
        assert oracle.evaluated==oracle.requested_evaluations==spec['maximum_new_esen_queries']
        queries=oracle.evaluated
    write(args.out/'results.json',dict(complete=True,protocol_sha256=ph,rows=rows,new_raw_queries=queries,
        reused_pilot_raw_queries=charged_pilot,reused_anchor_raw_queries=charged_anchors,
        seconds=time.perf_counter()-start,new_neural_outputs=0,new_training_steps=0))


if __name__=='__main__':
    main()
