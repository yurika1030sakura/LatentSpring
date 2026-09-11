#!/usr/bin/env python3
"""Replay the frozen particle pilot and retain all prescribed outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from cfm_mol.benchmarks.harness import CentredGaussianSource,effective_sample_size
from cfm_mol.benchmarks.particle_systems import reduced_energy
from cfm_mol.entropy_adapter_io import build_species_adapter
from cfm_mol.equivariant_pair_adapter import EquivariantPairAdapter


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    return {'mean':float(x.mean()),'row_sem':float(x.std()/len(x)**.5),'parents':len(x)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    p.add_argument('--capacity-pilot',action='store_true');args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    kinds=['pair','pair_affine'] if args.capacity_pilot else ['pair','pair_affine','whole_convex','convex']
    bound=.75 if args.capacity_pilot else .25
    report={'complete':False,'scope':__doc__,'expected_arms':2*len(kinds),'curvature_bound':bound,'arms':[],'paired':[],
        'additional_analytic_energy_evaluations':0,'molecular_oracle_evaluations':0,'scientific_submission_ready':False,
        'limitations':['Particle diagnostic only, not a pretrained molecular generator.',
            'ESS is a finite-sample diagnostic; no independent target reference is qualified by this audit.',
            'The convex index-split control is not permutation equivariant.',
            'Report each training/evaluation seed; no best-seed selection.']}
    completed={}
    for seed in [0,1]:
        for kind in kinds:
            tag=f'{kind}_s{seed}';path=args.runs/f'{tag}.json';row={'kind':kind,'seed':seed,'state':'unresolved'}
            report['arms'].append(row)
            if not path.exists():continue
            result=json.loads(path.read_text())
            if not result.get('complete'):
                if result.get('failure'):row.update(state='failed',failure=result['failure'])
                row['last_recorded_evaluations']=result.get('target_energy_evaluations');continue
            config=result['configuration']
            expected={'system':'dw4','kind':kind,'seed':seed,'steps':1000,'batch':128,'scale':1.6,
                'sweeps':4,'lr':.001,'eval_samples':8192,'anneal_from_tau':None,'device':'cpu'}
            if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen pilot recipe differs')
            if config.get('pair_curvature_bound',.25)!=bound:raise ValueError('Wrong capacity-pilot parameter')
            if result['target_energy_evaluations']!=160768 or not result['checkpoint_replay_passed']:
                raise ValueError('Pilot budget or saved-model qualification differs')
            for filename,digest in result['artifacts'].items():
                if sha(args.runs/filename)!=digest:raise ValueError('Saved particle artifact changed')
            checkpoint=torch.load(path.with_suffix('.adapter.pt'),map_location='cpu',weights_only=False)
            if checkpoint['kind']!=kind or checkpoint['configuration']!=result['adapter_configuration']:
                raise ValueError('Checkpoint architecture/recipe differs')
            if kind.startswith('pair'):
                model=EquivariantPairAdapter([6]*4,**checkpoint['configuration']).double()
                if model.curvature_bound!=bound:raise ValueError('Saved model curvature differs')
                if model.affine!=(kind=='pair_affine'):raise ValueError('Wrong pair-control nonlinearity')
            else:model=build_species_adapter([6]*4,checkpoint['configuration']).double()
            model.load_state_dict(checkpoint['state_dict'])
            samples=torch.load(path.with_suffix('.samples.pt'),map_location='cpu',weights_only=False)
            source=CentredGaussianSource(4,1.6)
            parent=source.sample(8192,generator=torch.Generator().manual_seed(seed+2000))
            torch.testing.assert_close(parent,samples['parent_positions'],atol=0,rtol=0)
            with torch.no_grad():
                parts=[model(x) for x in parent.split(256)]
                positions=torch.cat([v[0] for v in parts]);volume=torch.cat([v[1] for v in parts])
            torch.testing.assert_close(positions,samples['positions'],atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(volume,samples['log_volume'],atol=1e-9,rtol=1e-9)
            log_q0=source.log_density(parent)
            torch.testing.assert_close(log_q0,samples['log_q0'],atol=0,rtol=0)
            torch.testing.assert_close(log_q0-volume,samples['log_qT'],atol=1e-9,rtol=1e-9)
            before,after=reduced_energy('dw4',parent),reduced_energy('dw4',positions)
            report['additional_analytic_energy_evaluations']+=2*len(parent)
            torch.testing.assert_close(before,samples['base_energy'],atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(after,samples['adapted_energy'],atol=1e-9,rtol=1e-9)
            ess=effective_sample_size(-after-log_q0+volume)
            if abs(ess-result['final']['adapted']['ess'])>1e-10:raise ValueError('Stored ESS does not replay')
            change=after-before-volume
            row.update(state='complete',results_sha256=sha(path),parameters=result['n_parameters'],
                seconds=result['seconds'],target_energy_evaluations=result['target_energy_evaluations'],
                delta_kl=stats(change),ess_fraction=ess,
                base_ess_fraction=result['final']['base']['ess'],
                weight_concentration=result['final']['adapted']['concentration'],
                permutation_equivariant=model.permutation_equivariant,
                source_sha256=result['source_sha256'])
            completed[(seed,kind)]=(change,parent,result)
    for seed in [0,1]:
        for control in ['pair_affine','whole_convex','convex']:
            if (seed,'pair') not in completed or (seed,control) not in completed:continue
            a,x,first=completed[(seed,'pair')];b,y,second=completed[(seed,control)]
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            if control=='pair_affine':
                left={k:v for k,v in first['adapter_configuration'].items() if k!='affine'}
                right={k:v for k,v in second['adapter_configuration'].items() if k!='affine'}
                if left!=right or first['n_parameters']!=second['n_parameters']:
                    raise ValueError('Paired nonlinear/affine architecture differs')
            report['paired'].append({'seed':seed,'contrast':f'pair_minus_{control}','delta_kl':stats(a-b)})
    report['counts']={state:sum(row['state']==state for row in report['arms']) for state in ['complete','failed','unresolved']}
    if args.require_complete and report['counts']['unresolved']:raise ValueError('Prescribed pilot still has unresolved arms')
    report['complete']=True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'counts':report['counts'],'paired':report['paired']}))


if __name__=='__main__':main()
