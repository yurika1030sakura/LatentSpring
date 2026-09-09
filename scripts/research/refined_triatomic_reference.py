#!/usr/bin/env python3
"""Independent defensive shape-QMC reference for the fixed 300/1000-K targets.

Pilot centers come from frozen independent Gaussian quadrature, not learned
generator outputs. The new samples are independent of that pilot. Equal-count
Gaussian/shape strata match the explicitly evaluated 50/50 mixture density.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from scipy.special import logsumexp
from ase.data import atomic_numbers
from rdkit import Chem

from cfm_mol.electronic_metadata import ElectronicMetadata
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.rotation_mixture import random_rotations
from cfm_mol.triatomic_reference import gaussian_triatomic_shapes,invariant_observables
from cfm_mol.triatomic_shape_proposal import to_shape_coordinates,TriatomicShapeMixture,defensive_shape_log_density
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--power',type=int,default=11)
    p.add_argument('--scrambles',type=int,default=4);p.add_argument('--batch',type=int,default=64)
    args=p.parse_args()
    if args.power<5 or args.scrambles<2 or args.batch<1:raise ValueError('Invalid reference counts')
    output=args.out/'reference.json'
    if output.exists():raise FileExistsError(output)
    source=args.pilot/'reference.json';old=json.loads(source.read_text())
    if not old['complete'] or old['oracle_sha256']!=sha(args.oracle):raise ValueError('Pilot/checkpoint identity mismatch')
    numbers=[atomic_numbers[s] for s in old['symbols']]
    if len(numbers)!=3 or numbers[1]!=numbers[2]:raise ValueError('Protocol requires identical outer atoms')
    metadata=ElectronicMetadata(args.metadata,'val',list(range(1,84)))
    metadata.verify_processed_file(Path(old['configuration']['processed']))
    accepted=int(metadata.indices[old['source_row']]);charge=int(metadata.values['total_charge'][accepted]);spin=int(metadata.values['spin_multiplicity'][accepted])
    if (charge,spin)!=(old['charge'],old['spin_multiplicity']):raise ValueError('Pilot electronic state differs from original metadata')
    arrays=[];energies=[];pilot_hashes={}
    for index in range(len(old['rows'])):
        path=args.pilot/f'scramble_{index}.npz';data=np.load(path)
        arrays.append(data['positions'].reshape(-1,3,3));energies.append(data['energy_eV'].reshape(-1));pilot_hashes[str(path.resolve())]=sha(path)
    x=np.concatenate(arrays);e=np.concatenate(energies);restraint=float(old['configuration']['restraint'])
    reduced_energy=e+restraint/2*(x*x).sum((1,2))
    selected=np.argsort(reduced_energy,kind='stable')[:64]
    centers=to_shape_coordinates(x[selected]);centers=np.concatenate([centers,centers[:,[1,0,2]]])
    local=TriatomicShapeMixture(centers,(.05,.05,.5));defensive_std=math.sqrt(1./restraint)
    temperatures=[.025851999786435,.08617333262145]
    root=Path(__file__).resolve().parents[2];args.out.mkdir(parents=True,exist_ok=True)
    report={'complete':False,'scope':__doc__,'condition':{'numbers':numbers,'charge':charge,'spin_multiplicity':spin,'source_row':old['source_row']},
        'configuration':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'pilot_source_sha256':sha(source),'pilot_array_sha256':pilot_hashes,'pilot_oracle_queries':old['oracle_evaluations'],
        'oracle_sha256':sha(args.oracle),'metadata_progress_sha256':metadata.progress_sha256,
        'script_sha256':sha(Path(__file__)),'proposal_code_sha256':sha(root/'cfm_mol/triatomic_shape_proposal.py'),
        'selection':'64 lowest restrained pilot energies, then both outer-atom orders; fixed equal component weights',
        'pilot_selected_indices':selected.tolist(),'shape_centers':centers.tolist(),'shape_stds':local.stds.tolist(),
        'defensive_std_A':defensive_std,'defensive_weight':.5,'restraint_eV_A2':restraint,
        'contact_radii_source':'RDKit GetRcovalent','contact_factor':1.25,
        'kT_eV':temperatures,'rows':[],'convergence_certified':False,
        'limitations':['Four-scramble uncertainty cannot exclude commonly missed modes.',
            'Pilot construction cost is separate and explicitly retained.',
            'Both temperatures reuse the same new points and have correlated estimates.',
            'Contact graphs use distance heuristics, not chemical bond labels.']}
    write_json(output,report);start=time.perf_counter();worker=root/'scripts/research/oracle_worker.py'
    with EnergyOracle(args.oracle_python,worker,args.oracle,numbers=numbers,charge=charge,spin_multiplicity=spin,batch_size=16) as oracle:
        check=np.concatenate([gaussian_triatomic_shapes(2,defensive_std,19081),local.sample(2,19082)])
        batched,forces=oracle.evaluate(check)
        with EnergyOracle(args.oracle_python,worker,args.oracle,numbers=numbers,charge=charge,spin_multiplicity=spin,batch_size=1) as serial:
            serial_energy,serial_force=serial.evaluate(check);serial_queries=serial.evaluated
        rotations=random_rotations(len(check),torch.Generator().manual_seed(19083)).numpy();rotated,_=oracle.evaluate(check@rotations)
        validation={'max_serial_batch_energy_error_eV':float((batched-serial_energy).abs().max()),
            'max_serial_batch_force_error_eV_A':float((forces-serial_force).abs().max()),
            'max_rotation_energy_error_eV':float((batched-rotated).abs().max())}
        report['oracle_validation']=validation;write_json(output,report)
        if validation['max_serial_batch_energy_error_eV']>2e-4 or validation['max_serial_batch_force_error_eV_A']>2e-3 or validation['max_rotation_energy_error_eV']>1e-3:
            raise ValueError('Oracle consistency gate failed')
        i,j=np.triu_indices(3,k=1);table=Chem.GetPeriodicTable();radii=np.array([table.GetRcovalent(z) for z in numbers])
        for repeat in range(args.scrambles):
            sets=[gaussian_triatomic_shapes(args.power,defensive_std,29081+1009*repeat),local.sample(args.power,39081+1009*repeat)]
            values=[]
            for positions in sets:
                pieces=[]
                for begin in range(0,len(positions),args.batch):
                    energy,_=oracle.evaluate(positions[begin:begin+args.batch]);pieces.append(energy.numpy())
                values.append(np.concatenate(pieces))
            np.savez_compressed(args.out/f'scramble_{repeat}.npz',positions=np.stack(sets),energy_eV=np.stack(values))
            row={'scramble':repeat,'targets':[]}
            for kT in temperatures:
                resolutions=[]
                for power in range(args.power-3,args.power+1):
                    n=2**power;x=np.concatenate([v[:n] for v in sets]);e=np.concatenate([v[:n] for v in values])
                    logw=-(e+restraint/2*(x*x).sum((1,2)))/kT-defensive_shape_log_density(x,local,defensive_std)
                    weights=np.exp(logw-logw.max());weights/=weights.sum()
                    ratios=np.linalg.norm(x[:,i]-x[:,j],axis=-1)/(radii[i]+radii[j])
                    components=3-np.minimum((ratios<=1.25).sum(-1),2)
                    moments={**invariant_observables(x),'multiple_contact_components_probability':(components>1).astype(float)}
                    resolutions.append({'particles':len(x),'log_normalizer_estimate':float(logsumexp(logw)-math.log(len(x))),
                        'weight_ess':float(1/(weights@weights)),'maximum_weight':float(weights.max()),
                        'moments':{name:float(weights@value) for name,value in moments.items()}})
                row['targets'].append({'kT_eV':kT,'resolutions':resolutions})
            report['rows'].append(row);report['new_oracle_queries']=oracle.evaluated+serial_queries
            write_json(output,report);print(json.dumps({'scramble':repeat,'queries':report['new_oracle_queries'],'seconds':time.perf_counter()-start}),flush=True)
        aggregate=[]
        for target_index,kT in enumerate(temperatures):
            resolutions=[r['targets'][target_index]['resolutions'] for r in report['rows']]
            for level in range(4):
                selected=[r[level] for r in resolutions];logs=np.array([r['log_normalizer_estimate'] for r in selected]);z=np.exp(logs-logs.max())
                moments={}
                for name in selected[0]['moments']:
                    values=np.array([r['moments'][name] for r in selected]);mean=float(z@values/z.sum());influence=z*(values-mean)/z.mean()
                    moments[name]={'mean':mean,'delta_method_se':float(influence.std(ddof=1)/math.sqrt(len(z)))}
                aggregate.append({'kT_eV':kT,'particles_per_scramble':selected[0]['particles'],
                    'log_of_mean_normalizer_estimate':float(logsumexp(logs)-math.log(len(logs))),
                    'relative_se_across_scrambles':float(z.std(ddof=1)/math.sqrt(len(z))/z.mean()),
                    'minimum_scramble_ess':min(r['weight_ess'] for r in selected),'log_normalizer_range':float(np.ptp(logs)),
                    'moments':moments})
        report.update(complete=True,aggregate=aggregate,seconds=time.perf_counter()-start)
        write_json(output,report);print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
