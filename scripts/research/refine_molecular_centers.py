#!/usr/bin/env python3
"""Refine an independent FM pilot with the declared target, retaining all centers."""
import argparse,json,time
from pathlib import Path
import torch
from ase.data import atomic_numbers
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.proposal_refinement import refine_centers
from cfm_mol.tempered_smc import DensityValue
from molecular_tempered_pilot import sha,write_json,geometry_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--source-row',type=int,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--oracle-python',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--kT',type=float,default=1.);p.add_argument('--restraint',type=float,default=.1)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise FileExistsError(args.out/'results.json')
    source=args.pilot/'results.json';pilot=json.loads(source.read_text())
    if not pilot['complete']:raise ValueError('Incomplete original pilot')
    condition=dict(next(c for c in pilot['conditions'] if c['source_row']==args.source_row))
    original_file=Path(condition['center_file'])
    if sha(original_file)!=condition['center_sha256']:raise ValueError('Original centers changed')
    original=torch.load(str(original_file),map_location='cpu',weights_only=False)
    z=original['centers'].double();numbers=[atomic_numbers[s] for s in condition['symbols']];n=len(numbers)
    basis=centered_orthonormal_basis(n);root=Path(__file__).resolve().parents[2]
    start=time.perf_counter()
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,
            numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity']) as oracle:
        def target(latent):
            x=torch.einsum('nk,bkd->bnd',basis,latent.reshape(len(latent),n-1,3))
            energy,force=oracle.evaluate(x)
            score=(torch.einsum('nk,bnd->bkd',basis,force).reshape_as(latent)-args.restraint*latent)/args.kT
            return DensityValue(-(energy+args.restraint/2*latent.square().sum(-1))/args.kT,score)
        refined,diagnostics=refine_centers(z,target,args.steps)
        assert diagnostics['target_evaluations']==oracle.evaluated
    positions=torch.einsum('nk,bkd->bnd',basis,refined.reshape(len(z),n-1,3))
    output=args.out/f'centers_{args.source_row}.pt'
    torch.save({'positions':positions,'centers':refined,'condition':condition},output)
    condition.update(center_file=str(output.resolve()),center_sha256=sha(output),
        original_center_file=str(original_file),refinement_oracle_calls=diagnostics['target_evaluations'],
        refinement_seconds=time.perf_counter()-start)
    report={'complete':True,'scope':__doc__,'conditions':[condition],'source_pilot':str(source.resolve()),
        'source_pilot_sha256':sha(source),'oracle_sha256':sha(args.oracle),
        'source_sha256':{str(f):sha(f) for f in [Path(__file__).resolve(),root/'cfm_mol/proposal_refinement.py']},
        'target':{'kT':args.kT,'restraint':args.restraint},'diagnostics':diagnostics,
        'original_geometry':geometry_metrics(original['positions'].double(),torch.full((len(z),),1/len(z),dtype=torch.float64),numbers),
        'refined_geometry':geometry_metrics(positions,torch.full((len(z),),1/len(z),dtype=torch.float64),numbers)}
    write_json(args.out/'results.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
