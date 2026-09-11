#!/usr/bin/env python3
"""JSON-lines eSEN worker; execute only with the separate omol25 interpreter."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


PREFIX='BGFM_ORACLE_JSON '


def emit(payload):
    print(PREFIX+json.dumps(payload,allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=1)
    args=p.parse_args()
    if args.batch_size<1:raise ValueError('Positive batch size required')
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    import ase
    import torch
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    predictor=load_predict_unit(str(args.checkpoint),device=args.device)
    calculator=FAIRChemCalculator(predictor)
    settings=predictor.inference_settings
    emit({'ready':True,'device':args.device,'worker_batch_size':args.batch_size,
        'torch_version':torch.__version__,'base_precision_dtype':str(settings.base_precision_dtype),
        'tf32':bool(settings.tf32),'activation_checkpointing':bool(settings.activation_checkpointing),
        'execution_mode':settings.execution_mode,
        'gpu_name':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None})
    for line in sys.stdin:
        attempted=0
        try:
            request=json.loads(line)
            positions=np.asarray(request['positions'],dtype=np.float64)
            numbers=np.asarray(request['numbers'],dtype=np.int64)
            charge=int(request['charge']);spin=int(request['spin_multiplicity'])
            electrons=int(numbers.sum())-charge
            if electrons<1 or spin<1 or electrons<spin-1 or (electrons-(spin-1))%2:
                raise ValueError('Electronic state incompatible with electron parity')
            if positions.ndim!=3 or positions.shape[1:]!=(len(numbers),3) or len(positions)<1 or not np.isfinite(positions).all():
                raise ValueError('Require finite [batch, atoms, 3] positions')
            energies=[];forces=[]
            for start in range(0,len(positions),args.batch_size):
                structures=[]
                for x in positions[start:start+args.batch_size]:
                    atoms=ase.Atoms(numbers=numbers,positions=x)
                    atoms.info.update(charge=charge,spin=spin)
                    structures.append(atoms)
                if args.batch_size==1:
                    atoms=structures[0];atoms.calc=calculator;attempted+=1
                    energy=np.asarray([atoms.get_potential_energy()]);force=atoms.get_forces()[None]
                else:
                    for atoms in structures:
                        calculator._check_atoms_pbc(atoms)
                        calculator.predictor.validate_atoms_data(atoms,calculator.task_name)
                    batch=atomicdata_list_to_batch([calculator.a2g(atoms) for atoms in structures])
                    attempted+=len(structures)
                    prediction=calculator.predictor.predict(batch)
                    energy=prediction['energy'].detach().cpu().numpy().reshape(-1)
                    force=prediction['forces'].detach().cpu().numpy().reshape(len(structures),len(numbers),3)
                if energy.shape!=(len(structures),) or not np.isfinite(energy).all() or not np.isfinite(force).all():
                    raise FloatingPointError('Non-finite oracle energy/force')
                energies.extend(energy.tolist());forces.extend(force.tolist())
            emit({'ok':True,'energies_eV':energies,'forces_eV_A':forces,'attempted_evaluations':attempted})
        except Exception as exc:
            emit({'ok':False,'error':f'{type(exc).__name__}: {str(exc)[:1000]}','attempted_evaluations':attempted})


if __name__=='__main__':main()
