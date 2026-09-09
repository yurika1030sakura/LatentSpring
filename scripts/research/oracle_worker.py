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
    args=p.parse_args()
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    import ase
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    calculator=FAIRChemCalculator(load_predict_unit(str(args.checkpoint),device=args.device))
    emit({'ready':True})
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
            for x in positions:
                atoms=ase.Atoms(numbers=numbers,positions=x)
                atoms.info.update(charge=charge,spin=spin);atoms.calc=calculator
                attempted+=1
                energy=float(atoms.get_potential_energy());force=atoms.get_forces()
                if not np.isfinite(energy) or not np.isfinite(force).all():
                    raise FloatingPointError('Non-finite oracle energy/force')
                energies.append(energy);forces.append(force.tolist())
            emit({'ok':True,'energies_eV':energies,'forces_eV_A':forces,'attempted_evaluations':attempted})
        except Exception as exc:
            emit({'ok':False,'error':f'{type(exc).__name__}: {str(exc)[:1000]}','attempted_evaluations':attempted})


if __name__=='__main__':main()
