"""Check relocated release weights against archived generation trajectories."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cfm_mol.release import Generator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    r=a.release.resolve();generator=Generator(r/'weights',device=a.device)
    # Every exported tensor set is instantiated, including optional readouts.
    restored={}
    for name in generator.models:
        item=generator.load(name);recipe=item['recipe']
        if recipe.get('hydrogen'):generator._module(recipe['hydrogen'],'backbone')
        restored[name]=dict(parent=recipe['parent'],geometry=recipe.get('geometry'),physical=recipe.get('physical'))
    fixtures=json.loads((r/'examples/sampling_fixtures.json').read_text());arrays=dict(np.load(r/'examples/sampling_fixtures.npz'))
    rows=[]
    for index,f in enumerate(fixtures):
        if f['model'] not in generator.models:continue
        if a.device=='cpu' and index:continue
        generated=generator.generate(f['numbers'],model=f['model'],samples=16,seed=f['seed'],stream=f['stream'],hydrogen=f.get('hydrogen',False))
        expected=arrays[f'{index}_positions'];start=arrays[f'{index}_initial_positions']
        np.testing.assert_allclose(generated['initial_positions'].numpy(),start,atol=0,rtol=0)
        delta=generated['positions'].numpy()-expected
        maximum=float(abs(delta).max());rms=float(np.sqrt(np.mean(delta**2)))
        # CUDA scatter reductions in archived runs were nondeterministic.
        assert maximum<=1e-3 and rms<=1e-4,(f['model'],maximum,rms)
        assert generated['costs']['attempted']==generated['costs']['returned']==16
        assert generated['costs']['energy_queries']==0 and generated['costs']['geometry_optimizer_steps']==0
        rows.append(dict(model=f['model'],maximum_coordinate_difference_A=maximum,
            rms_coordinate_difference_A=rms,costs=generated['costs']))
        print(json.dumps(rows[-1]),flush=True)
    assert 'cfm_mol.energy_oracle' not in sys.modules
    assert 'scripts.research.run_matched_connection' not in sys.modules
    assert 'dgl' not in sys.modules and 'flowmol' not in sys.modules
    result=dict(complete=True,device=a.device,model_recipes_loaded=len(restored),
        distinct_weight_modules_loaded=len(generator.modules),rows=rows,
        checked_trajectories=16*len(rows),new_independent_scientific_attempts=0,
        weights_manifest_sha256=hashlib.sha256((r/'weights/manifest.json').read_bytes()).hexdigest(),
        training_or_energy_dependencies_imported=False)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
