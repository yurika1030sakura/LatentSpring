"""Compare inference-only bundles against saved parent and hydrogen outputs."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from cfm_mol.latentspring_generator import LatentSpringGenerator

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','bundle','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();assert not a.out.exists();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=LatentSpringGenerator(a.bundle,device=a.device);rows=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name in (['fm','gaga'] if a.device=='cuda' else ['fm']):
        f=a.project/f'runs/hydrogen_physical_confirmation_v1/s0/parents/{name}/{name}_a0_c14.pt';g=a.project/f'runs/hydrogen_physical_confirmation_v1/s0/readouts/raw/{name}_molecule_start0_c14.pt'
        parent=torch.load(f,map_location='cpu',weights_only=False);expected=torch.load(g,map_location='cpu',weights_only=False)
        result=model.generate(parent['condition']['numbers'],samples=16,seed=63051,stream=14,family=name)
        torch.testing.assert_close(result['initial_positions'],parent['initial_positions'],atol=0,rtol=0)
        torch.testing.assert_close(result['parent_positions'],parent['positions'],atol=3e-5,rtol=1e-5);torch.testing.assert_close(result['positions'],expected['positions'],atol=3e-5,rtol=1e-5)
        torch.testing.assert_close(result['hydrogen_changed'],expected['decoder_info']['changed'],atol=0,rtol=0)
        assert result['costs']['attempted']==result['costs']['returned']==16 and result['costs']['oracle_queries']==0
        rows.append(dict(family=name,max_coordinate_difference_A=float((result['positions']-expected['positions']).abs().max()),changed=int(result['hydrogen_changed'].sum()),costs=result['costs'],reference_files={str(p.relative_to(a.project)):sha(p) for p in [f,g]}))
    assert 'cfm_mol.energy_oracle' not in sys.modules and 'scripts.research.run_matched_connection' not in sys.modules
    receipt=dict(complete=True,device=a.device,rows=rows,bundle_manifest_sha256=sha(a.bundle/'manifest.json'),inference_module_sha256=sha(a.project/'cfm_mol/latentspring_generator.py'),new_independent_scientific_attempts=0,training_or_oracle_imported=False)
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
