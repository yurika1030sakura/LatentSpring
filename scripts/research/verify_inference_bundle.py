"""Compare inference-only bundles against saved parent and hydrogen outputs."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from cfm_mol.latentspring_generator import LatentSpringGenerator
from cfm_mol import matched_egnn as base,connectivity_feedback as feedback
from cfm_mol.matched_physical_connection import PhysicalFieldTransform

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','bundle','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();assert not a.out.exists();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model=LatentSpringGenerator(a.bundle,device=a.device);rows=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name in (['fm','gaga'] if a.device=='cuda' else ['fm']):
        f=a.project/f'runs/hydrogen_physical_confirmation_v1/s0/parents/{name}/{name}_a0_c14.pt';g=a.project/f'runs/hydrogen_physical_confirmation_v1/s0/readouts/raw/{name}_molecule_start0_c14.pt'
        parent=torch.load(f,map_location='cpu',weights_only=False);expected=torch.load(g,map_location='cpu',weights_only=False)
        result=model.generate(parent['condition']['numbers'],samples=16,seed=63051,stream=14,family=name)
        torch.testing.assert_close(result['initial_positions'],parent['initial_positions'],atol=0,rtol=0)
        torch.testing.assert_close(result['parent_positions'],parent['positions'],atol=1e-3,rtol=0);torch.testing.assert_close(result['positions'],expected['positions'],atol=1e-3,rtol=0)
        assert float((result['positions']-expected['positions']).square().mean().sqrt())<=1e-4
        torch.testing.assert_close(result['hydrogen_changed'],expected['decoder_info']['changed'],atol=0,rtol=0)
        assert result['costs']['attempted']==result['costs']['returned']==16 and result['costs']['oracle_queries']==0
        rows.append(dict(family=name,max_coordinate_difference_A=float((result['positions']-expected['positions']).abs().max()),changed=int(result['hydrogen_changed'].sum()),costs=result['costs'],reference_files={str(p.relative_to(a.project)):sha(p) for p in [f,g]}))
    transfers=[];classification_pairs=[]
    if a.device=='cuda':
        for target,source in [('fm','gaga'),('gaga','fm')]:
            fresh=LatentSpringGenerator(a.bundle,device=a.device);file=a.project/f'runs/cross_generator_head_v1/s0/parents/{target}/{target}_cross_a0_c0.pt';expected=torch.load(file,map_location='cpu',weights_only=False)
            result=fresh.generate(expected['condition']['numbers'],samples=16,seed=64601,family=target,head_family=source,hydrogen=False)
            torch.testing.assert_close(result['initial_positions'],expected['initial_positions'],atol=0,rtol=0)
            parent,_,_,spec,_,prior,context=fresh.loaded[target+'_s0'];head=fresh.heads[source+'_s0'];transform=PhysicalFieldTransform(parent,spec,head,4.,strength_limit=4.);direct=[];numbers=expected['condition']['numbers']
            for begin in [0,8]:
                seed=64601*1000003+begin
                if context:x,_=feedback.sample(parent,numbers,spec,prior,context,seed,8,128,field_transform=transform)
                else:x,_=base.sample(parent,numbers,spec['kind'],spec,prior,seed,8,128,field_transform=transform)
                direct.append(x.cpu().double())
            direct=torch.cat(direct);torch.testing.assert_close(result['positions'],direct,atol=0,rtol=0)
            # Archived generation used CUDA scatter-add reductions without
            # deterministic mode. Check physical-scale agreement separately
            # from exact equality to the direct deterministic sampler.
            difference=result['positions']-expected['positions']
            torch.testing.assert_close(result['positions'],expected['positions'],atol=1e-3,rtol=0)
            assert float(difference.square().mean().sqrt())<=1e-4
            assert set(fresh.loaded)=={target+'_s0'} and result['costs']['oracle_queries']==0
            classification_pairs.append((result['positions'],expected['positions'],expected['condition']))
            transfers.append(dict(target=target,head_source=source,source_parent_loaded=False,max_coordinate_difference_A=float(difference.abs().max()),rms_coordinate_difference_A=float(difference.square().mean().sqrt()),same_process_direct_sampler_difference_A=float((result['positions']-direct).abs().max()),archived_max_coordinate_tolerance_A=1e-3,archived_rms_coordinate_tolerance_A=1e-4,reference_sha256=sha(file),costs=result['costs']))
    assert 'cfm_mol.energy_oracle' not in sys.modules and 'scripts.research.run_matched_connection' not in sys.modules
    from scripts.research.audit_generator_output_support import assess
    if a.device=='cuda':
        for record,(observed,archived,condition) in zip(transfers,classification_pairs):
            left=assess(observed,condition,list(range(16)));right=assess(archived,condition,list(range(16)))
            for observed_row,archived_row in zip(left['records'],right['records']):
                for key in ['graph_supported','geometrically_supported']:assert observed_row[key]==archived_row[key]
            record['exact_same_process_sampler_match']=record['same_process_direct_sampler_difference_A']==0.
            record['graph_and_geometry_classifications_identical']=True
    receipt=dict(complete=True,device=a.device,rows=rows,transfers=transfers,deterministic_reduction_mode=True,bundle_manifest_sha256=sha(a.bundle/'manifest.json'),inference_module_sha256=sha(a.project/'cfm_mol/latentspring_generator.py'),new_independent_scientific_attempts=0,verification_parent_trajectory_replays=16*len(rows)+32*len(transfers),training_or_oracle_imported=False)
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
