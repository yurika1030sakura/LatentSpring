"""Re-evaluate the fixed geometric candidate and matched continuation on the64 panel."""
import argparse,json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.geometry_recovery_field import GeometryRecoveryField
from scripts.research.run_geometry_recovery_field import evaluate
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    assert spec['frozen'];a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for fit in [0,1]:
        info=spec['geometry_heads'][fit];path=root/info['path'];assert sha(path)==info['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        field=GeometryRecoveryField(**saved['configuration']).cuda().float().eval()
        field.load_state_dict(saved['ema_state_dict']);field.requires_grad_(False)
        assert base.state_hash(field)==info['ema_state_sha256']
        for method in spec['variants']:
            case=dict(spec,parents=spec['parents_by_method'][method])
            geometry_strength=1. if method=='full_geometry_physics' else 0.
            evaluate(root,case,ph,fit,method,field,a.out/f's{fit}'/method/'evaluation',
                geometry_strength=geometry_strength,physical_strength=4.)
    write(a.out/'generation_complete.json',dict(complete=True,protocol_sha256=ph,
        new_generation_outputs=4096,new_gfn2_attempts=4096,new_optimizer_steps=0,new_esen_queries=0))


if __name__=='__main__':main()
