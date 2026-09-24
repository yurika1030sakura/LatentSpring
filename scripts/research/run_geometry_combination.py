"""Evaluate geometric and physical fields on the selected recovery-trained parent."""
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
        folder=a.out/f's{fit}';folder.mkdir(exist_ok=True)
        for name,(geometry_strength,physical_strength) in spec['field_settings'].items():
            if name=='frozen':continue
            evaluate(root,spec,ph,fit,name,field,folder/name/'evaluation',
                geometry_strength=geometry_strength,physical_strength=physical_strength)
    write(a.out/'generation_complete.json',dict(complete=True,protocol_sha256=ph,
        new_generation_outputs=spec['budget']['new_generation_outputs'],new_gfn2_attempts=spec['budget']['new_gfn2_attempts'],
        new_optimizer_steps=0,new_esen_queries=0))


if __name__=='__main__':main()
