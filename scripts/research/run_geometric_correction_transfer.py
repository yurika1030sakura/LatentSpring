"""Reuse both frozen correction fields on additional existing generative parents."""
import argparse,copy,json
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
    for target in spec['targets']:
        case=copy.deepcopy(spec);case['parents']=spec['target_parents'][target];case['evaluation_seeds']=spec['target_seeds'][target]
        for fit in [0,1]:
            info=spec['geometry_heads'][fit];path=root/info['path'];assert sha(path)==info['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False)
            field=GeometryRecoveryField(**saved['configuration']).cuda().float().eval();field.load_state_dict(saved['ema_state_dict']);field.requires_grad_(False)
            assert base.state_hash(field)==info['ema_state_sha256']
            evaluate(root,case,ph,fit,target+'_geometric_physical',field,a.out/target/f's{fit}'/'evaluation')
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,new_generation_outputs=8192,new_gfn2_attempts=8192,new_optimizer_steps=0,new_esen_queries=0))


if __name__=='__main__':main()
