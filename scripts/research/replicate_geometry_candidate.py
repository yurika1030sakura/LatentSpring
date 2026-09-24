"""Repeat the frozen selected geometry recipe on the three additional parent fits."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.geometry_recovery_field import GeometryRecoveryField
from scripts.research.run_geometry_recovery_field import evaluate
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def frozen_json(path,data):
    if path.exists():assert json.loads(path.read_text())==data
    else:write(path,data)


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fits',type=int,nargs='+',required=True);a=p.parse_args();root=a.project.resolve();master=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    assert master['frozen'] and set(a.fits)<=set([2,3,4]);a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    recovery=json.loads((root/master['recovery_protocol']).read_text());field=json.loads((root/master['field_protocol']).read_text())
    refs=json.loads((root/master['reference_selection']).read_text());assert refs['complete'] and sha(root/master['reference_selection'])==master['reference_selection_sha256']
    for fit in a.fits:
        folder=a.out/f's{fit}';folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph;continue
        parent=master['parents'][fit];head=master['physical_heads'][fit]
        rec=copy.deepcopy(recovery);rec.update(parent_fit_index=fit,replication_protocol_sha256=ph,
            parents=[parent,parent],physical_heads=[head,head],conditions=master['conditions'],
            batch_seeds=[master['recovery_batch_seeds'][fit]]*2,noise_seeds=[master['recovery_noise_seeds'][fit]]*2,
            evaluation_seeds=[master['evaluation_seeds'][fit]]*2)
        rp=folder/'recovery_protocol.json';frozen_json(rp,rec)
        subprocess.run([sys.executable,'-s','-u','-m','scripts.research.run_geometry_recovery','--project',str(root),
            '--protocol',str(rp),'--out',str(folder/'parent'),'--fit','0','--variant','recovery_local','--train-only'],check=True)
        fs=copy.deepcopy(field);fs.update(parent_fit_index=fit,replication_protocol_sha256=ph,
            parents=[parent,parent],physical_heads=[head,head],conditions=master['conditions'],
            context_seeds=[master['geometry_initialization_seeds'][fit]]*2,batch_seeds=[master['geometry_batch_seeds'][fit]]*2,
            noise_seeds=[master['geometry_noise_seeds'][fit]]*2,evaluation_seeds=[master['evaluation_seeds'][fit]]*2)
        fp=folder/'field_protocol.json';frozen_json(fp,fs)
        reference_view=dict(refs,protocol_sha256=sha(fp),source_reference_audit_sha256=master['reference_selection_sha256'])
        reference_file=folder/'reference_selection.json';frozen_json(reference_file,reference_view)
        subprocess.run([sys.executable,'-s','-u','-m','scripts.research.run_geometry_recovery_field','--project',str(root),
            '--protocol',str(fp),'--out',str(folder/'geometry'),'--references',str(reference_file),
            '--fit','0','--variant','radial_geometry','--train-only'],check=True)
        parent_training=json.loads((folder/'parent/training.json').read_text());geometry_training=json.loads((folder/'geometry/training.json').read_text())
        assert parent_training['complete'] and geometry_training['complete']
        parent_ck=folder/'parent/last.ckpt';geometry_ck=folder/'geometry/last.ckpt'
        assert sha(parent_ck)==parent_training['checkpoint_sha256'] and sha(geometry_ck)==geometry_training['checkpoint_sha256']
        resolved=copy.deepcopy(master);resolved['parents'][fit]=dict(parent,checkpoint=str(parent_ck.relative_to(root)),checkpoint_sha256=sha(parent_ck))
        saved=torch.load(geometry_ck,map_location='cpu',weights_only=False)
        geometry=GeometryRecoveryField(**saved['configuration']).cuda().float().eval();geometry.load_state_dict(saved['ema_state_dict']);geometry.requires_grad_(False)
        assert base.state_hash(geometry)==geometry_training['ema_state_sha256']
        frozen_json(folder/'resolved_inference.json',dict(protocol_sha256=ph,parent=resolved['parents'][fit],
            geometry=dict(path=str(geometry_ck.relative_to(root)),sha256=sha(geometry_ck),ema_state_sha256=geometry_training['ema_state_sha256']),physical_head=head))
        evaluate(root,resolved,ph,fit,'full_geometry_physics',geometry,folder/'evaluation')
        done=dict(complete=True,protocol_sha256=ph,parent_fit_index=fit,parent_training_sha256=sha(folder/'parent/training.json'),
            geometry_training_sha256=sha(folder/'geometry/training.json'),evaluation_complete_sha256=sha(folder/'evaluation/complete.json'),
            new_backbone_updates=4000,new_geometry_updates=10000,new_generation_outputs=1024,new_gfn2_attempts=1024,new_esen_queries=0)
        write(folder/'complete.json',done);print(json.dumps(dict(phase='replication_complete',**done)),flush=True)


if __name__=='__main__':main()
