"""Export inference weights without optimizer states, training data, or oracle models."""
import argparse,json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.physical_connection import make_physical_connection
from scripts.research.run_matched_physical import make_model
from scripts.research.train_electronic_fm import sha

def portable_spec(spec):
    names=['upstream','upstream_git','upstream_sha256','upstream_args','atomic_numbers','initialization_seed','kind',
        'context','two_pass','edge_log_width','gaga_max_t','data_variance_per_dof','tree_regularization','model_max_atoms']
    return {k:spec[k] for k in names if k in spec}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fits',nargs='+',type=int,default=list(range(5)));a=p.parse_args();torch.set_num_threads(2);root=a.project.resolve();a.out.mkdir(parents=True,exist_ok=False)
    manifest=dict(format='latentspring_egnn_inference_v1',models={},files={},provenance={},fixed_settings=dict(physical_strength=4.,backbone_calls=128,hydrogen_steps=4,hydrogen_velocity_cap_A=2.,batch_size=8),
        scope='Shared EGNN composition-conditioned generator with learned physical correction and optional conditional H readout. Neutral organic singlets; validation covers17--40 atoms. No equilibrium or comprehensive-superiority guarantee. Original dataset and upstream-code licenses continue to apply.')
    for si in a.fits:
        run=root/f'runs/seed_replication_v1/evaluation/s{si}';done=json.loads((run/'complete.json').read_text());assert done['complete'];protocol=json.loads((run/'resolved_protocol.json').read_text())
        hp=root/f'runs/seed_replication_v1/hydrogen_training/s{si}/step_10000.pt';hs=torch.load(hp,map_location='cpu',weights_only=False);hm=base.initialize(hs['network_spec'],'cpu');hm.load_state_dict(hs['ema_state_dict'],strict=True)
        hname=f'hydrogen_s{si}.pt';torch.save(dict(spec=portable_spec(hs['network_spec']),state_dict=hs['ema_state_dict'],state_sha256=base.state_hash(hm)),a.out/hname);manifest['provenance'][hname]=dict(source_checkpoint=str(hp.relative_to(root)),source_checkpoint_sha256=sha(hp))
        for family in ['fm','gaga']:
            arm=protocol['parents'][si][family];pp=root/arm['checkpoint'];assert sha(pp)==arm['checkpoint_sha256'];ps=torch.load(pp,map_location='cpu',weights_only=False);pm=make_model(arm,ps['ema_state_dict'],device='cpu')
            name=f'{family}_s{si}.pt';torch.save(dict(spec=portable_spec(arm['spec']),state_dict=ps['ema_state_dict'],state_sha256=base.state_hash(pm)),a.out/name)
            info=done['heads'][family];file=root/info['path'];assert sha(file)==info['sha256'];saved=torch.load(file,map_location='cpu',weights_only=False);head=make_physical_connection(**saved['configuration']);head.load_state_dict(saved['state_dict'],strict=True)
            hn=f'{family}_head_s{si}.pt';torch.save(dict(configuration=saved['configuration'],state_dict=saved['state_dict'],state_sha256=base.state_hash(head)),a.out/hn)
            manifest['models'][f'{family}_s{si}']=dict(parent=name,head=hn,hydrogen=hname)
            for name_,source in [(name,pp),(hn,file)]:manifest['provenance'][name_]=dict(source_checkpoint=str(source.relative_to(root)),source_checkpoint_sha256=sha(source))
    for f in a.out.glob('*.pt'):manifest['files'][f.name]=sha(f)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(models=len(manifest['models']),files=len(manifest['files']),bytes=sum(f.stat().st_size for f in a.out.glob('*.pt')))))

if __name__=='__main__':main()
