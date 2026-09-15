#!/usr/bin/env python3
"""Original endpoint weights with explicitly clamped molecular conditions."""
import argparse,json
from pathlib import Path
import torch
from flowmol.model_utils.load import model_from_config,read_config_file
from cfm_mol.native_endpoint_sampling import sample_native_endpoint
from scripts.research.tree_prior_fm import evaluate
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen']
 ckpt=Path(spec['checkpoint']);assert sha(ckpt)==spec['checkpoint_sha256']
 cfgpath=a.project/spec['config'];assert sha(cfgpath)==spec['config_sha256'];cfg=read_config_file(cfgpath);cfg['mol_fm'].pop('bgfm',None)
 assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
 state=torch.load(ckpt,map_location='cpu',weights_only=False);assert state['global_step']==30000
 model=model_from_config(cfg);model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().eval();del state
 manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256']
 assert all(c['charge']==0 and c['spin_multiplicity']==1 for c in json.loads(manifest.read_text())['rows'])
 a.out.mkdir(parents=True)
 for method in spec['methods']:
  def sampler(model,g,nbi,uem,*,x0,**kwargs):
   x,info=sample_native_endpoint(model,g,nbi,uem,x0,primitive_calls=128,history=spec['history_modes'][method])
   assert info['primitive_denoiser_calls']==128;return x
  evaluate(model,None,method,cfg,spec,ph,a.out,spec['checkpoint_sha256'],manifest,sampler=sampler)
 assert sha(ckpt)==spec['checkpoint_sha256']
 write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,methods=spec['methods'],checkpoint_sha256=spec['checkpoint_sha256'],new_training_runs=0,new_molecular_oracle_calls=0))


if __name__=='__main__':main()
