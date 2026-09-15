#!/usr/bin/env python3
"""One fixed task-arithmetic transfer of paired physics-versus-replay updates.

Parameter arithmetic is an empirical intervention, not an exact cancellation of
functions, a KL guarantee or a new task-arithmetic principle. Coefficient is1.
"""
import argparse,json,gc
from pathlib import Path
import torch
from flowmol.model_utils.load import read_config_file
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.tree_prior_fm import restore_model,evaluate
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'] and spec['coefficient']==1.
 cfgfile=a.project/spec['config'];assert sha(cfgfile)==spec['config_sha256'];cfg=read_config_file(cfgfile);cfg['mol_fm'].pop('bgfm',None)
 states={}
 for name,ref in spec['checkpoints'].items():
  path=a.project/ref['path'];assert sha(path)==ref['sha256'];states[name]=torch.load(path,map_location='cpu',weights_only=False)
 warm=states['frozen'];replay=states['replay'];prior=prior_from_checkpoint(warm);a.out.mkdir(parents=True);out=a.out/'evaluation';out.mkdir()
 manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256']
 for method in spec['methods']:
  parent=states[spec['updates'][method]];model=restore_model(cfg,warm);parameter_names={name for name,_ in model.named_parameters()}
  merged={};delta2=0.;base2=0.
  assert set(warm['state_dict'])==set(replay['state_dict'])==set(parent['state_dict'])
  for name,value in warm['state_dict'].items():
   if name in parameter_names:
    delta=parent['state_dict'][name]-replay['state_dict'][name];merged[name]=value+delta;delta2+=float(delta.double().square().sum());base2+=float(value.double().square().sum())
   else:
    assert torch.equal(value,replay['state_dict'][name]) and torch.equal(value,parent['state_dict'][name]);merged[name]=value.clone()
  model.load_state_dict(merged,strict=True);directory=a.out/method;directory.mkdir()
  recipe={**warm['research_protocol'],'paired_update_transfer_protocol_sha256':ph,'paired_update_transfer_method':method,'coefficient':1.}
  checkpoint=directory/'last.ckpt';torch.save(dict(state_dict=merged,research_protocol=recipe,source_prior=warm['source_prior']),checkpoint)
  write(directory/'transfer.json',dict(complete=True,checkpoint_sha256=sha(checkpoint),relative_parameter_delta_norm=(delta2/base2)**.5,coefficient=1.,new_optimizer_steps=0))
  evaluate(model,prior,method,cfg,spec,ph,out,sha(checkpoint),manifest)
  del model;gc.collect();torch.cuda.empty_cache()
 worker=Path(__file__).resolve().parent/'oracle_worker.py';assert sha(worker)==spec['oracle_worker_sha256'] and sha(spec['oracle_checkpoint'])==spec['oracle_sha256']
 quality=a.out/'physical_eval';quality.mkdir();rows=[];queries=0
 for i in spec['conditions']:
  c=torch.load(out/f'{spec["methods"][0]}_c{i}.pt',map_location='cpu',weights_only=False)['condition']
  with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16) as oracle:
   assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
   for method in spec['methods']:
    sample=out/f'{method}_c{i}.pt';x=torch.load(sample,map_location='cpu',weights_only=False)['positions'];n=len(x);e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=32)
    file=quality/f'{method}_c{i}.pt';torch.save(dict(positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=(e[:n]+e[n:])/2,even_force_eV_A=(f[:n]-f[n:])/2,source_sample_sha256=sha(sample)),file)
    rows.append(dict(method=method,condition_index=i,artifact=file.name,artifact_sha256=sha(file),source_sample_sha256=sha(sample)))
   assert oracle.evaluated==oracle.requested_evaluations;queries+=oracle.evaluated
  write(quality/'results.json',dict(complete=False,rows=rows,raw_queries=queries,protocol_sha256=ph))
 assert queries==2560;write(quality/'results.json',dict(complete=True,rows=rows,raw_queries=queries,protocol_sha256=ph))
 write(a.out/'complete.json',dict(complete=True,methods=spec['methods'],protocol_sha256=ph,raw_queries=queries,new_training_steps=0,scientific_submission_ready=False))


if __name__=='__main__':main()
