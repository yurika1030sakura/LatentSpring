#!/usr/bin/env python3
"""Frozen four-model confirmation on fresh compositions, followed by eSEN readout."""
import argparse,json,gc,time
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
 spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'];start=time.perf_counter()
 manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256'];panel=json.loads(manifest.read_text())
 overlapfile=a.project/spec['overlap_audit'];assert sha(overlapfile)==spec['overlap_audit_sha256'];overlap=json.loads(overlapfile.read_text());assert overlap['complete'] and overlap['panel_sha256']==sha(manifest) and all(r['overlapping_compositions']==0 for r in overlap['corpora'].values())
 cfgfile=a.project/spec['config'];assert sha(cfgfile)==spec['config_sha256'];cfg=read_config_file(cfgfile);cfg['mol_fm'].pop('bgfm',None)
 assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
 a.out.mkdir(parents=True);out=a.out/'evaluation';out.mkdir();completed=[]
 for method in spec['methods']:
  ref=spec['checkpoints'][method];path=a.project/ref['path'];assert sha(path)==ref['sha256'];state=torch.load(path,map_location='cpu',weights_only=False)
  assert state['research_protocol']['source_prior_kind']==spec['source_kinds'][method]
  assert state['research_protocol']['geometry_self_conditioning']==dict(zero_init=True,deep_supervision=True,edge_feedback='clamped')
  model=restore_model(cfg,state);prior=prior_from_checkpoint(state)
  evaluate(model,prior,method,cfg,spec,ph,out,ref['sha256'],manifest)
  completed.append(method);write(out/'complete.json',dict(complete=completed==spec['methods'],methods=completed,protocol_sha256=ph))
  del model,state,prior;gc.collect();torch.cuda.empty_cache()
 quality=a.out/'physical_eval';quality.mkdir();rows=[];queries=0
 poolfile=a.project/panel['source_pool'];assert sha(poolfile)==panel['source_pool_sha256'];pool=json.loads(poolfile.read_text())
 worker=Path(__file__).resolve().parent/'oracle_worker.py';assert sha(worker)==spec['oracle_worker_sha256'];assert sha(spec['oracle_checkpoint'])==spec['oracle_sha256']
 for i in spec['conditions']:
  c=torch.load(out/f'harmonic_tree_c{i}.pt',map_location='cpu',weights_only=False)['condition'];reference=pool['rows'][panel['rows'][i]['pool_index']];assert reference['condition']['atomic_numbers']==c['numbers'] and reference['condition']['raw_index']==c['raw_index'];rx=torch.tensor(reference['reference_positions'],dtype=torch.float64)
  with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16) as oracle:
   assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
   er,fr=oracle.evaluate(torch.stack([rx,-rx]));rp=quality/f'reference_c{i}.pt';torch.save(dict(positions=rx,raw_energy_eV=er,raw_force_eV_A=fr),rp)
   rows.append(dict(method='reference',condition_index=i,artifact=rp.name,artifact_sha256=sha(rp)))
   for method in spec['methods']:
    path=out/f'{method}_c{i}.pt';sample=torch.load(path,map_location='cpu',weights_only=False);assert sample['condition']==c;x=sample['positions'];n=len(x);e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=32)
    file=quality/f'{method}_c{i}.pt';torch.save(dict(source_sample_sha256=sha(path),positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=(e[:n]+e[n:])/2,even_force_eV_A=(f[:n]-f[n:])/2),file)
    rows.append(dict(method=method,condition_index=i,artifact=file.name,artifact_sha256=sha(file),source_sample_sha256=sha(path)))
   assert oracle.evaluated==oracle.requested_evaluations;queries+=oracle.evaluated
  write(quality/'results.json',dict(complete=False,rows=rows,raw_queries=queries,protocol_sha256=ph));print(json.dumps(dict(phase='esen',condition=i,queries=queries)),flush=True)
 assert queries==spec['esen_queries_per_seed'];write(quality/'results.json',dict(complete=True,rows=rows,raw_queries=queries,protocol_sha256=ph))
 write(a.out/'complete.json',dict(complete=True,methods=completed,protocol_sha256=ph,raw_esen_queries=queries,new_training_steps=0,seconds=time.perf_counter()-start,scientific_submission_ready=False))


if __name__=='__main__':main()
