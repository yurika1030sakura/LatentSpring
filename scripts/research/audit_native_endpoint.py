#!/usr/bin/env python3
"""Audit original-checkpoint conditional endpoint outputs and frozen references."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts.research.tree_prior_fm import sample_source,geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import intervals,flags
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','run','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 reports={};summary={};artifacts=[];total=0
 parentfile=a.project/'research/evidence/source_sc_confirmation_audit_v1.json';parent=json.loads(parentfile.read_text());assert parent['complete']
 for seed in [0,1]:
  path=a.project/f'research/evidence/native_endpoint_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path);assert spec['frozen'] and spec['parent_confirmation_audit_sha256']==sha(parentfile)
  assert sha(spec['checkpoint'])==spec['checkpoint_sha256'];manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256'];conditions=json.loads(manifest.read_text())['rows']
  root=a.run/f's{seed}/evaluation';done=json.loads((root/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['methods']==spec['methods']
  reports[seed]={};summary[seed]={}
  for method in spec['methods']:
   file=root/f'{method}_results.json';report=json.loads(file.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==spec['checkpoint_sha256']
   assert [r['condition_index'] for r in report['rows']]==list(range(10))
   for i,row in enumerate(report['rows']):
    sample=root/f'{method}_c{i}.pt';assert sha(sample)==row['sample_sha256'];saved=torch.load(sample,map_location='cpu',weights_only=False);c=saved['condition']
    assert c==row['condition'] and c['composition_hex']==conditions[i]['composition_hex'] and c['atomic_numbers']==conditions[i]['atomic_numbers']
    assert c['charge']==0 and c['spin_multiplicity']==1 and saved['positions'].shape==(64,c['n_atoms'],3)
    seeds=[spec['evaluation_seed']*1000003+i*100003+j for j in range(64)];assert saved['seeds']==seeds
    for j,s in enumerate(seeds):
     x,t=sample_source(None,c['numbers'],0,1,s);torch.testing.assert_close(x,saved['initial_positions'][j],atol=0,rtol=0);assert saved['auxiliary_tree_edges'][j]==t;total+=1
    result=assess(saved['positions'],c,list(range(64)));assert all(row[k]==v for k,v in result.items());assert geometry_counts(saved['positions'],c['numbers'])==row['final_geometry']
   reports[seed][method]=report['rows'];artifacts.append(dict(stream=seed,method=method,checkpoint_sha256=spec['checkpoint_sha256'],report_sha256=sha(file),protocol_sha256=ph))
  for method in ['gaussian','harmonic_tree']:
   file=a.project/f'runs/source_sc_confirmation_v1/s{seed}/evaluation/{method}_results.json';assert any(r['seed']==seed and r['method']==method and r['report_sha256']==sha(file) for r in parent['artifacts'])
   reports[seed][method]=json.loads(file.read_text())['rows']
  for m,rows in reports[seed].items():summary[seed][m]={k:sum(r[k] for r in rows) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','validator_errors','generation_seconds']}
  print(json.dumps(dict(stream=seed,summary=summary[seed])),flush=True)
 rng=np.random.default_rng(36591);comparisons={}
 for method in ['native_history','clamped_history']:
  comparisons['harmonic_tree minus '+method]={}
  for metric in ['graph_supported','geometrically_supported']:
   diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[s]['harmonic_tree'],reports[s][method])]) for s in [0,1]])
   comparisons['harmonic_tree minus '+method][metric]=intervals(diff,[0]*10,rng)
 write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,artifacts=artifacts,new_outputs_replayed=total,new_training_runs=0,new_molecular_oracle_calls=0,scientific_submission_ready=False,
  scope='One original30k checkpoint, two sampling streams, native endpoint Euler and bootstrap/history at128 actual denoiser calls; current composition/no-bond interface clamped. This is task-adapted native sampling, not the published joint-generator benchmark. Adapted Gaussian/harmonic models have additional shared pretraining and3000 specialized updates. Training costs and conditioning procedures differ, so differences do not isolate source choice or prove SOTA.'))


if __name__=='__main__':main()
