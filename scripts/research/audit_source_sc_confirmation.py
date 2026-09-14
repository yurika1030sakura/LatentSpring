#!/usr/bin/env python3
"""Verify the frozen two-source comparison on ten additional compositions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source,geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import intervals,flags
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['project','run','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 panelpath=a.project/'research/evidence/source_sc_confirmation_panel_v1.json';panel=json.loads(panelpath.read_text())
 overlapfile=a.project/'research/evidence/source_sc_confirmation_overlap_v1.json';overlap=json.loads(overlapfile.read_text())
 assert overlap['complete'] and overlap['panel_sha256']==sha(panelpath)
 assert all(r['overlapping_compositions']==0 for r in overlap['corpora'].values())
 parentfile=a.project/'research/evidence/source_sc_audit_v1.json';parent=json.loads(parentfile.read_text());assert parent['complete']
 poolfile=a.project/'research/evidence/monomer_reference_pool_v1.json';assert sha(poolfile)==panel['source_pool_sha256'];pool=json.loads(poolfile.read_text())
 qfile=a.project/'research/evidence/monomer_qualification_v1.json';assert sha(qfile)==panel['qualification_audit_sha256'];qualified={r['candidate_index'] for r in json.loads(qfile.read_text())['decisions'] if r['qualified']}
 oldfile=a.project/'research/evidence/monomer_panel_v1.json';assert sha(oldfile)==panel['excluded_monomer_panel_sha256'];old={r['composition_hex'] for r in json.loads(oldfile.read_text())['rows']}
 remaining=[r for r in pool['rows'] if r['condition']['candidate_index'] in qualified and r['condition']['composition_hex'] not in old]
 assert [r['condition'] for r in remaining]==panel['rows'] and len(remaining)==10
 for row in remaining:
  c=dict(row['condition'],numbers=row['condition']['atomic_numbers']);x=torch.tensor(row['reference_positions'],dtype=torch.float64)[None]
  assert assess(x,c,[0])['graph_supported']==1 and geometry_counts(x,c['numbers'])==dict(disconnected=0,overlap=0)
 summary={};reports={};artifacts=[];total=0
 for s in [0,1]:
  specfile=a.project/f'research/evidence/source_sc_confirmation_s{s}_v1.json';spec=json.loads(specfile.read_text());ph=sha(specfile)
  assert spec['frozen'] and spec['condition_manifest_sha256']==sha(panelpath) and spec['source_sc_selection_audit_sha256']==sha(parentfile)
  root=a.run/f's{s}/evaluation';done=json.loads((root/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['methods']==spec['methods']
  summary[s]={};reports[s]={}
  for m in spec['methods']:
   ref=spec['frozen_references'][m];ckpt=a.project/ref['checkpoint'];digest=sha(ckpt);assert digest==ref['checkpoint_sha256']
   assert any(r['seed']==s and r['method']==m and r['checkpoint_sha256']==digest for r in parent['artifacts'])
   state=torch.load(ckpt,map_location='cpu',weights_only=False);prior=prior_from_checkpoint(state);del state
   reportfile=root/f'{m}_results.json';report=json.loads(reportfile.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==digest
   assert [r['condition_index'] for r in report['rows']]==list(range(10))
   for i,row in enumerate(report['rows']):
    file=root/f'{m}_c{i}.pt';assert sha(file)==row['sample_sha256'];saved=torch.load(file,map_location='cpu',weights_only=False);c=saved['condition']
    assert c==row['condition'] and c['atomic_numbers']==panel['rows'][i]['atomic_numbers'] and c['composition_hex']==panel['rows'][i]['composition_hex']
    assert c['charge']==0 and c['spin_multiplicity']==1 and 'reference_positions' not in c and 'energy_eV' not in c
    expected=[spec['evaluation_seed']*1000003+i*100003+j for j in range(64)];assert saved['seeds']==expected
    assert saved['positions'].shape==saved['initial_positions'].shape==(64,c['n_atoms'],3)
    for j,draw_seed in enumerate(expected):
     x,tree=sample_source(prior,c['numbers'],0,1,draw_seed);torch.testing.assert_close(x,saved['initial_positions'][j],atol=0,rtol=0);assert tree==saved['auxiliary_tree_edges'][j];total+=1
    result=assess(saved['positions'],c,list(range(64)));assert all(result[k]==row[k] for k in result)
    assert geometry_counts(saved['positions'],c['numbers'])==row['final_geometry']
   reports[s][m]=report['rows'];summary[s][m]={k:sum(r[k] for r in report['rows']) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','validator_errors','generation_seconds']}
   artifacts.append(dict(seed=s,method=m,checkpoint_sha256=digest,report_sha256=sha(reportfile),protocol_sha256=ph))
   print(json.dumps(dict(seed=s,method=m,**summary[s][m])),flush=True)
 rng=np.random.default_rng(35991);comparisons={}
 for metric in ['graph_supported','geometrically_supported']:
  d=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[s]['harmonic_tree'],reports[s]['gaussian'])]) for s in [0,1]])
  comparisons[metric]=intervals(d,[0]*10,rng)
 point=all(summary[s]['harmonic_tree']['graph_supported']>summary[s]['gaussian']['graph_supported'] for s in [0,1])
 distinct=sum(summary[s]['harmonic_tree']['distinct_connectivity']-summary[s]['gaussian']['distinct_connectivity'] for s in [0,1])/1280
 write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,artifacts=artifacts,source_and_structural_outputs_replayed=total,replayed_reference_geometries=10,overlap_audit_sha256=sha(overlapfile),parent_selection_audit_sha256=sha(parentfile),confirmation_gate_passed=point and comparisons['graph_supported']['paired_draw95'][0]>0 and distinct>=-.02,
  new_training_runs=0,new_molecular_oracle_calls=0,scientific_submission_ready=False,scope='All10 remaining reference-qualified pool compositions,17-28 atoms; excluded from old21-composition monomer generation panel and two verified processed corpora. Existing two model seeds reused without fitting. Conditional and descriptive composition intervals are not broad population guarantees or an originality/Boltzmann certificate.'))


if __name__=='__main__':main()
