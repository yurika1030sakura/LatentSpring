#!/usr/bin/env python3
"""Reference qualification and fixed rank selection before model evaluation."""
import argparse,json
from pathlib import Path
import torch
from scripts.research.audit_generator_output_support import assess
from scripts.research.tree_prior_fm import geometry_counts
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['pool','panel','audit']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.panel.exists() or a.audit.exists():raise FileExistsError('Use fresh outputs')
 pool=json.loads(a.pool.read_text());assert pool['complete'];decisions=[];accepted=[]
 for i,r in enumerate(pool['rows']):
  c=dict(r['condition'],numbers=r['condition']['atomic_numbers']);x=torch.tensor(r['reference_positions'],dtype=torch.float64)[None]
  assessment=assess(x,c,[0]);geometry=geometry_counts(x,c['numbers']);passed=assessment['graph_supported']==1 and assessment['validator_errors']==0 and geometry==dict(disconnected=0,overlap=0)
  decisions.append(dict(pool_index=i,composition_hex=c['composition_hex'],qualified=passed,assessment=assessment,geometry=geometry))
  if passed:accepted.append((c['selection_rank'],i,r['condition']))
 selected=[];counts={}
 for lo,hi in [(17,20),(21,24),(25,28)]:
  group=sorted((rank,i,c) for rank,i,c in accepted if lo<=c['n_atoms']<=hi);counts[f'{lo}-{hi}']=len(group)
  if len(group)<8:continue
  selected.extend(dict(c,pool_index=i,panel_stratum=[lo,hi,'neutral','singlet']) for _,i,c in group[:8])
 write(a.audit,dict(complete=True,source_pool_sha256=sha(a.pool),decisions=decisions,qualified_by_size=counts,selected=len(selected),selection='Eight minimum pre-existing metadata hash ranks in each17-20/21-24/25-28 size bin after unchanged reference assay; no model-output or energy selection.',new_molecular_oracle_calls=0))
 assert len(selected)==24,(len(selected),counts)
 assert len({c['composition_hex'] for c in selected})==24 and not set(pool['excluded_compositions']).intersection(c['composition_hex'] for c in selected)
 write(a.panel,dict(complete=True,role='new_development',rows=selected,source_pool=str(a.pool),source_pool_sha256=sha(a.pool),qualification_audit=str(a.audit),qualification_audit_sha256=sha(a.audit),source_archive_sha256=pool['source_archive_sha256'],old_partition_seed=pool['old_partition_seed'],excluded_manifests=pool['excluded_manifests'],selection='24 fresh reference-qualified compositions; eight per17-20/21-24/25-28 bin. Freeze model/metric protocols before generation.',reserved_outcomes_allowed=False,new_molecular_oracle_calls=0,scientific_submission_ready=False))
 print(json.dumps(dict(qualified=len(accepted),counts=counts,selected=24)),flush=True)


if __name__=='__main__':main()
