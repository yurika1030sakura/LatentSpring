#!/usr/bin/env python3
"""Prospective monomer training selection and geometric scaffolds, no bond labels."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol.degree_tree import ORGANIC_CAPS
from cfm_mol.tree_manifold import geometric_tree,tree_geometry,from_coordinates
from cfm_mol.latent_tree_context import tree_features
from cfm_mol.electronic_metadata import ElectronicMetadata
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();spec=json.loads(a.protocol.read_text());assert spec['frozen'];torch.set_num_threads(2)
 assert {int(k):v for k,v in spec['coordination_caps'].items()}==ORGANIC_CAPS
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);start=time.perf_counter()
 meta=ElectronicMetadata(a.project/spec['metadata'],'train',list(range(1,84)))
 datafile=Path(spec['processed_train']);meta.verify_processed_file(datafile)
 assert meta.state['protocol']['processed_sha256']['train']==spec['processed_train_sha256']
 data=torch.load(str(datafile),mmap=True,map_location='cpu',weights_only=False)
 excluded=set()
 for name,digest in spec['training_exclusion_manifests'].items():
  file=a.project/name;assert sha(file)==digest
  excluded.update(r['composition_hex'] for r in json.loads(file.read_text())['rows'])
 order=np.random.default_rng(spec['data_seed']).permutation(len(meta.indices));selected=[];decisions=[];seen=0
 for index in order[:spec['max_selection_scan']]:
  seen+=1;index=int(index);lo,hi=map(int,data['node_idx_array'][index]);n=hi-lo
  if not spec['train_min_atoms']<=n<=spec['train_max_atoms']:continue
  accepted=meta.indices[index]
  if not meta.values['charge_known'][accepted] or not meta.values['spin_known'][accepted]:continue
  if meta.values['total_charge'][accepted]!=0 or meta.values['spin_multiplicity'][accepted]!=1:continue
  numbers=(data['atom_types'][lo:hi].long()+1).tolist()
  if not set(numbers)<=set(ORGANIC_CAPS) or not {1,6}<=set(numbers):continue
  key=np.bincount(np.array(numbers)-1,minlength=83).astype(np.uint8).tobytes().hex()
  if key in excluded:continue
  x=data['positions'][lo:hi].double();x=x-x.mean(0)
  edges=geometric_tree(x,numbers,spec['radial_lower']+spec['reference_margin'],spec['radial_upper']-spec['reference_margin'])
  if edges is None:
   decisions.append(dict(processed_index=index,reason='no_declared_scaffold'));continue
  c=dict(numbers=numbers,atomic_numbers=numbers,n_atoms=n,charge=0,spin_multiplicity=1,composition_hex=key,
      processed_index=index,source_split='train',requested_kT_eV=1.)
  assessment=assess(x[None],c,[0]);passed=assessment['graph_supported']==1 and assessment['validator_errors']==0
  decisions.append(dict(processed_index=index,reason='selected' if passed else 'reference_assay_failed',assessment=assessment))
  if not passed:continue
  b,inv,length=tree_geometry(edges,numbers,like=x);y,u=from_coordinates(x,b,length,spec['radial_lower'],spec['radial_upper'])
  selected.append(dict(condition=c,positions=x,tree=edges,incidence=b,inverse=inv,lengths=length,
      radial_logits=y,directions=u,context=tree_features(n,edges).float()))
  if len(selected)%500==0:print(json.dumps(dict(selected=len(selected),scanned=seen,seconds=time.perf_counter()-start)),flush=True)
  if len(selected)==spec['fm_steps']+spec['prior_validation_rows']:break
 assert len(selected)==spec['fm_steps']+spec['prior_validation_rows'],'Not enough prespecified eligible data'
 torch.save(dict(training=selected[:spec['fm_steps']],prior_validation=selected[spec['fm_steps']:]),a.out/'data.pt')
 write(a.out/'selection.json',dict(complete=True,protocol_sha256=sha(a.protocol),data_sha256=sha(a.out/'data.pt'),
  processed_train_sha256=spec['processed_train_sha256'],metadata_progress_sha256=meta.progress_sha256,scanned=seen,
  selected_indices=[r['condition']['processed_index'] for r in selected],excluded_compositions=len(excluded),
  decisions=decisions,seconds=time.perf_counter()-start,new_molecular_oracle_calls=0,
  scope='Train-only neutral organic singlets8-40 atoms; unchanged graph assay and declared bounded-edge/capped-degree geometric scaffold. Derived geometric structural supervision, no supplied chemical bond labels. Every method shares these rows.'))
 print('Preparation complete',flush=True)


if __name__=='__main__':main()
