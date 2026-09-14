#!/usr/bin/env python3
"""Independently check exact training composition exclusions in the new FM run."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from cfm_mol.electronic_metadata import ElectronicMetadata
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['project','run','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 torch.set_num_threads(2)
 datafile=Path('/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed/train_data_processed.pt')
 metadata=ElectronicMetadata(a.project/'runs/raw_metadata_replay_v1','train',list(range(1,84)))
 metadata.verify_processed_file(datafile)
 data=torch.load(str(datafile),map_location='cpu',weights_only=False,mmap=True)
 rows=[]
 for seed in [0,1]:
  path=a.project/f'research/evidence/dynamic_tree_s{seed}_v1.json';spec=json.loads(path.read_text())
  selection_path=a.run/f's{seed}/study/selection.json';selection=json.loads(selection_path.read_text())
  excluded=set()
  for filename,digest in spec['training_exclusion_manifests'].items():
   file=a.project/filename;assert sha(file)==digest
   excluded.update(r['composition_hex'] for r in json.loads(file.read_text())['rows'])
  ids=selection['selected'];assert len(ids)==spec['fm_steps']+spec['prior_validation_rows']
  assert len(ids)==len(set(ids)) and selection['excluded_compositions']==len(excluded)
  compositions=[]
  for index in ids:
   lo,hi=map(int,data['node_idx_array'][index]);numbers=data['atom_types'][lo:hi].long().numpy()+1
   assert len(numbers)==hi-lo and numbers.min()>=1 and numbers.max()<=83
   counts=np.zeros(83,dtype=np.uint16)
   for z in numbers:counts[int(z)-1]+=1
   key=counts.astype(np.uint8).tobytes().hex();assert key not in excluded
   accepted=metadata.indices[index]
   assert metadata.values['charge_known'][accepted] and metadata.values['spin_known'][accepted]
   compositions.append(key)
  row=dict(seed=seed,selected_rows=len(ids),fm_rows=spec['fm_steps'],distinct_selected_compositions=len(set(compositions)),
   excluded_compositions=len(excluded),overlap=0,selection_sha256=sha(selection_path),protocol_sha256=sha(path),
   selected_composition_sequence_sha256=hashlib.sha256(''.join(compositions).encode()).hexdigest())
  rows.append(row);print(json.dumps(row),flush=True)
 write(a.out,dict(complete=True,rows=rows,processed_train_sha256=metadata.state['protocol']['processed_sha256']['train'],
  metadata_progress_sha256=metadata.progress_sha256,scope='Independent exact count-vector loop for every selected train/prior-held row, using checksum-verified processed corpus. No generation/energy query or claim about unrecorded global pretraining.'))


if __name__=='__main__':main()
