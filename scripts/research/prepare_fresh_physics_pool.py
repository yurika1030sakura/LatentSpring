#!/usr/bin/env python3
"""New metadata-only development candidates outside every old official candidate."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 oldpath=a.project/'runs/official_validation_audit_v1/audit.json';old=json.loads(oldpath.read_text());assert old['complete'] and old['counts']['scanned']==old['source_raw_records']
 excluded=set();old_indices=set();exclusions={}
 for name in ['official_development_candidates','official_reserved_candidates']:
  path=a.project/'research/evidence'/f'{name}.json';d=json.loads(path.read_text());assert d['complete'];exclusions[name]=sha(path)
  excluded.update(r['composition_hex'] for r in d['rows']);old_indices.update(r['raw_index'] for r in d['rows'])
 from fairchem.core.datasets import AseDBDataset
 from ase.calculators.singlepoint import SinglePointCalculator
 raw=AseDBDataset({'src':old['raw_directory']});metadata=np.load(Path(old['raw_directory'])/'metadata.npz');sizes=metadata['natoms'];assert len(raw)==len(sizes)==old['source_raw_records']
 order=np.flatnonzero((sizes>=17)&(sizes<=28));np.random.default_rng(37691).shuffle(order)
 allowed={1,5,6,7,8,9,14,15,16,17,35,53};rows=[];seen=set();scanned=0;start=time.perf_counter()
 for index in order:
  index=int(index)
  if index in old_indices:continue
  scanned+=1;atoms=raw.get_atoms(index);assert len(atoms)==sizes[index];numbers=atoms.numbers.tolist();info=atoms.info
  if not set(numbers)<=allowed or not {1,6}<=set(numbers):continue
  if info.get('charge') is None or info.get('spin') is None or int(info['charge'])!=0 or int(info['spin'])!=1:continue
  if sum(numbers)%2:continue
  key=np.bincount(np.asarray(numbers)-1,minlength=83).astype(np.uint8).tobytes();hexkey=key.hex()
  if hexkey in excluded or hexkey in seen:continue
  if int.from_bytes(hashlib.sha256(str(old['seed']).encode()+key).digest()[:8],'big')%5!=0:continue
  assert info.get('source') and isinstance(atoms.calc,SinglePointCalculator)
  x=atoms.positions.astype(np.float64);x-=x.mean(0)
  if not np.isfinite(x).all() or not np.isfinite(atoms.get_potential_energy()) or not np.isfinite(atoms.get_forces()).all():continue
  c=dict(raw_index=index,source=info['source'],reference_source=info.get('reference_source'),data_id=str(info.get('data_id','unknown')),atomic_numbers=numbers,n_atoms=len(numbers),charge=0,spin_multiplicity=1,composition_hex=hexkey,partition='new_development',selection_rank=hashlib.sha256(('37791|'+hexkey).encode()).hexdigest())
  rows.append(dict(condition=c,reference_positions=x.tolist()));seen.add(hexkey)
  if len(rows)==256:break
 assert len(rows)==256,'Insufficient frozen metadata pool'
 report=dict(complete=True,role='prospective_monomer_reference_pool',source_audit_sha256=sha(oldpath),source_archive_sha256=old['validation_archive_sha256'],source_raw_records=len(raw),selection_seed=37691,rank_seed=37791,old_partition_seed=old['seed'],excluded_manifests=exclusions,excluded_compositions=sorted(excluded),rows=rows,scanned_eligible_size_rows=scanned,seconds=time.perf_counter()-start,selection='First256 distinct neutral-singlet organic17-28-atom compositions in a fixed shuffled raw-index stream, excluding all old official candidate compositions and preserving the old development-only partition hash. References are read only after metadata eligibility; no generator outcomes, energies or forces rank candidates.',reference_coordinates_for_generation_or_training=False,new_molecular_oracle_calls=0,reserved_outcomes_allowed=False)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(pool=256,scanned=scanned,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
