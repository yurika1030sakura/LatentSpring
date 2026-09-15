#!/usr/bin/env python3
"""Replay every raw physical readout and compare the frozen source methods."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.audit_source_utility import intervals


def masked_intervals(difference,mask,seed=37291):
    rng=np.random.default_rng(seed);cells=[];means=np.full(difference.shape[:2],np.nan);counts=[]
    for s in range(difference.shape[0]):
        for i in range(difference.shape[1]):
            values=difference[s,i][mask[s,i]];counts.append(dict(seed=s,condition=i,count=len(values)))
            if len(values):cells.append(values);means[s,i]=values.mean()
    if not cells:return dict(mean=None,paired_draw95=None,descriptive_composition95=None,counts=counts)
    draws=np.stack([v[rng.integers(len(v),size=(10000,len(v)))].mean(1) for v in cells]).mean(0)
    by_condition=np.array([col[np.isfinite(col)].mean() for col in means.T if np.isfinite(col).any()]);indices=rng.integers(len(by_condition),size=(10000,len(by_condition)))
    return dict(mean=float(np.mean([v.mean() for v in cells])),paired_draw95=np.quantile(draws,[.025,.975]).tolist(),descriptive_composition95=np.quantile(by_condition[indices].mean(1),[.025,.975]).tolist(),counts=counts,scope='Conditions on paired common-graph support; not an equilibrium or unselected-population expectation.')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','run','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 arrays={};records=[];total=0;artifacts=[]
 for s in [0,1]:
  path=a.project/f'research/evidence/source_sc_energy_s{s}_v1.json';spec=json.loads(path.read_text());root=a.run/f's{s}/evaluation';resultfile=root/'results.json';report=json.loads(resultfile.read_text())
  assert report['complete'] and report['protocol_sha256']==sha(path) and report['new_molecular_oracle_calls']==report['requested_raw_queries']==2580
  parentfile=a.project/spec['structural_audit'];assert sha(parentfile)==spec['structural_audit_sha256'];parent=json.loads(parentfile.read_text());assert parent['complete']
  poolfile=a.project/spec['reference_pool'];assert sha(poolfile)==spec['reference_pool_sha256'];pool=json.loads(poolfile.read_text())['rows']
  assert len(report['rows'])==20;seen=set();refs=set();arrays[s]={m:[] for m in spec['methods']}
  for row in report['rows']:
   m,i=row['method'],row['condition_index'];assert (m,i) not in seen;seen.add((m,i));c=row['condition']
   file=root/row['artifact'];assert sha(file)==row['artifact_sha256'];d=torch.load(file,map_location='cpu',weights_only=False)
   sourcefile=a.project/spec['sources'][m]/f'{m}_c{i}.pt';assert sha(sourcefile)==row['source_sample_sha256'];source=torch.load(sourcefile,map_location='cpu',weights_only=False)
   assert source['condition']==c;torch.testing.assert_close(d['positions'],source['positions'],atol=0,rtol=0)
   refpath=root/row['reference_artifact'];assert sha(refpath)==row['reference_sha256'];ref=torch.load(refpath,map_location='cpu',weights_only=False)
   original=[r for r in pool if r['condition']['composition_hex']==c['composition_hex']];assert len(original)==1 and original[0]['condition']['atomic_numbers']==c['numbers'];torch.testing.assert_close(ref['positions'],torch.tensor(original[0]['reference_positions'],dtype=torch.float64),atol=0,rtol=0)
   re=ref['raw_energy_eV'];rf=ref['raw_force_eV_A'];torch.testing.assert_close(ref['even_energy_eV'],re.mean(),atol=0,rtol=0);torch.testing.assert_close(ref['even_force_eV_A'],(rf[0]-rf[1])/2,atol=0,rtol=0)
   e=d['raw_energy_eV'];f=d['raw_force_eV_A'];assert e.shape==(128,) and f.shape==(128,c['n_atoms'],3) and torch.isfinite(e).all() and torch.isfinite(f).all()
   even=(e[:64]+e[64:])/2;force=(f[:64]-f[64:])/2;frms=force.square().sum(-1).mean(-1).sqrt();relative=(even-re.mean())/c['n_atoms']
   for key,value in [('even_energy_eV',even),('even_force_eV_A',force),('force_rms_eV_A',frms),('relative_energy_per_atom_eV',relative)]:torch.testing.assert_close(d[key],value,atol=0,rtol=0)
   for key,value in [('even_energy_eV',even),('force_rms_eV_A',frms),('relative_energy_per_atom_eV',relative)]:np.testing.assert_allclose(row[key],value.numpy(),atol=1e-12,rtol=0)
   source_report=a.project/spec['sources'][m]/f'{m}_results.json';assert sha(source_report)==row['source_report_sha256'];assert any(x['seed']==s and x['method']==m and x['report_sha256']==sha(source_report) for x in parent['artifacts'])
   sr=json.loads(source_report.read_text())['rows'][i];assert row['graph_support']==[r['graph_supported'] for r in sr['records']]
   arrays[s][m].append(dict(index=i,energy=relative.numpy(),force=frms.numpy(),mask=np.array(row['graph_support'],bool)))
   records.append(dict(seed=s,method=m,condition=i,mean_relative_energy_per_atom_eV=float(relative.mean()),mean_force_rms_eV_A=float(frms.mean()),graph_supported=int(sum(row['graph_support']))))
   total+=128
   if i not in refs:total+=2;refs.add(i)
  artifacts.append(dict(seed=s,report_sha256=sha(resultfile),protocol_sha256=sha(path)))
 assert total==5160
 rng=np.random.default_rng(37291);comparisons={}
 for metric in ['energy','force']:
  differences=[];masks=[]
  for s in [0,1]:
   h=sorted(arrays[s]['harmonic_tree'],key=lambda r:r['index']);g=sorted(arrays[s]['gaussian'],key=lambda r:r['index'])
   differences.append(np.stack([x[metric]-y[metric] for x,y in zip(h,g)]));masks.append(np.stack([x['mask']&y['mask'] for x,y in zip(h,g)]))
  difference=np.stack(differences);mask=np.stack(masks)
  comparisons[metric]=dict(all_outputs=intervals(difference,[0]*10,rng),common_graph_supported=masked_intervals(difference,mask))
 write(a.out,dict(complete=True,raw_energy_force_rows_checked=total,artifacts=artifacts,rows=records,comparisons=comparisons,new_oracle_queries_for_audit=0,scientific_submission_ready=False,scope='Original surrogate outputs and derived E_plus/F_plus replayed; no independent quantum/energy-model re-evaluation. All-output and paired common-graph-supported readouts retained separately. Lower energy is not a Boltzmann law.'))
 print(json.dumps(comparisons),flush=True)


if __name__=='__main__':main()
