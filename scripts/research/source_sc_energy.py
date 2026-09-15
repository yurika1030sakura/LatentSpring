#!/usr/bin/env python3
"""All-output energy/force readout of frozen harmonic and Gaussian generators."""
import argparse,json,time
from pathlib import Path
import torch
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen']
 auditfile=a.project/spec['structural_audit'];assert sha(auditfile)==spec['structural_audit_sha256'];audit=json.loads(auditfile.read_text());assert audit['complete']
 poolfile=a.project/spec['reference_pool'];assert sha(poolfile)==spec['reference_pool_sha256'];pool=json.loads(poolfile.read_text())
 checkpoint=Path(spec['oracle_checkpoint']);assert sha(checkpoint)==spec['oracle_sha256']
 worker=Path(__file__).resolve().parent/'oracle_worker.py';assert sha(worker)==spec['oracle_worker_sha256']
 a.out.mkdir(parents=True);start=time.perf_counter();report=dict(complete=False,protocol_sha256=ph,rows=[],requested_raw_queries=0,new_molecular_oracle_calls=0,seconds=0.,scientific_submission_ready=False)
 for i in spec['conditions']:
  files={m:a.project/spec['sources'][m]/f'{m}_c{i}.pt' for m in spec['methods']}
  samples={m:torch.load(f,map_location='cpu',weights_only=False) for m,f in files.items()};c=samples['harmonic_tree']['condition']
  assert all(x['condition']==c for x in samples.values())
  reference=next(r for r in pool['rows'] if r['condition']['composition_hex']==c['composition_hex'])
  rx=torch.tensor(reference['reference_positions'],dtype=torch.float64)
  oracle=None;startup=time.perf_counter()
  try:
   oracle=EnergyOracle(spec['oracle_interpreter'],worker,checkpoint,numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16)
   startup=time.perf_counter()-startup;assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
   er,fr=oracle.evaluate(torch.stack([rx,-rx]));ref_even=(er[0]+er[1])/2
   refpath=a.out/f'reference_c{i}.pt';torch.save(dict(positions=rx,raw_energy_eV=er,raw_force_eV_A=fr,even_energy_eV=ref_even,even_force_eV_A=(fr[0]-fr[1])/2),refpath)
   for m in spec['methods']:
    reportfile=a.project/spec['sources'][m]/f'{m}_results.json';source=json.loads(reportfile.read_text());assert any(r['seed']==spec['model_seed'] and r['method']==m and r['report_sha256']==sha(reportfile) for r in audit['artifacts'])
    row=source['rows'][i];assert sha(files[m])==row['sample_sha256'];x=samples[m]['positions'];assert len(x)==64
    e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=32);even=(e[:64]+e[64:])/2;force=(f[:64]-f[64:])/2
    force_rms=force.square().sum(-1).mean(-1).sqrt();relative=(even-ref_even)/c['n_atoms']
    artifact=a.out/f'{m}_c{i}.pt';torch.save(dict(positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=even,even_force_eV_A=force,force_rms_eV_A=force_rms,relative_energy_per_atom_eV=relative),artifact)
    report['rows'].append(dict(method=m,condition_index=i,condition=c,source_sample_sha256=sha(files[m]),source_report_sha256=sha(reportfile),artifact=artifact.name,artifact_sha256=sha(artifact),reference_artifact=refpath.name,reference_sha256=sha(refpath),graph_support=[r['graph_supported'] for r in row['records']],geometry_support=[r['geometrically_supported'] for r in row['records']],even_energy_eV=even.tolist(),relative_energy_per_atom_eV=relative.tolist(),force_rms_eV_A=force_rms.tolist(),mean_energy_eV=float(even.mean()),mean_force_rms_eV_A=float(force_rms.mean()),oracle_startup_seconds=startup))
    print(json.dumps(dict(method=m,condition=i,mean_energy_eV=float(even.mean()),mean_force_rms_eV_A=float(force_rms.mean()))),flush=True)
  finally:
   if oracle is not None:
    report['requested_raw_queries']+=oracle.requested_evaluations;report['new_molecular_oracle_calls']+=oracle.evaluated;oracle.close()
   report['seconds']=time.perf_counter()-start;write(a.out/'results.json',report)
 assert report['requested_raw_queries']==report['new_molecular_oracle_calls']==spec['raw_oracle_calls']
 report.update(complete=True,scope='Every stored output and original reference scored by the parity-symmetrized eSEN surrogate at its original charge/spin. Energy means are within composition; no physical-accuracy, temperature-law or equilibrium claim. No evaluated output is used for training.');write(a.out/'results.json',report)


if __name__=='__main__':main()
