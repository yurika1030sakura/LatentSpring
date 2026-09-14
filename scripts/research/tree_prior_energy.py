#!/usr/bin/env python3
"""Prespecified all-output physical readout after the connected-target comparison."""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['protocol','audit','fixed','learned','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    spec=json.loads(a.protocol.read_text());audit=json.loads(a.audit.read_text())
    assert spec['frozen'] and audit['complete'] and sha(a.audit)==spec['structural_audit_sha256']
    assert audit['protocol_sha256']==spec['generator_protocol_sha256']
    assert audit['comparisons']['fixed minus gaussian']['conditional_paired_bootstrap95'][0]>0
    a.out.mkdir(parents=True);root=Path(__file__).resolve().parents[2]
    store=Path('/n/holylabs/woo_lab/Lab/yulili/bgfm');oracle_path=store/'checkpoints/omol25/esen_sm_conserving_all.pt'
    assert sha(oracle_path)==spec['oracle_sha256']
    report=dict(complete=False,protocol_sha256=sha(a.protocol),structural_audit_sha256=sha(a.audit),rows=[],
        new_molecular_oracle_calls=0,requested_raw_queries=0,oracle_sha256=spec['oracle_sha256'],scientific_submission_ready=False,
        scope='E_plus and quadratic confinement on every fixed generated output. Geometric/graph support is reported separately; no finite-time equilibrium or final-density claim.')
    write(a.out/'results.json',report);completed=0;requested=0
    for index in spec['conditions']:
        condition=next(r['condition'] for r in audit['rows'] if r['condition_index']==index)
        oracle=None
        try:
            oracle=EnergyOracle(store/'envs/omol25/bin/python',root/'scripts/research/oracle_worker.py',oracle_path,
                numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],
                device='cpu',batch_size=16)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
            for method in spec['methods']:
                group=a.fixed if method in ['warm','gaussian','fixed'] else a.learned
                rp=group/f'evaluation/{method}_results.json';source=json.loads(rp.read_text())
                assert sha(rp)==audit['sources'][method]['results_sha256']
                row=next(r for r in source['rows'] if r['condition_index']==index)
                data_path=group/f'evaluation/{method}_c{index}.pt';assert sha(data_path)==row['sample_sha256']
                saved=torch.load(data_path,map_location='cpu',weights_only=False)
                assert saved['condition']==condition
                x=saved['positions'];assert len(x)==spec['samples_per_condition']
                chunks=[];energies=[];potentials=[]
                for begin in range(0,len(x),16):
                    part=x[begin:begin+16];both=torch.cat([part,-part]);n=len(part)
                    e,f=oracle.evaluate(both)
                    even=(e[:n]+e[n:])/2;u=even+.5*spec['restraint_eV_A2']*part.square().sum((1,2))
                    path=a.out/f'{method}_c{index}_b{begin//16}.pt'
                    torch.save(dict(positions=part,raw_energy_eV=e[:n],inverted_energy_eV=e[n:],
                        raw_force_eV_A=f[:n],inverted_force_eV_A=f[n:],even_energy_eV=even,unrestricted_potential_eV=u),path)
                    energies.append(even);potentials.append(u);chunks.append(dict(path=path.name,sha256=sha(path)))
                    report.update(new_molecular_oracle_calls=completed+oracle.evaluated,requested_raw_queries=requested+oracle.requested_evaluations)
                    write(a.out/'results.json',report)
                even=torch.cat(energies);u=torch.cat(potentials)
                geom=torch.tensor([r['geometrically_supported'] for r in row['records']])
                graph=torch.tensor([r['graph_supported'] for r in row['records']])
                out=dict(method=method,condition_index=index,condition=condition,source_sample_sha256=sha(data_path),chunks=chunks,
                    even_energy_eV=even.tolist(),unrestricted_potential_eV=u.tolist(),geometric_support=geom.tolist(),graph_support=graph.tolist(),
                    mean_even_energy_eV=float(even.mean()),mean_unrestricted_potential_eV=float(u.mean()),
                    geometric_support_count=int(geom.sum()),graph_support_count=int(graph.sum()),
                    mean_potential_on_geometric_support_eV=float(u[geom].mean()) if geom.any() else None,
                    mean_potential_on_graph_support_eV=float(u[graph].mean()) if graph.any() else None,
                    scope='Conditional energy means have method-specific support denominators; they are not unbiased equilibrium expectations.')
                report['rows'].append(out);write(a.out/'results.json',report)
                print(json.dumps({k:out[k] for k in ['method','condition_index','mean_even_energy_eV','geometric_support_count','graph_support_count']}),flush=True)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            if oracle is not None:
                completed+=oracle.evaluated;requested+=oracle.requested_evaluations
                report.update(new_molecular_oracle_calls=completed,requested_raw_queries=requested)
                oracle.close()
            write(a.out/'results.json',report)
    assert completed==requested==spec['raw_oracle_calls']
    report['complete']=True;write(a.out/'results.json',report)


if __name__=='__main__':main()
