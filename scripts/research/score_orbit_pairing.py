#!/usr/bin/env python3
"""Execute the frozen all-output energy follow-up only after the structural gate."""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','run','audit','out']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    protocol=json.loads(a.protocol.read_text())
    audit=json.loads(a.audit.read_text());generation=json.loads((a.run/'results.json').read_text())
    assert protocol['frozen'] and audit['complete'] and generation['complete']
    assert audit['protocol_sha256']==generation['protocol_sha256']==sha(a.protocol)
    assert audit['source_results_sha256']==sha(a.run/'results.json')
    a.out.mkdir(parents=True)
    report=dict(complete=False,protocol_sha256=sha(a.protocol),audit_sha256=sha(a.audit),
        source_results_sha256=sha(a.run/'results.json'),gate_passed=audit['proceed_to_frozen_energy_check'],
        acknowledged_raw_queries=0,requested_raw_queries=0,new_molecular_oracle_calls=0,scientific_submission_ready=False,rows=[])
    if not report['gate_passed']:
        report.update(complete=True,skipped=True,reason='The prespecified structural screen did not pass; no physical queries are released.')
        write(a.out/'results.json',report);print(json.dumps(report),flush=True);return
    root=Path(__file__).resolve().parents[2]
    store=Path('/n/holylabs/woo_lab/Lab/yulili/bgfm')
    checkpoint=store/'checkpoints/omol25/esen_sm_conserving_all.pt'
    digest=sha(checkpoint)
    assert digest=='01f63da2d071e39fc46a5f22f8369d0fc9de317ab2ef361a76603f5661238025'
    report.update(oracle_sha256=digest,oracle_device='cpu',restraint_eV_A2=.1,
        physical_target_kT_eV=protocol['physical_target_kT_eV'],skipped=False,
        scope='Actual E_plus and COM-restraint readout for every fixed output, including structural failures. Energies are compared within the same composition and electronic state. No importance weights or equilibrium-law claim.')
    completed=0;requested_completed=0
    write(a.out/'results.json',report)
    for index in protocol['conditions']:
        condition=next(r['condition'] for r in generation['rows'] if r['condition_index']==index)
        oracle=None
        try:
            oracle=EnergyOracle(store/'envs/omol25/bin/python',root/'scripts/research/oracle_worker.py',checkpoint,
                numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],
                device='cpu',batch_size=16)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
            for method in ['warm']+protocol['methods']:
                source=a.run/f'{method}_c{index}.pt';expected=next(r for r in generation['rows'] if r['method']==method and r['condition_index']==index)
                assert sha(source)==expected['sample_sha256']
                data=torch.load(source,map_location='cpu',weights_only=False);x=data['positions']
                assert data['condition']==condition and len(x)==protocol['samples_per_condition']
                values=[];chunks=[]
                for begin in range(0,len(x),16):
                    part=x[begin:begin+16];both=torch.cat([part,-part])
                    energy,force=oracle.evaluate(both);n=len(part)
                    even=(energy[:n]+energy[n:])/2
                    potential=even+.05*part.square().sum((1,2))
                    record=dict(positions=part,raw_energy_eV=energy[:n],inverted_energy_eV=energy[n:],
                        raw_force_eV_A=force[:n],inverted_force_eV_A=force[n:],even_energy_eV=even,potential_eV=potential)
                    path=a.out/f'{method}_c{index}_b{begin//16}.pt';torch.save(record,path)
                    chunks.append(dict(path=path.name,sha256=sha(path)));values.append(potential)
                    report.update(acknowledged_raw_queries=completed+oracle.evaluated,
                        requested_raw_queries=requested_completed+oracle.requested_evaluations,
                        new_molecular_oracle_calls=completed+oracle.evaluated)
                    write(a.out/'results.json',report)
                potential=torch.cat(values)
                row=dict(method=method,condition_index=index,condition=condition,source_sample_sha256=sha(source),
                    samples=len(x),potential_eV=potential.tolist(),mean_potential_eV=float(potential.mean()),
                    median_potential_eV=float(potential.median()),graph_supported=expected['graph_supported'],
                    chunks=chunks,oracle_handshake=oracle.handshake)
                report['rows'].append(row);write(a.out/'results.json',report)
                print(json.dumps({k:row[k] for k in ['method','condition_index','samples','mean_potential_eV','graph_supported']}),flush=True)
            assert oracle.evaluated==oracle.requested_evaluations==8*protocol['samples_per_condition']
        except Exception as exc:
            report['error']=repr(exc)
            raise
        finally:
            if oracle is not None:
                completed+=oracle.evaluated;requested_completed+=oracle.requested_evaluations
                report.update(acknowledged_raw_queries=completed,requested_raw_queries=requested_completed,new_molecular_oracle_calls=completed)
                oracle.close()
            write(a.out/'results.json',report)
    assert completed==requested_completed==8*len(protocol['conditions'])*protocol['samples_per_condition']
    report['complete']=True;write(a.out/'results.json',report)


if __name__=='__main__':main()
