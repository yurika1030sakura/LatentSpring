"""Build full-Cartesian physical TRAIN targets, preserving every selected reference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii
from cfm_mol.geometric_domain import connected_nonoverlapping
from cfm_mol.escorted_thermal_teacher import graph_key
from cfm_mol.physical_endpoint_relaxation import relax_endpoint
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from scripts.research.run_gaga_feedback import atomic_save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    assert spec['frozen'] and sha(args.project/spec['data'])==spec['data_sha256']
    data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training']
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'trajectories').mkdir()
    oracle_spec=spec['oracle'];worker=Path(__file__).resolve().parent/'oracle_worker.py'
    assert sha(worker)==oracle_spec['oracle_worker_sha256'] and sha(oracle_spec['oracle_checkpoint'])==oracle_spec['oracle_sha256']
    c=data[spec['training_rows'][0]]['condition'];rows=[];start=time.perf_counter()
    with EnergyOracle(oracle_spec['oracle_interpreter'],worker,oracle_spec['oracle_checkpoint'],
            numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],
            device='cuda',batch_size=2,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        write(args.out/'handshake.json',oracle.handshake)
        for slot,index in enumerate(spec['training_rows']):
            item=data[index];c=item['condition'];reference=item['positions'].double();reference-=reference.mean(0)
            original=infer_chemical_graph(reference,c['numbers'],c['charge']);key=graph_key(original)
            radii=covalent_radii(c['numbers']);probes=[];raw_queries=[]
            oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
            before=oracle.evaluated
            def allowed(x):
                if not bool(connected_nonoverlapping(x[None],radii)[0]):return False
                try:return graph_key(infer_chemical_graph(x,c['numbers'],c['charge']))==key
                except (ValueError,RuntimeError):return False
            def evaluate(x):
                e,f=oracle.evaluate(torch.stack([x,-x]));raw_queries.append(dict(positions=x.clone(),raw_energy_eV=e,raw_force_eV_A=f))
                return float(e.mean()),(f[0]-f[1])/2
            try:result=relax_endpoint(reference,evaluate,allowed,**spec['optimizer'])
            except Exception as exc:
                atomic_save(dict(condition=c,raw_queries=raw_queries,error=repr(exc),
                    attempted=oracle.requested_evaluations,acknowledged=oracle.evaluated),args.out/f'failed_c{slot}.pt')
                raise
            assert allowed(result['final']['positions'])
            count=oracle.evaluated-before;assert count==2*result['evaluations']==2*len(raw_queries)
            file=args.out/'trajectories'/f'c{slot}.pt'
            atomic_save(dict(condition=c,training_row=index,protocol_sha256=ph,result=result,raw_queries=raw_queries),file)
            initial,final=result['initial'],result['final']
            row=dict(condition=c,training_row=index,reference=reference,physical=final['positions'],
                initial_energy_eV=initial['energy_eV'],final_energy_eV=final['energy_eV'],
                initial_force_eV_A=initial['force_eV_A'],final_force_eV_A=final['force_eV_A'],
                converged=result['converged'],status=result['status'],accepted_steps=result['accepted_steps'],
                oracle_queries=count,trajectory_sha256=sha(file),graph_id=original['connectivity_smiles'])
            rows.append(row)
            progress=dict(complete=False,completed=len(rows),oracle_queries=oracle.evaluated,
                converged=sum(r['converged'] for r in rows),seconds=time.perf_counter()-start)
            write(args.out/'progress.json',progress)
            print(json.dumps(dict(composition=slot,n_atoms=c['n_atoms'],status=result['status'],steps=result['accepted_steps'],
                force_rms_before=float(initial['force_eV_A'].square().sum(-1).mean().sqrt()),
                force_rms_after=float(final['force_eV_A'].square().sum(-1).mean().sqrt()),queries=count)),flush=True)
        total=oracle.evaluated;assert total==oracle.requested_evaluations==sum(r['oracle_queries'] for r in rows)
    bank=args.out/'bank.pt';atomic_save(dict(protocol_sha256=ph,rows=rows,target=spec['target']),bank)
    before=[float(r['initial_force_eV_A'].square().sum(-1).mean().sqrt()) for r in rows]
    after=[float(r['final_force_eV_A'].square().sum(-1).mean().sqrt()) for r in rows]
    write(args.out/'complete.json',dict(complete=True,protocol_sha256=ph,bank_sha256=sha(bank),compositions=len(rows),
        oracle_queries=total,converged=sum(r['converged'] for r in rows),unchanged=sum(r['accepted_steps']==0 for r in rows),
        initial_median_force_rms=float(np.median(before)),final_median_force_rms=float(np.median(after)),
        decreased_force_compositions=sum(b<a for a,b in zip(before,after)),
        seconds=time.perf_counter()-start,all_selected_compositions_retained=True,target=spec['target']))


if __name__=='__main__':main()
