#!/usr/bin/env python3
"""Reparse every raw GFN2 result and compare all predefined paired endpoints."""
import argparse
import json
from pathlib import Path
import torch
from ase.data import chemical_symbols
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.evaluate_chemical_policy import sha
from scripts.research.summarize_fresh_reuse import bootstrap_mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','tasks','run','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.project
    pp=root/'research/evidence/fresh_primary_xtb_protocol_v1.json';protocol=json.loads(pp.read_text())
    data=json.loads(args.tasks.read_text());report=json.loads((args.run/'results.json').read_text())
    assert data['complete'] and report['complete'] and report['protocol_sha256']==data['protocol_sha256']==sha(pp)
    assert report['tasks_sha256']==sha(args.tasks) and report['xtb_binary_sha256']==protocol['xtb_binary_sha256']
    assert report['condition']==data['condition'] and '6.7.1' in report['version_output']
    tasks={t['task_id']:t for t in data['tasks']};rows={r['task_id']:r for r in report['rows']}
    assert len(tasks)==len(rows)==report['attempted']==report['requested_attempts']==protocol['maximum_single_point_attempts']
    assert set(tasks)==set(rows)
    condition=data['condition'];symbols=[chemical_symbols[z] for z in condition['numbers']]
    for key,task in tasks.items():
        row=rows[key];work=args.run/'details'/key
        assert sha(work/'input.xyz')==row['input_xyz_sha256']
        lines=(work/'input.xyz').read_text().splitlines();assert int(lines[0])==len(symbols)
        assert [line.split()[0] for line in lines[2:]]==symbols
        xyz=torch.tensor([[float(v) for v in line.split()[1:]] for line in lines[2:]],dtype=torch.float64)
        torch.testing.assert_close(xyz,torch.tensor(task['positions'],dtype=torch.float64),atol=1e-12,rtol=0)
        for file in ['stdout','stderr']:assert sha(work/(file+'.txt'))==row[file+'_sha256']
        gradient=(work/'gradient').read_text() if (work/'gradient').exists() else ''
        assert (sha(work/'gradient') if gradient else None)==row['gradient_sha256']
        assert row['original_charge']==condition['charge'] and row['original_spin_multiplicity']==condition['spin_multiplicity']
        assert row['uhf']==condition['spin_multiplicity']-1
        if row['returncode'] is None:
            assert not row['success'] and row['failure']=='single_point_timeout'
        else:
            parsed=parse_singlepoint((work/'stdout.txt').read_text(),(work/'stderr.txt').read_text(),row['returncode'],gradient,len(symbols))
            for k,v in parsed.items():assert row[k]==v
        if row['success']:
            expected=row['energy_eV']+protocol['restraint_eV_A2']/2*sum(v*v for x in task['positions'] for v in x)
            assert abs(row['restrained_potential_eV']-expected)<1e-10
    inversions=[]
    for task in tasks.values():
        if not task['inversion_check']:continue
        a=rows[task['task_id']];b=rows[task['mirrored_task_id']]
        entry=dict(task_id=task['task_id'],successful_pair=a['success'] and b['success'])
        if entry['successful_pair']:
            entry.update(energy_error_eV=abs(a['energy_eV']-b['energy_eV']),
                force_error_eV_A=float((torch.tensor(a['force_eV_A'])+torch.tensor(b['force_eV_A'])).abs().max()))
        entry['passed']=entry['successful_pair'] and entry['energy_error_eV']<=protocol['energy_inversion_tolerance_eV'] and entry['force_error_eV_A']<=protocol['force_inversion_tolerance_eV_A']
        inversions.append(entry)
    failures=[{k:r.get(k) for k in ['task_id','method','replica','parent_id','failure']} for r in rows.values() if not r['success']]
    endpoint_rows=[r for r in rows.values() if not r['inversion_check'] and r['method']!='warm']
    assert len(endpoint_rows)==288 and len(inversions)==4
    available=all(r['success'] for r in endpoint_rows) and all(r['passed'] for r in inversions)
    result=dict(complete=True,protocol_sha256=sha(pp),tasks_sha256=sha(args.tasks),results_sha256=sha(args.run/'results.json'),
        raw_logs_reparsed=True,all_input_coordinates_and_electronic_states_verified=True,
        attempted=len(rows),successful=sum(r['success'] for r in rows.values()),failures=failures,inversions=inversions,
        primary_energy_comparison_available=available,new_esen_queries=0,new_xtb_queries_in_audit=0,
        scientific_submission_ready=False,scope=protocol['scope'])
    if available:
        for field in ['energy_eV','restrained_potential_eV']:
            diffs=[[rows[f'learned_vector_s{rep}_p{parent}'][field]-rows[f'site_s{rep}_p{parent}'][field]
                for parent in data['parent_ids']] for rep in [0,1]]
            result['gfn2_learned_minus_site_'+field]=bootstrap_mean(diffs,torch.Generator().manual_seed(protocol['bootstrap_seed']),protocol['bootstrap_replicates'])
            result['gfn2_'+field+'_replica_means']=[sum(x)/len(x) for x in diffs]
        # Keep the sampled oracle's energy and restraint contributions distinct.
        for field in ['esen_even_energy_eV','esen_potential_eV']:
            diffs=[[tasks[f'learned_vector_s{rep}_p{parent}'][field]-tasks[f'site_s{rep}_p{parent}'][field]
                for parent in data['parent_ids']] for rep in [0,1]]
            result[field+'_paired_mean_difference']=sum(sum(x) for x in diffs)/144
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['failures','inversions']}))


if __name__=='__main__':main()
