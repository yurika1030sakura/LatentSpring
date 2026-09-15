#!/usr/bin/env python3
"""Reparse every GFN2 result and evaluate frozen cross-potential confirmation gates."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ase.data import chemical_symbols
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.audit_source_utility import intervals
from scripts.research.audit_source_sc_energy import masked_intervals
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','run','generation_audit','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 prior=json.loads(a.generation_audit.read_text());assert prior['complete'];arrayfile=a.generation_audit.with_name(prior['arrays_file']);assert sha(arrayfile)==prior['arrays_sha256'];arrays=np.load(arrayfile);methods=prior['methods'];graph=arrays['graph'];shape=graph.shape
 e=np.full(shape,np.nan);f=np.full(shape,np.nan);success=np.zeros(shape,bool);summaries={};artifacts=[];checked=0;inversions=[]
 panelpath=a.project/'research/evidence/fresh_physics_panel_v1.json';assert sha(panelpath)==prior['panel_sha256'];panel=json.loads(panelpath.read_text())['rows']
 for seed in [0,1]:
  path=a.project/f'research/evidence/fresh_physics_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path);root=a.run/f's{seed}/xtb';resultfile=root/'results.json';r=json.loads(resultfile.read_text());tasksfile=root/'tasks.json';assert r['complete'] and r['protocol_sha256']==ph and r['tasks_sha256']==sha(tasksfile) and r['xtb_binary_sha256']==spec['xtb_binary_sha256']
  tasks=json.loads(tasksfile.read_text());mapping={x['task']['task_id']:x for x in tasks['tasks']};assert len(mapping)==len(r['rows'])==r['attempted']==spec['xtb_attempts_per_seed']==3120
  cache={};groot=Path(prior['generation_run'])/f's{seed}/study'
  for source in tasks['sources']:
   assert any(v['seed']==seed and v['method']==source['method'] and v['report_sha256']==source['report_sha256'] for v in prior['artifacts'])
   samplefile=groot/'evaluation'/f"{source['method']}_c{source['condition_index']}.pt";assert sha(samplefile)==source['sample_sha256'];cache[source['method'],source['condition_index']]=torch.load(samplefile,map_location='cpu',weights_only=False)['positions'].numpy()
  by_id={};failures={}
  for row in r['rows']:
   item=mapping[row['task_id']];task=item['task'];c=item['condition'];i=task['condition_index'];assert c['composition_hex']==panel[i]['composition_hex'] and c['numbers']==panel[i]['atomic_numbers'] and row['condition_index']==i and row['method']==task['method'] and row['sample_index']==task['sample_index']
   assert row['original_charge']==c['charge']==0 and row['original_spin_multiplicity']==c['spin_multiplicity']==1 and row['uhf']==0
   if task['method']=='reference':
    expected=torch.load(groot/'physical_eval'/f'reference_c{i}.pt',map_location='cpu',weights_only=False)['positions'].numpy();expected=-expected if task['inversion_check'] else expected
   else:expected=cache[task['method'],i][task['sample_index']]
   np.testing.assert_array_equal(np.asarray(task['positions']),expected)
   directory=root/'details'/row['task_id'];assert sha(directory/'input.xyz')==row['input_xyz_sha256'] and sha(directory/'stdout.txt')==row['stdout_sha256'] and sha(directory/'stderr.txt')==row['stderr_sha256']
   lines=(directory/'input.xyz').read_text().splitlines();assert int(lines[0])==c['n_atoms'] and len(lines)==c['n_atoms']+2;assert [line.split()[0] for line in lines[2:]]==[chemical_symbols[z] for z in c['numbers']];coords=np.array([[float(v) for v in line.split()[1:]] for line in lines[2:]]);np.testing.assert_allclose(coords,task['positions'],atol=1e-12,rtol=0)
   if row['returncode'] is None:assert not row['success'] and row['failure']=='single_point_timeout'
   else:
    gradient=(directory/'gradient').read_text() if (directory/'gradient').exists() else ''
    if row['gradient_sha256'] is not None:assert sha(directory/'gradient')==row['gradient_sha256']
    parsed=parse_singlepoint((directory/'stdout.txt').read_text(),(directory/'stderr.txt').read_text(),row['returncode'],gradient,c['n_atoms'])
    assert parsed['success']==row['success']
    if parsed['success']:
     assert parsed['energy_eV']==row['energy_eV'];np.testing.assert_allclose(parsed['force_eV_A'],row['force_eV_A'],atol=0,rtol=0)
    else:assert parsed['failure']==row['failure']
   assert '--opt' not in row['command'] and row['command'][-1]=='--grad' and row['command'][2:8]==['--gfn','2','--chrg','0','--uhf','0']
   checked+=1;by_id[row['task_id']]=row
   if row['method']=='reference':continue
   mi=methods.index(row['method']);j=row['sample_index'];success[seed,mi,i,j]=row['success']
   if row['success']:
    e[seed,mi,i,j]=row['energy_eV']/c['n_atoms'];force=np.asarray(row['force_eV_A']);f[seed,mi,i,j]=np.sqrt(np.mean(np.sum(force**2,axis=-1)))
   else:failures[row['failure']]=failures.get(row['failure'],0)+1
  for i in range(24):
   plus=by_id[f'reference_c{i}_plus'];minus=by_id[f'reference_c{i}_minus'];both=plus['success'] and minus['success'];item=dict(seed=seed,condition=i,successful=bool(both),passed=False)
   if both:
    item.update(energy_error_eV=abs(plus['energy_eV']-minus['energy_eV']),force_error_eV_A=float(np.max(np.abs(np.asarray(plus['force_eV_A'])+np.asarray(minus['force_eV_A'])))));item['passed']=item['energy_error_eV']<=spec['xtb_energy_inversion_tolerance_eV'] and item['force_error_eV_A']<=spec['xtb_force_inversion_tolerance_eV_A']
   inversions.append(item)
  summaries[seed]={}
  for mi,m in enumerate(methods):
   n_graph=int(graph[seed,mi].sum());n_ok=int((success[seed,mi]&graph[seed,mi]).sum());summaries[seed][m]=dict(attempts=int(success[seed,mi].size),successful=int(success[seed,mi].sum()),graph_supported=n_graph,successful_graph_supported=n_ok,graph_supported_success_fraction=n_ok/n_graph if n_graph else None)
  summaries[seed]['failure_reasons']=failures;artifacts.append(dict(seed=seed,result_sha256=sha(resultfile),tasks_sha256=sha(tasksfile),protocol_sha256=ph));print(json.dumps(dict(seed=seed,summary=summaries[seed])),flush=True)
 assert checked==6240;comparisons={};rng=np.random.default_rng(38091);strata=[i//8 for i in range(24)]
 for left,right in [('harmonic_tree','gaussian'),('escort_delta','harmonic_tree'),('work_delta','harmonic_tree'),('work_delta','escort_delta')]:
  li,ri=methods.index(left),methods.index(right);paired_ok=success[:,li]&success[:,ri];common=paired_ok&graph[:,li]&graph[:,ri];comp={}
  for name,array in [('energy_per_atom_eV',e),('force_rms_eV_A',f)]:
   diff=array[:,li]-array[:,ri];comp[name]=dict(common_success=masked_intervals(diff,paired_ok),common_graph_supported=masked_intervals(diff,common),common_graph_by_seed=[masked_intervals(diff[s:s+1],common[s:s+1],38092+s) for s in [0,1]])
  comp['comparable_cells_by_seed']=[int(common[s].any(-1).sum()) for s in [0,1]];comparisons[left+' minus '+right]=comp
 def energy_gate(group):
  metric=group['energy_per_atom_eV'];r=metric['common_graph_supported'];return bool(r['mean'] is not None and r['paired_draw95'][1]<0 and all(s['mean'] is not None and s['mean']<0 for s in metric['common_graph_by_seed']))
 key='escort_delta minus harmonic_tree';esen_gate=energy_gate(prior['comparisons'][key]);xtb_gate=energy_gate(comparisons[key]);coverage=all(summaries[s][m]['graph_supported_success_fraction'] is not None and summaries[s][m]['graph_supported_success_fraction']>=.9 for s in [0,1] for m in ['escort_delta','harmonic_tree']) and min(comparisons[key]['comparable_cells_by_seed'])>=19
 support=all(sum(prior['summary'][str(s)]['escort_delta'][metric]-prior['summary'][str(s)]['harmonic_tree'][metric] for s in [0,1])/1536>=-.02 for metric in ['graph_supported','distinct_connectivity'])
 inversion_pass=all(r['passed'] for r in inversions);arrayfile=a.out.with_suffix('.npz');np.savez_compressed(arrayfile,energy_per_atom=e,force_rms=f,success=success)
 write(a.out,dict(complete=True,summary=summaries,comparisons=comparisons,reference_inversions=inversions,reference_inversion_gate=inversion_pass,esen_primary_gate=esen_gate,xtb_primary_gate=xtb_gate,primary_coverage_gate=coverage,observed_validity_gate=support,cross_potential_confirmation_gate=bool(esen_gate and xtb_gate and coverage and support and inversion_pass),raw_xtb_attempts_reparsed=checked,artifacts=artifacts,arrays_file=arrayfile.name,arrays_sha256=sha(arrayfile),generation_audit_sha256=sha(a.generation_audit),scientific_submission_ready=False,
  scope='Independent approximate GFN2 evaluator at frozen coordinates. All SCC failures retained. Energy gates use paired common-graph and common-success subsets; they are selected-support estimands, not equilibrium expectations. New composition evidence uses the same two trained models. No DFT, SOTA, global Boltzmann or accepted-paper claim.'))


if __name__=='__main__':main()
