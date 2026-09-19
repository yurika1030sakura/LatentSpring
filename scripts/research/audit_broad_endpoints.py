"""Replay TRAIN relaxation and audit matched raw generator/physical results."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.physical_endpoint_relaxation import relax_endpoint
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii
from cfm_mol.geometric_domain import connected_nonoverlapping
from cfm_mol.escorted_thermal_teacher import graph_key
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.weighted_endpoints import EndpointDraw
from cfm_mol.weighted_endpoint_fm import checkpoint_source
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.audit_generator_output_support import assess
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def teacher_audit(project,run,spec,ph):
    folder=run/'teacher';completion=json.loads((folder/'complete.json').read_text())
    assert completion['complete'] and completion['protocol_sha256']==ph and sha(folder/'bank.pt')==completion['bank_sha256']
    bank=torch.load(folder/'bank.pt',map_location='cpu',weights_only=False)
    data=torch.load(project/spec['data'],map_location='cpu',weights_only=False)['training']
    assert sha(project/spec['data'])==spec['data_sha256'];assert len(bank['rows'])==128
    queries=0;converged=0
    for slot,(row,index) in enumerate(zip(bank['rows'],spec['training_rows'])):
        source=data[index];c=source['condition'];x=source['positions'].double();x=x-x.mean(0)
        assert row['condition']==c and row['training_row']==index
        torch.testing.assert_close(row['reference'],x,atol=0,rtol=0)
        p=folder/'trajectories'/f'c{slot}.pt';assert sha(p)==row['trajectory_sha256']
        saved=torch.load(p,map_location='cpu',weights_only=False);result=saved['result'];raw=saved['raw_queries']
        assert saved['protocol_sha256']==ph and len(raw)==result['evaluations'] and row['oracle_queries']==2*len(raw)
        original=graph_key(infer_chemical_graph(x,c['numbers'],c['charge']));radii=covalent_radii(c['numbers'])
        def allowed(y):
            if not bool(connected_nonoverlapping(y[None],radii)[0]):return False
            try:return graph_key(infer_chemical_graph(y,c['numbers'],c['charge']))==original
            except (ValueError,RuntimeError):return False
        counter=[0]
        def recorded(y):
            q=raw[counter[0]];counter[0]+=1
            torch.testing.assert_close(y,q['positions'],atol=1e-10,rtol=0)
            return float(q['raw_energy_eV'].mean()),(q['raw_force_eV_A'][0]-q['raw_force_eV_A'][1])/2
        replay=relax_endpoint(x,recorded,allowed,**spec['optimizer'])
        assert counter[0]==len(raw) and replay['status']==row['status'] and replay['converged']==row['converged']
        assert replay['accepted_query_indices']==result['accepted_query_indices']
        torch.testing.assert_close(replay['final']['positions'],row['physical'],atol=1e-10,rtol=0)
        torch.testing.assert_close(replay['initial']['force_eV_A'],row['initial_force_eV_A'],atol=0,rtol=0)
        torch.testing.assert_close(replay['final']['force_eV_A'],row['final_force_eV_A'],atol=0,rtol=0)
        assert replay['initial']['energy_eV']==row['initial_energy_eV'] and replay['final']['energy_eV']==row['final_energy_eV']
        assert allowed(row['physical']);converged+=replay['converged'];queries+=row['oracle_queries']
    assert queries==completion['oracle_queries'] and converged==completion['converged']
    record=dict(complete=True,protocol_sha256=ph,compositions=128,oracle_queries=queries,
        converged=converged,all_optimization_queries_and_decisions_replayed=True,
        all_selected_references_retained=True,bank_sha256=sha(folder/'bank.pt'),
        scope=spec['target'])
    write(run/'teacher_audit.json',record)
    return bank,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--teacher-only',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen']
    bank,teacher=teacher_audit(a.project,a.run,spec,ph)
    if a.teacher_only:
        write(a.out,teacher);print(json.dumps(teacher),flush=True);return
    parent=torch.load(a.project/spec['checkpoint'],map_location='cpu',weights_only=False);prior=prior_from_checkpoint(parent)
    methods=spec['methods'];shape=(2,3,16,16)
    graph=np.zeros(shape,bool);success=np.zeros(shape,bool);force=np.full(shape,np.inf);energy=np.full(shape,np.nan)
    sources=[];attempts=0
    for si,seed in enumerate(spec['training_seeds']):
        folder=a.run/f's{si}';done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['seed']==seed
        raw_report=json.loads((folder/'audit.json').read_text());assert sha(folder/'audit.json')==done['raw_audit_sha256']
        for method in ['reference_ft','relaxed_ft']:
            fit=folder/method;state=torch.load(fit/'last.ckpt',map_location='cpu',weights_only=False)
            assert state['global_step']==spec['training_steps'] and state['training_seed']==seed
            target=state['research_protocol']['direct_endpoint_training'];assert target['protocol_sha256']==ph and target['bank_sha256']==teacher['bank_sha256'] and target['method']==method
            assert not target['thermal_distribution_claim'] and target['terminal_noise_std_A']==0
            log=[json.loads(l) for l in (fit/'metrics.jsonl').read_text().splitlines()];assert len(log)==2000
            rng=np.random.default_rng(seed)
            for step,row in enumerate(log,1):
                index=int(rng.integers(128));c=bank['rows'][index]['condition'];rng.normal(size=(c['n_atoms'],3))
                assert row['step']==step and row['bank_row']==index and row['training_row']==spec['training_rows'][index]
                assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
        raw_positions={}
        for mi,method in enumerate(methods):
            directory=folder/(method+'_raw');report=json.loads((directory/'generation.json').read_text())
            ckpt=folder/('base.ckpt' if method=='base' else method+'/last.ckpt')
            assert report['complete'] and report['checkpoint_sha256']==sha(ckpt)
            assert report['primitive_network_calls_per_attempt']==128 and not report['geometry_refinement'] and not report['energy_filter'] and report['terminal_noise_A']==0
            rows=[r for r in raw_report['rows'] if r['arm']==method];assert len(rows)==16
            for i,row in enumerate(report['rows']):
                c=row['condition'];expected=spec['test_rows'][i]
                for key in ['atomic_numbers','charge','spin_multiplicity']:assert c[key]==expected[key]
                file=directory/row['file'];assert sha(file)==row['sha256']==rows[i]['raw_sha256']
                with np.load(file) as values:x=torch.from_numpy(values['raw_positions']);initial=values['source_positions'].copy()
                result=assess(x,c,list(range(len(x))))
                for key in result:assert result[key]==rows[i][key]
                graph[si,mi,i]=[r['graph_supported'] for r in result['records']];raw_positions[method,i]=x.numpy()
                for k in [0,8]:
                    draws=[EndpointDraw('audit','none','none',c,np.zeros((c['n_atoms'],3))) for _ in range(8)]
                    start=checkpoint_source(prior,draws,spec['evaluation_seeds'][si]*1000003+i*100003+k)
                    np.testing.assert_array_equal(start.reshape(8,c['n_atoms'],3).numpy(),initial[k:k+8])
            sources.append(dict(seed=seed,method=method,generation_sha256=sha(directory/'generation.json')))
        xp=folder/'xtb';physical=json.loads((xp/'audit.json').read_text());assert physical['complete'] and sha(xp/'audit.json')==done['xtb_audit_sha256']
        assert sha(xp/'tasks.json')==physical['tasks_sha256'];tasks={r['task']['task_id']:r['task'] for r in json.loads((xp/'tasks.json').read_text())['tasks']}
        assert len(physical['rows'])==768;attempts+=len(physical['rows'])
        for row in physical['rows']:
            task=tasks[row['task_id']];mi=methods.index(task['method']);i,j=task['condition_index'],task['sample_index'];directory=xp/'details'/row['task_id']
            np.testing.assert_array_equal(task['positions'],raw_positions[task['method'],i][j])
            for name,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(directory/name)==row[key]
            np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in (directory/'input.xyz').read_text().splitlines()[2:]],task['positions'],atol=1e-12,rtol=0)
            assert row['command'][-1]=='--grad' and '--opt' not in row['command']
            if row['returncode'] is not None:
                gradient=(directory/'gradient').read_text() if (directory/'gradient').exists() else ''
                parsed=parse_singlepoint((directory/'stdout.txt').read_text(),(directory/'stderr.txt').read_text(),row['returncode'],gradient,len(task['positions']))
                assert parsed['success']==row['success']
                if row['success']:np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A']);assert parsed['energy_eV']==row['energy_eV']
            if row['success']:
                success[si,mi,i,j]=True;force[si,mi,i,j]=np.sqrt(np.square(row['force_eV_A']).sum(-1).mean());energy[si,mi,i,j]=row['energy_eV']/len(task['positions'])
    yield5=graph & success & (force<=5.);rng=np.random.default_rng(52081);contrasts={}
    for name,left,right in [('relaxed_minus_reference_ft',2,1),('relaxed_minus_base',2,0),('reference_ft_minus_base',1,0)]:
        contrasts[name]=bootstrap(yield5[:,left].mean(-1)-yield5[:,right].mean(-1),rng,20000)
    summary={}
    for mi,method in enumerate(methods):
        valid=graph[:,mi]&success[:,mi]
        summary[method]=dict(graph_rate=float(graph[:,mi].mean()),graph_counts_by_seed=graph[:,mi].sum((1,2)).tolist(),
            median_valid_force=float(np.median(force[:,mi][valid])) if valid.any() else None,
            force_yields=[dict(threshold=t,mean=float((valid & (force[:,mi]<=t)).mean()),by_seed=(valid & (force[:,mi]<=t)).mean((1,2)).tolist()) for t in spec['thresholds']])
    arrayfile=a.out.with_suffix('.npz');np.savez_compressed(arrayfile,graph=graph,success=success,force=force,energy=energy)
    primary=contrasts['relaxed_minus_reference_ft'];baseline=contrasts['relaxed_minus_base']
    result=dict(complete=True,protocol_sha256=ph,teacher_audit_sha256=sha(a.run/'teacher_audit.json'),summary=summary,contrasts=contrasts,
        primary_gate=bool(primary['ci95'][0]>0 and min(primary['by_seed'])>0 and baseline['mean']>0),
        arrays_file=arrayfile.name,arrays_sha256=sha(arrayfile),sources=sources,
        new_neural_outputs=1536,new_optimizer_steps=8000,new_esen_queries=teacher['oracle_queries'],new_gfn2_attempts=attempts,
        scope=spec['scope'],interval_scope='Composition bootstrap conditional on two fine-tuning runs from one shared pretrained parent.')
    write(a.out,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
