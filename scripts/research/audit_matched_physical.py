#!/usr/bin/env python3
"""Replay paired fits and both physical readouts of the matched-generator study."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ase.data import chemical_symbols
from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.escorted_thermal_teacher import escorted_candidates,graph_key
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.confirm_gaga_feedback import audit_report
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','out']:p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    labels=['distance_base','distance_physical','gaga_base','gaga_physical'];shape=(2,4,32,32)
    graph=np.zeros(shape,bool);energy={k:np.full(shape,np.nan) for k in ['esen','xtb']};force={k:np.full(shape,np.nan) for k in energy};success={k:np.zeros(shape,bool) for k in energy}
    sources=[];teacher_queries=0;esen_queries=0;xtb_attempts=0
    for seed in [0,1]:
        version='v2' if (args.run/'corrected_run.json').exists() else 'v1'
        pp=args.project/f'research/evidence/matched_physical_s{seed}_{version}.json';spec=json.loads(pp.read_text());ph=sha(pp)
        root=args.run/f's{seed}/study';done=json.loads((root/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
        panel=json.loads((args.project/spec['panel']).read_text())['test_rows'];data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training']
        teacher_conditions=[data[i]['condition'] for i in spec['teacher_rows']]
        for name,arm in spec['arms'].items():
            folder=root/name;parent=torch.load(args.project/arm['checkpoint'],map_location='cpu',weights_only=False)['ema_state_dict']
            assert sha(args.project/arm['checkpoint'])==arm['checkpoint_sha256']
            raw=audit_report(folder/'raw_teacher/fit_results.json',teacher_conditions,64);assert raw['model_state_sha256']==state_hash(parent)
            pools=[];queries=0
            for i,row in enumerate(raw['rows']):
                file=folder/f'teacher/c{i}.pt'
                if not file.exists():assert row['graph_supported']==0;continue
                saved=torch.load(file,map_location='cpu',weights_only=False);record=saved['record'];n=len(record['anchor']);c=saved['condition']
                source=folder/f'raw_teacher/fit_c{i}.pt';assert saved['raw_source_sha256']==sha(source)
                original=torch.load(source,map_location='cpu',weights_only=False);flags=torch.tensor([r['graph_supported'] for r in row['records']])
                torch.testing.assert_close(record['anchor'],original['positions'][flags],atol=0,rtol=0)
                f=(record['anchor_raw_force_eV_A'][:n]-record['anchor_raw_force_eV_A'][n:])/2
                expected=escorted_candidates(record['anchor'],f,**spec['thermal_teacher'],seed=spec['teacher_seed']*100003+i)
                for key,value in zip(['source','proposal','sigma','shift'],expected):torch.testing.assert_close(record[key],value,atol=0,rtol=0)
                keys=[graph_key(infer_chemical_graph(x,c['numbers'],c['charge'])) for x in record['anchor']];valid=torch.zeros_like(record['valid'])
                for j in range(n):
                    for k in range(8):
                        try:valid[j,k]=graph_key(infer_chemical_graph(record['proposal'][j,k],c['numbers'],c['charge']))==keys[j]
                        except (ValueError,RuntimeError):pass
                assert torch.equal(valid,record['valid']) and torch.equal(valid.any(1),record['eligible'])
                torch.testing.assert_close(saved['uniform_weights'],(valid.double()/valid.sum(-1).clamp_min(1)[:,None])[record['eligible']],atol=0,rtol=0)
                torch.testing.assert_close(saved['raw_positions'],record['anchor'][record['eligible']],atol=0,rtol=0)
                torch.testing.assert_close(saved['proposals'],record['proposal'][record['eligible']],atol=0,rtol=0)
                assert saved['oracle_queries']==2*n+2*int(valid.sum());queries+=saved['oracle_queries']
                if len(saved['raw_positions']):pools.append(saved)
            report=json.loads((folder/'teacher.json').read_text());assert report['complete'] and report['queries']==queries;teacher_queries+=queries
            states={}
            for role in ['replay','physical']:
                log=[json.loads(l) for l in (folder/role/'metrics.jsonl').read_text().splitlines()];assert len(log)==arm['student_steps']
                training=json.loads((folder/role/'training.json').read_text());assert training['complete'] and training['backbone_example_passes']==2000
                file=folder/role/'last.ckpt';assert sha(file)==training['checkpoint_sha256'];saved=torch.load(file,map_location='cpu',weights_only=False)
                assert saved['protocol_sha256']==spec.get('reused_arm_protocol_sha256',{}).get(name,ph);states[role]=saved['state_dict']
                torch.testing.assert_close(states[role]['gamma.gamma'],parent['gamma.gamma'],atol=0,rtol=0)
                rng=torch.Generator().manual_seed(spec['training_seed']+17)
                for step,row in enumerate(log,1):
                    if step%2:
                        index=int(torch.randint(len(data),(1,),generator=rng));label=dict(kind='reference',row=index);condition=data[index]['condition']
                    else:
                        index=int(torch.randint(len(pools),(1,),generator=rng));entry=pools[index];j=int(torch.randint(len(entry['raw_positions']),(1,),generator=rng));u=float(torch.rand((),generator=rng));w=entry['uniform_weights'][j]
                        particle=-1 if role=='replay' else int(torch.searchsorted(w.cumsum(0),u,right=True).clamp_max(7))
                        label=dict(kind='generated_fit',pool=index,row=j,uniform_draw=u,particle=particle);condition=entry['condition']
                    assert row['step']==step and row['selection']==label and row['composition']==condition['composition_hex']
            final=torch.load(folder/'paired.ckpt',map_location='cpu',weights_only=False);assert final['protocol_sha256']==spec.get('reused_arm_protocol_sha256',{}).get(name,ph)
            for key,value in parent.items():
                expected=value+(states['physical'][key]-states['replay'][key]) if value.is_floating_point() else value
                torch.testing.assert_close(final['state_dict'][key],expected,atol=0,rtol=0)
            gen=json.loads((root/f'evaluation/{name}_physical_results.json').read_text());assert gen['model_state_sha256']==state_hash(final['state_dict'])
            assert done['parent_state_sha256'][name]==state_hash(parent)
        positions={};quality=json.loads((root/'physical_eval/results.json').read_text());assert quality['complete'] and quality['protocol_sha256']==ph
        artifacts={(r['method'],r['condition_index']):r for r in quality['rows']}
        for mi,label in enumerate(labels):
            report_path=root/f'evaluation/{label}_results.json';report=audit_report(report_path,panel,32)
            for i,row in enumerate(report['rows']):
                file=root/f'evaluation/{label}_c{i}.pt';x=torch.load(file,map_location='cpu',weights_only=False)['positions'];positions[label,i]=x.numpy()
                graph[seed,mi,i]=[r['graph_supported'] for r in row['records']]
                reference=artifacts[label,i];qp=root/'physical_eval'/reference['artifact'];assert sha(qp)==reference['artifact_sha256'] and reference['source_sample_sha256']==sha(file)
                q=torch.load(qp,map_location='cpu',weights_only=False);torch.testing.assert_close(q['positions'],x,atol=0,rtol=0)
                e=(q['raw_energy_eV'][:32]+q['raw_energy_eV'][32:])/2;f=(q['raw_force_eV_A'][:32]-q['raw_force_eV_A'][32:])/2
                torch.testing.assert_close(q['even_energy_eV'],e,atol=0,rtol=0);torch.testing.assert_close(q['even_force_eV_A'],f,atol=0,rtol=0)
                assert torch.isfinite(e).all() and torch.isfinite(f).all()
                energy['esen'][seed,mi,i]=e.numpy()/panel[i]['n_atoms'];force['esen'][seed,mi,i]=f.square().sum(-1).mean(-1).sqrt().numpy();success['esen'][seed,mi,i]=True
            sources.append(dict(seed=seed,method=label,report_sha256=sha(report_path)))
        assert quality['queries']==done['evaluation_raw_queries']==8256;esen_queries+=quality.get('new_queries',quality['queries'])
        xtb=args.run/f's{seed}/xtb';r=json.loads((xtb/'results.json').read_text());assert r['complete'] and r['protocol_sha256']==ph and r['attempted']==4160 and r['tasks_sha256']==sha(xtb/'tasks.json')
        tasks=json.loads((xtb/'tasks.json').read_text());mapping={v['task']['task_id']:v for v in tasks['tasks']};references={}
        for row in r['rows']:
            item=mapping[row['task_id']];task=item['task'];c=item['condition'];i=task['condition_index'];directory=xtb/'details'/row['task_id']
            assert row['original_charge']==c['charge']==0 and row['original_spin_multiplicity']==c['spin_multiplicity']==1 and '--opt' not in row['command'] and row['command'][-1]=='--grad'
            for filename,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(directory/filename)==row[key]
            lines=(directory/'input.xyz').read_text().splitlines();assert int(lines[0])==len(c['numbers']);assert [l.split()[0] for l in lines[2:]]==[chemical_symbols[z] for z in c['numbers']]
            np.testing.assert_allclose([[float(v) for v in l.split()[1:]] for l in lines[2:]],task['positions'],atol=1e-12,rtol=0)
            if task['method']!='reference':np.testing.assert_array_equal(task['positions'],positions[task['method'],i][task['sample_index']])
            if row['returncode'] is not None:
                gradient=(directory/'gradient').read_text() if (directory/'gradient').exists() else ''
                if row['gradient_sha256'] is not None:assert sha(directory/'gradient')==row['gradient_sha256']
                parsed=parse_singlepoint((directory/'stdout.txt').read_text(),(directory/'stderr.txt').read_text(),row['returncode'],gradient,c['n_atoms']);assert parsed['success']==row['success']
                if parsed['success']:assert parsed['energy_eV']==row['energy_eV'];np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A'])
            else:assert not row['success'] and row['failure']=='single_point_timeout'
            if not spec.get('reuse_xtb_run') or task['method']=='gaga_physical':xtb_attempts+=1
            if task['method']=='reference':references[row['task_id']]=row;continue
            mi=labels.index(task['method']);j=task['sample_index'];success['xtb'][seed,mi,i,j]=row['success']
            if row['success']:energy['xtb'][seed,mi,i,j]=row['energy_eV']/c['n_atoms'];force['xtb'][seed,mi,i,j]=np.sqrt(np.mean(np.sum(np.asarray(row['force_eV_A'])**2,axis=-1)))
        for i in range(32):
            a,b=references[f'reference_c{i}_plus'],references[f'reference_c{i}_minus'];assert a['success'] and b['success']
            assert abs(a['energy_eV']-b['energy_eV'])<=spec['reference_inversion_energy_tolerance_eV']
            assert np.max(np.abs(np.asarray(a['force_eV_A'])+np.asarray(b['force_eV_A'])))<=spec['reference_inversion_force_tolerance_eV_A']
        print(json.dumps(dict(seed=seed,audited=True)),flush=True)
    indices=np.random.default_rng(46301).integers(0,32,size=(20000,32));summary={};contrasts={}
    def metric(values):
        cell=values.mean(-1);draws=cell.mean(0)[indices].mean(1)
        return dict(mean=float(cell.mean()),by_seed=cell.mean(1).tolist(),ci95=np.quantile(draws,[.025,.975]).tolist())
    for kind in energy:
        summary[kind]={}
        for mi,label in enumerate(labels):
            valid=graph[:,mi]&success[kind][:,mi]
            summary[kind][label]=dict(graph_counts_by_seed=graph[:,mi].sum((1,2)).tolist(),attempts_per_seed=1024,
                valid_force_quantiles=np.quantile(force[kind][:,mi][valid],[.1,.5,.9]).tolist() if valid.any() else None,
                force_yield=[dict(threshold=t,**metric(valid&(force[kind][:,mi]<=t))) for t in [1,2,5,10,20,50,100]])
        yield5=graph&success[kind]&(force[kind]<=5);contrasts[kind]={}
        for name,left,right in [('adapted_fm_minus_adapted_gaga',1,3),('physical_in_fm',1,0),('physical_in_gaga',3,2)]:contrasts[kind][name]=metric(yield5[:,left].astype(float)-yield5[:,right].astype(float))
    gate=all(min(contrasts[k]['adapted_fm_minus_adapted_gaga']['by_seed'])>0 and contrasts[k]['adapted_fm_minus_adapted_gaga']['ci95'][0]>0 for k in energy)
    file=args.out.with_suffix('.npz');np.savez_compressed(file,graph=graph,**{f'{k}_energy':v for k,v in energy.items()},**{f'{k}_force':v for k,v in force.items()},**{f'{k}_success':v for k,v in success.items()})
    corrected=(args.run/'corrected_run.json').exists()
    write(args.out,dict(complete=True,primary_gate=gate,methods=labels,summary=summary,contrasts=contrasts,sources=sources,arrays_file=file.name,arrays_sha256=sha(file),
        fixed_noise_schedules_verified=True,new_fit_outputs=0 if corrected else 2048,new_evaluation_outputs=2048 if corrected else 4096,new_optimizer_steps=8000 if corrected else 12000,new_teacher_queries=0 if corrected else teacher_queries,new_evaluation_esen_queries=esen_queries,new_xtb_attempts=xtb_attempts,
        raw_graphs_teacher_selection_and_parameter_updates_replayed=True,physical_readouts_reparsed=True,reserved_outcomes_queried=False,
        scope='Fixed follow-up on the already evaluated32-composition panel. Each parent has960000 retained backbone training passes; each physical/replay student adds2000. All generators use128 inference calls. No wall-time or global-equilibrium equality is claimed.'))
    print(json.dumps(dict(complete=True,primary_gate=gate,contrasts=contrasts)),flush=True)


if __name__=='__main__':main()
