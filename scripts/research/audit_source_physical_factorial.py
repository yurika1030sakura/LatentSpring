#!/usr/bin/env python3
"""Replay the new factorial cell and combine it with the three audited cells."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.escorted_thermal_teacher import escorted_candidates,graph_key
from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def interval(value,indices):
    cells=value.mean(-1);boot=cells.mean(0)[indices].mean(1)
    return dict(mean=float(cells.mean()),by_seed=cells.mean(1).tolist(),composition_ci95=np.quantile(boot,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','out']:p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    old={}
    for kind,name in [('esen','fresh_physics_esen_audit_v1'),('xtb','fresh_physics_xtb_audit_v1')]:
        report=json.loads((args.project/f'research/evidence/{name}.json').read_text());path=args.project/f'research/evidence/{name}.npz'
        assert report['complete'] and sha(path)==report['arrays_sha256'];old[kind]=np.load(path)
    methods=['gaussian','harmonic_tree','gaussian_physical','harmonic_physical'];shape=(2,4,24,32)
    graph=np.zeros(shape,bool);energy={k:np.full(shape,np.nan) for k in old};force={k:np.full(shape,np.nan) for k in old};success={k:np.zeros(shape,bool) for k in old}
    for new_index,old_index in [(0,0),(1,1),(3,2)]:
        graph[:,new_index]=old['esen']['graph'][:,old_index]
        for kind in old:
            energy[kind][:,new_index]=old[kind]['energy_per_atom'][:,old_index]
            force[kind][:,new_index]=old[kind]['force_rms'][:,old_index]
            success[kind][:,new_index]=old[kind]['success'][:,old_index] if kind=='xtb' else True
    receipts=[];queries=0;fit_outputs=0;parsed_count=0
    for seed in [0,1]:
        spec_path=args.project/f'research/evidence/source_physical_factorial_s{seed}_v1.json';spec=json.loads(spec_path.read_text());ph=sha(spec_path)
        root=args.run/f's{seed}/study';done=json.loads((root/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
        panel=json.loads((args.project/spec['condition_manifest']).read_text())['rows'];pools=[];count=0
        for i in range(8):
            raw=torch.load(root/f'teacher/raw_c{i}.pt',map_location='cpu',weights_only=False);c=raw['condition'];x=raw['raw_positions']
            assert len(x)==32 and raw['role']=='FIT';fit_outputs+=len(x)
            repeated=assess(x,c,list(range(32)));assert repeated==raw['assessment']
            assert c['composition_hex'] not in {r['composition_hex'] for r in panel}
            file=root/f'teacher/refined_c{i}.pt'
            if not file.exists():assert not raw['graph_supported'].any();continue
            saved=torch.load(file,map_location='cpu',weights_only=False);record=saved['record'];n=len(record['anchor'])
            assert saved['raw_source_sha256']==sha(root/f'teacher/raw_c{i}.pt')
            torch.testing.assert_close(record['anchor'],x[raw['graph_supported']],atol=0,rtol=0)
            f=(record['anchor_raw_force_eV_A'][:n]-record['anchor_raw_force_eV_A'][n:])/2
            generated=escorted_candidates(record['anchor'],f,**spec['thermal_teacher'],seed=spec['teacher_seed']*100003+i)
            for key,value in zip(['source','proposal','sigma','shift'],generated):torch.testing.assert_close(record[key],value,atol=0,rtol=0)
            keys=[graph_key(infer_chemical_graph(a,c['numbers'],c['charge'])) for a in record['anchor']]
            valid=torch.zeros_like(record['valid'])
            for j in range(n):
                for k in range(8):
                    try:valid[j,k]=graph_key(infer_chemical_graph(record['proposal'][j,k],c['numbers'],c['charge']))==keys[j]
                    except (ValueError,RuntimeError):pass
            assert torch.equal(valid,record['valid']) and torch.equal(valid.any(1),record['eligible'])
            uniform=valid.double()/valid.sum(-1).clamp_min(1)[:,None]
            torch.testing.assert_close(saved['uniform_weights'],uniform[record['eligible']],atol=0,rtol=0)
            torch.testing.assert_close(saved['raw_positions'],record['anchor'][record['eligible']],atol=0,rtol=0)
            torch.testing.assert_close(saved['proposals'],record['proposal'][record['eligible']],atol=0,rtol=0)
            assert saved['oracle_queries']==2*n+2*int(valid.sum());count+=saved['oracle_queries']
            if len(saved['raw_positions']):pools.append(saved)
        assert count==done['teacher_raw_queries'];queries+=count+done['evaluation_raw_queries']
        data=torch.load(args.project/spec['data'],map_location='cpu',weights_only=False)['training']
        logs={m:[json.loads(l) for l in (root/m/'metrics.jsonl').read_text().splitlines()] for m in ['replay','escort']}
        for m in logs:
            assert len(logs[m])==1000;rng=torch.Generator().manual_seed(spec['training_seed']+17)
            for step,row in enumerate(logs[m],1):
                assert row['step']==step
                if step%2:
                    idx=int(torch.randint(len(data),(1,),generator=rng));expected=dict(kind='reference',row=idx);condition=data[idx]['condition']
                else:
                    idx=int(torch.randint(len(pools),(1,),generator=rng));entry=pools[idx];j=int(torch.randint(len(entry['raw_positions']),(1,),generator=rng));u=float(torch.rand((),generator=rng))
                    weights=entry['uniform_weights'][j];particle=-1 if m=='replay' else int(torch.searchsorted(weights.cumsum(0),u,right=True).clamp_max(7))
                    expected=dict(kind='generated_fit',pool=idx,row=j,particle=particle,uniform_draw=u);condition=entry['condition']
                assert row['selection']==expected and row['composition']==condition['composition_hex']
        checkpoints={m:torch.load(root/m/'last.ckpt',map_location='cpu',weights_only=False) for m in ['replay','escort','gaussian_physical']}
        warm=torch.load(args.project/spec['warm_checkpoint'],map_location='cpu',weights_only=False)
        for name,value in warm['state_dict'].items():
            if value.is_floating_point():expected=value+(checkpoints['escort']['state_dict'][name]-checkpoints['replay']['state_dict'][name])
            else:
                assert torch.equal(value,checkpoints['escort']['state_dict'][name]) and torch.equal(value,checkpoints['replay']['state_dict'][name]);expected=value
            torch.testing.assert_close(checkpoints['gaussian_physical']['state_dict'][name],expected,atol=0,rtol=0)
        report_path=root/'evaluation/gaussian_physical_results.json';report=json.loads(report_path.read_text())
        assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==sha(root/'gaussian_physical/last.ckpt')
        positions={}
        for i,row in enumerate(report['rows']):
            file=root/f'evaluation/gaussian_physical_c{i}.pt';assert sha(file)==row['sample_sha256']
            saved=torch.load(file,map_location='cpu',weights_only=False);c=saved['condition'];x=saved['positions'];positions[i]=x.numpy()
            assert c['composition_hex']==panel[i]['composition_hex'] and len(x)==32
            result=assess(x,c,list(range(32)))
            for key in result:assert result[key]==row[key]
            graph[seed,2,i]=[r['graph_supported'] for r in result['records']]
            quality=torch.load(root/f'physical_eval/gaussian_physical_c{i}.pt',map_location='cpu',weights_only=False)
            assert quality['source_sample_sha256']==sha(file);torch.testing.assert_close(quality['positions'],x,atol=0,rtol=0)
            ep=(quality['raw_energy_eV'][:32]+quality['raw_energy_eV'][32:])/2;fp=(quality['raw_force_eV_A'][:32]-quality['raw_force_eV_A'][32:])/2
            torch.testing.assert_close(quality['even_energy_eV'],ep,atol=0,rtol=0);torch.testing.assert_close(quality['even_force_eV_A'],fp,atol=0,rtol=0)
            assert torch.isfinite(ep).all() and torch.isfinite(fp).all()
            energy['esen'][seed,2,i]=ep.numpy()/c['n_atoms'];force['esen'][seed,2,i]=fp.square().sum(-1).mean(-1).sqrt().numpy();success['esen'][seed,2,i]=True
        xtb=args.run/f's{seed}/xtb';r=json.loads((xtb/'results.json').read_text());tasks=json.loads((xtb/'tasks.json').read_text())
        assert r['complete'] and r['protocol_sha256']==ph and r['attempted']==816 and r['tasks_sha256']==sha(xtb/'tasks.json')
        mapping={x['task']['task_id']:x for x in tasks['tasks']};reference={}
        for row in r['rows']:
            item=mapping[row['task_id']];task=item['task'];c=item['condition'];i=task['condition_index'];directory=xtb/'details'/row['task_id']
            assert '--opt' not in row['command'] and row['command'][-1]=='--grad'
            assert row['original_charge']==c['charge']==0 and row['original_spin_multiplicity']==c['spin_multiplicity']==1
            for name,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(directory/name)==row[key]
            coords=np.asarray([[float(v) for v in line.split()[1:]] for line in (directory/'input.xyz').read_text().splitlines()[2:]])
            np.testing.assert_allclose(coords,task['positions'],atol=1e-12,rtol=0)
            if task['method']!='reference':np.testing.assert_array_equal(task['positions'],positions[i][task['sample_index']])
            if row['returncode'] is not None:
                gradient=(directory/'gradient').read_text() if (directory/'gradient').exists() else ''
                if row['gradient_sha256'] is not None:assert sha(directory/'gradient')==row['gradient_sha256']
                parsed=parse_singlepoint((directory/'stdout.txt').read_text(),(directory/'stderr.txt').read_text(),row['returncode'],gradient,c['n_atoms'])
                assert parsed['success']==row['success']
                if parsed['success']:
                    assert parsed['energy_eV']==row['energy_eV'];np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A'])
            else:assert not row['success'] and row['failure']=='single_point_timeout'
            parsed_count+=1
            if task['method']=='reference':reference[row['task_id']]=row;continue
            j=task['sample_index'];success['xtb'][seed,2,i,j]=row['success']
            if row['success']:
                energy['xtb'][seed,2,i,j]=row['energy_eV']/c['n_atoms'];force['xtb'][seed,2,i,j]=np.sqrt(np.mean(np.sum(np.asarray(row['force_eV_A'])**2,axis=-1)))
        for i in range(24):
            a,b=reference[f'reference_c{i}_plus'],reference[f'reference_c{i}_minus'];assert a['success'] and b['success']
            assert abs(a['energy_eV']-b['energy_eV'])<=spec['xtb_energy_inversion_tolerance_eV']
            assert np.max(np.abs(np.asarray(a['force_eV_A'])+np.asarray(b['force_eV_A'])))<=spec['xtb_force_inversion_tolerance_eV_A']
        receipts.append(dict(seed=seed,protocol_sha256=ph,generation_report_sha256=sha(report_path),completion_sha256=sha(root/'complete.json'),xtb_result_sha256=sha(xtb/'results.json')))
    indices=np.random.default_rng(45213).integers(0,24,size=(20000,24));summary={};contrasts={}
    for kind in old:
        summary[kind]={};contrasts[kind]={}
        for j,m in enumerate(methods):
            good=graph[:,j]&success[kind][:,j];f=force[kind][:,j]
            summary[kind][m]=dict(graph_counts_by_seed=graph[:,j].sum((1,2)).tolist(),attempts_per_seed=768,
                valid_force_quantiles=np.quantile(f[good],[.1,.5,.9]).tolist() if good.any() else None,
                force_yield=[dict(threshold=t,**interval(good&(f<=t),indices)) for t in [1,2,5,10,20,50,100]])
        for name,left,right in [('physical_in_gaussian',2,0),('physical_in_harmonic',3,1),('source_without_physics',1,0),('source_with_physics',3,2)]:
            a=graph[:,left]&success[kind][:,left]&(force[kind][:,left]<=5)
            b=graph[:,right]&success[kind][:,right]&(force[kind][:,right]<=5)
            contrasts[kind][name]=interval(a.astype(float)-b.astype(float),indices)
    target=args.out.with_suffix('.npz');np.savez_compressed(target,graph=graph,**{f'{k}_energy':v for k,v in energy.items()},**{f'{k}_force':v for k,v in force.items()},**{f'{k}_success':v for k,v in success.items()})
    write(args.out,dict(complete=True,methods=methods,summary=summary,joint_yield5_contrasts=contrasts,receipts=receipts,
        arrays_file=target.name,arrays_sha256=sha(target),new_training_steps=4000,new_fit_outputs=fit_outputs,new_evaluation_outputs=1536,
        new_esen_queries=queries,new_xtb_attempts=parsed_count,teacher_geometry_and_selection_replayed=True,parameter_arithmetic_replayed=True,
        raw_graph_and_energy_readouts_replayed=True,scope='Fixed follow-up on the previously evaluated24-composition panel; three original cells reused. Report actual source-specific teacher support and cost; no fresh-composition or five-training-seed claim.'))
    print(json.dumps(dict(complete=True,contrasts=contrasts,new_esen_queries=queries,new_xtb_attempts=parsed_count)),flush=True)


if __name__=='__main__':main()
