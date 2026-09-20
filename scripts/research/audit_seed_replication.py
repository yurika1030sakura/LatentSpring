"""Audit a complete independent-fit comparison, including all unchanged-score reuse."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.audit_matched_connection import audit_fit
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_hydrogen_completion import readout,coordinate_hash
from scripts.research.run_matched_physical import make_model
from scripts.research.run_matched_generators import batches,write
from scripts.research.train_electronic_fm import sha


def verify_physics(root,record,folder,coordinates,c,cache):
    key=str(folder)
    if key not in cache:
        for name,digest in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(folder/name)==record[digest]
        if record['gradient_sha256']:assert sha(folder/'gradient')==record['gradient_sha256']
        assert record['command'][-1]=='--grad' and '--opt' not in record['command']
        assert record['original_charge']==c['charge'] and record['original_spin_multiplicity']==c['spin_multiplicity']
        if record['returncode'] is not None:
            gradient=(folder/'gradient').read_text() if (folder/'gradient').exists() else ''
            parsed=parse_singlepoint((folder/'stdout.txt').read_text(),(folder/'stderr.txt').read_text(),record['returncode'],gradient,c['n_atoms'])
            assert parsed['success']==record['success']
            if parsed['success']:np.testing.assert_array_equal(parsed['force_eV_A'],record['force_eV_A']);assert parsed['energy_eV']==record['energy_eV']
        cache.add(key)
    np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in (folder/'input.xyz').read_text().splitlines()[2:]],coordinates,atol=1e-12,rtol=0)
    force=float(np.sqrt(np.square(record['force_eV_A']).sum(-1).mean())) if record['success'] else np.inf
    return bool(record['success']),float(record['energy_eV'])/c['n_atoms'] if record['success'] else np.nan,force


def audit_parent_training(root,campaign,fit_index,family,data):
    arm=campaign['fits'][fit_index]['parents'][family];file=root/arm['checkpoint'];saved=torch.load(file,map_location='cpu',weights_only=False)
    if fit_index<2:assert sha(file)==arm['checkpoint_sha256'];return
    folder=file.parent;done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['checkpoint_sha256']==sha(file)
    spec=arm['spec'];assert saved['protocol_sha256']==arm['protocol_sha256']==done['protocol_sha256'] and saved['global_step']==spec['training_steps']
    expected=batches(data['training'],30000,32,spec['batch_seed'])[:spec['training_steps']];np.testing.assert_array_equal(np.load(folder/'batch_indices.npy'),expected)
    model=make_model(arm,saved['ema_state_dict'],device='cpu');assert base.state_hash(model)==done['ema_state_sha256']
    init=base.initialize(spec,'cpu');assert base.state_hash(init)==saved['initial_state_sha256']==done['initial_state_sha256']
    for n,p in init.named_parameters():
        if not p.requires_grad:torch.testing.assert_close(saved['state_dict'][n],p,atol=0,rtol=0);torch.testing.assert_close(saved['ema_state_dict'][n],p,atol=0,rtol=0)
    state=saved['optimizer_state_dict']['state'];assert state and all(int(v['step'])==spec['training_steps'] for v in state.values())
    logs=[json.loads(line) for f in sorted(folder.glob('metrics_attempt*.jsonl')) for line in f.read_text().splitlines()]
    assert logs[-1]['step']==spec['training_steps'] and all(np.isfinite([r['loss'],r['gradient_norm']]).all() for r in logs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=range(5),required=True);a=p.parse_args();root=a.project.resolve();folder=a.run.resolve();si=a.fit;torch.set_num_threads(2)
    assert not a.out.exists();campaign=json.loads(a.protocol.read_text());cp=sha(a.protocol);done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['campaign_sha256']==cp and done['fit']==si
    protocol=folder/'resolved_protocol.json';spec=json.loads(protocol.read_text());ph=sha(protocol);assert ph==done['resolved_protocol_sha256'];conditions=spec['test_rows'];assert len(conditions)==64
    data=torch.load(root/campaign['data'],map_location='cpu',weights_only=False);assert sha(root/campaign['data'])==campaign['data_sha256']
    keys={c['composition_hex'] for c in conditions};assert len(keys)==64 and not keys&{r['condition']['composition_hex'] for r in data['training']+data['validation']}
    parent_models={};heads={};provenance={};cache=set();positions={};graphs={};initial={};timing={}
    methods=[name+'_'+suffix for name in ['fm','gaga'] for suffix in ['parent','physical','radial','hydrogen']];shape=(8,64,16)
    arrays=dict(graph=np.zeros(shape,bool),success=np.zeros(shape,bool),force=np.full(shape,np.inf),energy=np.full(shape,np.nan),coordinate_hash=np.full(shape,'',dtype='U64'))
    for family in ['fm','gaga']:
        audit_parent_training(root,campaign,si,family,data)
        arm=spec['parents'][si][family];model=make_model(arm,torch.load(root/arm['checkpoint'],map_location='cpu',weights_only=False)['ema_state_dict'],device='cpu');parent_models[family]=base.state_hash(model)
        info=audit_fit(root,folder,spec,ph,si,family,fit_subdir='head') if si>=2 else done['heads'][family]
        assert sha(root/info['path'])==info['sha256'];heads[family]=info
        gf=folder/'parents'/family/'generation.json';report=json.loads(gf.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['model_state_sha256']==parent_models[family];provenance[str(gf.relative_to(root))]=sha(gf)
        assert len(report['rows'])==128
        if 'head_state_sha256' in info:assert report['head_state_sha256']==info['head_state_sha256']
        for row in report['rows']:
            label=row['method'];i=row['condition_index'];ai=int(label[-1]);file=gf.parent/row['file'];assert sha(file)==row['sha256'];saved=torch.load(file,map_location='cpu',weights_only=False);c=conditions[i];x=saved['positions']
            assert saved['condition']==c and saved['protocol_sha256']==ph and saved['strength']==[0.,4.][ai] and saved['seed']==campaign['evaluation_seeds'][si]
            assert saved['model_state_sha256']==parent_models[family] and saved['head_state_sha256']==report['head_state_sha256']
            assert saved['core_calls']==256 and saved['head_calls']==(128 if family=='fm' else 256)
            quality=assess(x,c,list(range(16)));assert all(row[k]==v for k,v in quality.items())
            positions[label,i]=x;graphs[label,i]=quality;initial[label,i]=saved['initial_positions']
            if ai:torch.testing.assert_close(initial[label,i],initial[family+'_a0',i],atol=0,rtol=0)
            timing.setdefault(label,[]).append(saved['generation_seconds'])
    xp=folder/'parent_xtb';pf=xp/'results.json';physical=json.loads(pf.read_text());assert physical['complete'] and physical['protocol_sha256']==ph and physical['new_gfn2_attempts']==4096
    tasksfile=xp/'tasks.json';assert sha(tasksfile)==physical['tasks_sha256'];tasks={r['task']['task_id']:r for r in json.loads(tasksfile.read_text())['tasks']};seen=set();parent_records={}
    for r in physical['rows']:
        task=tasks[r['task_id']];t=task['task'];i,j=t['condition_index'],t['sample_index'];label=t['method'];family=label.split('_')[0];ai=int(label[-1]);method=family+('_parent' if ai==0 else '_physical');mi=methods.index(method)
        assert (mi,i,j) not in seen;seen.add((mi,i,j));assert t['replica']==si and task['condition']==conditions[i];x=positions[label,i][j];np.testing.assert_array_equal(t['positions'],x)
        ok,energy,force=verify_physics(root,r,xp/'details'/r['task_id'],x,conditions[i],cache);graph=graphs[label,i]['records'][j]['graph_supported'];assert r['graph']==graph and r['joint']==bool(graph and ok and force<=5)
        for k,v in [('graph',graph),('success',ok),('energy',energy),('force',force),('coordinate_hash',coordinate_hash(x))]:arrays[k][mi,i,j]=v
        parent_records[r['task_id']]=r
    parent_physical_hash=sha(pf)
    assert len(seen)==4096;provenance[str(pf.relative_to(root))]=parent_physical_hash;assert parent_physical_hash==done['parent_results_sha256']
    hproto=folder/'readout_protocol.json';hs=json.loads(hproto.read_text());hph=sha(hproto);assert hph==done['readout_protocol_sha256']
    train=root/f'runs/seed_replication_v1/hydrogen_training/s{si}';tdone=json.loads((train/'complete.json').read_text());hf=train/'step_10000.pt';assert tdone['complete'] and sha(hf)==tdone['checkpoint_sha256'] and tdone['protocol_sha256']==hs['training_protocol_sha256']
    hydrogen_hash=sha(hf)
    hsave=torch.load(hf,map_location='cpu',weights_only=False);model=base.initialize(hsave['network_spec'],'cpu');hinit=base.state_hash(model);assert hinit==hsave['initial_state_sha256'];model.load_state_dict(hsave['ema_state_dict'],strict=True);model.eval().requires_grad_(False)
    assert hsave['seed']==campaign['fits'][si]['hydrogen_seed'] and hsave['step']==10000
    np.testing.assert_array_equal(np.load(train/'batch_indices.npy'),batches(data['training'],10000,32,campaign['h_batch_seeds'][si]))
    assert all(int(v['step'])==10000 for v in hsave['optimizer_state_dict']['state'].values())
    rd=folder/'readouts';gd=json.loads((rd/'generation.json').read_text());pd=json.loads((rd/'physical.json').read_text());hc=json.loads((rd/'complete.json').read_text());assert hc['complete'] and hc['protocol_sha256']==hph and sha(rd/'complete.json')==done['readout_complete_sha256']
    assert sha(rd/'generation.json')==hc['generation_sha256'] and sha(rd/'physical.json')==hc['physical_sha256']
    readouts={};head_calls=0;example_calls=0
    for row in gd['rows']:
        label=row['method'];family,method=label.split('_',1);i=row['condition_index'];f=rd/row['file'];assert sha(f)==row['sha256'];saved=torch.load(f,map_location='cpu',weights_only=False);x=positions[family+'_a1',i];c=conditions[i]
        assert saved['condition']==c and saved['protocol_sha256']==hph and saved['decoder_checkpoint_sha256']==hydrogen_hash
        expected,info=readout(model,x,c,hs,method);torch.testing.assert_close(saved['positions'],expected,atol=3e-5 if method=='molecule_start0' else 0,rtol=1e-5 if method=='molecule_start0' else 0)
        for key in ['changed','network_calls','network_example_calls']:
            if isinstance(info[key],torch.Tensor):torch.testing.assert_close(info[key],saved['decoder_info'][key],atol=0,rtol=0)
            else:assert info[key]==saved['decoder_info'][key]
        head_calls+=info['network_calls'];example_calls+=info['network_example_calls'];y=saved['positions'];quality=assess(y,c,list(range(16)));assert all(row[k]==v for k,v in quality.items())
        old=torch.tensor([r['graph_supported'] for r in graphs[family+'_a1',i]['records']]);torch.testing.assert_close(y[old],x[old],atol=0,rtol=0)
        readouts[label,i]=(y,quality)
    assert len(gd['rows'])==384 and head_calls==hc['decoder_network_calls'] and example_calls==hc['decoder_network_example_calls']
    seen=set();new_ids=set()
    for row in pd['rows']:
        label=row['method'];family,method=label.split('_',1);i,j=row['condition_index'],row['sample_index'];assert (label,i,j) not in seen;seen.add((label,i,j));x,quality=readouts[label,i];x=x[j]
        assert coordinate_hash(x)==row['coordinate_sha256'];r=row['physical']['result'];folder_physical=root/row['physical']['details']
        if row['physical']['reused_parent']:
            assert r==parent_records[r['task_id']] and row['physical']['source_results_sha256']==parent_physical_hash;torch.testing.assert_close(x,positions[family+'_a1',i][j],atol=0,rtol=0)
        else:new_ids.add(r['task_id'])
        ok,energy,force=verify_physics(root,r,folder_physical,x,conditions[i],cache);graph=quality['records'][j]['graph_supported'];assert row['graph']==graph and row['joint']==bool(graph and ok and force<=5)
        mi=methods.index(family+'_'+{'base':'physical','radial':'radial','molecule_start0':'hydrogen'}[method])
        if method=='base':assert arrays['coordinate_hash'][mi,i,j]==coordinate_hash(x)
        for k,v in [('graph',graph),('success',ok),('energy',energy),('force',force),('coordinate_hash',coordinate_hash(x))]:arrays[k][mi,i,j]=v
    assert len(seen)==6144 and len(new_ids)==pd['new_gfn2_attempts']==hc['new_gfn2_attempts'] and done['new_gfn2_attempts']==4096+len(new_ids)
    for f in [rd/'generation.json',rd/'physical.json',protocol,hproto,folder/'complete.json']:provenance[str(f.relative_to(root))]=sha(f)
    target=a.out.with_suffix('.npz');np.savez_compressed(target,**arrays);joint=arrays['graph']&arrays['success']&(arrays['force']<=5)
    summary={m:dict(attempted=1024,graph_rate=float(arrays['graph'][i].mean()),joint_rate=float(joint[i].mean()),failures=int((~arrays['success'][i]).sum())) for i,m in enumerate(methods)}
    write(a.out,dict(complete=True,campaign_sha256=cp,fit=si,methods=methods,summary=summary,heads=heads,provenance=provenance,arrays_sha256=sha(target),costs={k:v for k,v in done.items() if k.startswith('new_')},decoder_network_example_calls=example_calls,timing=timing,scope='Every parent/readout, physical parse and denominator audited. Learned H outputs replayed on CPU. New parent and head training checked without changing parameters.'))
    print(json.dumps(dict(fit=si,summary=summary)),flush=True)

if __name__=='__main__':main()
