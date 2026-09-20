"""Compare learned H readouts with identity/radial controls on cached validation."""
import argparse,copy,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.hydrogen_completion import complete,detached_hydrogens
from scripts.research.probe_training_hydrogen_readout import project_detached_hydrogens
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def coordinate_hash(x):return hashlib.sha256(x.double().contiguous().numpy().tobytes()).hexdigest()


def readout(model,x,c,spec,method):
    if method=='base':return x.clone(),dict(network_calls=0,network_example_calls=0,changed=torch.zeros(len(x),dtype=torch.bool))
    if method=='radial':
        y,mask=project_detached_hydrogens(x,c['numbers'],spec['radial_contact'],spec['radial_target']);changed=mask.any(-1)
        y=y+x.mean(1,keepdim=True);y=torch.where(changed[:,None,None],y,x)
        return y,dict(network_calls=0,network_example_calls=0,changed=changed)
    mode='molecule' if method.startswith('molecule') else 'atom';start=.5 if method.endswith('half') else 0.
    device=next(model.parameters()).device;z=torch.tensor(c['numbers'],device=device)[None].expand(len(x),-1)
    torch.testing.assert_close(x.float().double(),x,atol=0,rtol=0)
    pieces=[];infos=[]
    for begin in range(0,len(x),spec['evaluation_batch']):
        y,info=complete(model,x[begin:begin+spec['evaluation_batch']].to(device).float(),z[begin:begin+spec['evaluation_batch']],spec,
            mode=mode,start_time=start,steps=spec['decoder_steps'],velocity_cap=spec['velocity_cap_A'])
        pieces.append(y.cpu().double());infos.append({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in info.items()})
    y=torch.cat(pieces);info=dict(changed=torch.cat([v['changed'] for v in infos]),active_atoms=torch.cat([v['active_atoms'] for v in infos]),
        network_calls=sum(v['network_calls'] for v in infos),network_example_calls=sum(v['network_example_calls'] for v in infos))
    torch.testing.assert_close(y[~info['changed']],x[~info['changed']],atol=0,rtol=0)
    return y,info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','training','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,required=True);p.add_argument('--parents',type=Path);a=p.parse_args();torch.set_num_threads(2);root=a.project.resolve();a.out=a.out.resolve();a.training=a.training.resolve()
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);si=a.seed_index;assert spec['frozen'];a.out.mkdir(parents=True,exist_ok=False)
    assert 0<=si<len(spec['seeds'])
    if a.parents:
        manifest=json.loads(a.parents.read_text());assert manifest['complete'] and manifest['protocol_sha256']==ph and manifest['seed_index']==si
        input_sources=manifest['sources'];conditions=spec['test_rows']
    else:input_sources=spec['validation_sources'][si];conditions=spec['validation_rows']
    counts={v['samples_per_condition'] for v in input_sources.values()};assert len(counts)==1;count=counts.pop();n_conditions=len(conditions)
    trained=a.training/f's{si}';done=json.loads((trained/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==spec.get('training_protocol_sha256',ph) and done['steps']==spec['steps']
    file=trained/f'step_{spec["steps"]}.pt';checkpoint_hash=sha(file);assert checkpoint_hash==done['checkpoint_sha256'];saved=torch.load(file,map_location='cpu',weights_only=False)
    model=base.initialize(saved['network_spec'],'cuda').eval();model.load_state_dict(saved['ema_state_dict'],strict=True);model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==spec['parameter_count'];model_hash=base.state_hash(model)
    binary=Path(spec['xtb_binary']);assert sha(binary)==spec['xtb_binary_sha256'];raw=a.out/'raw';raw.mkdir();xp=a.out/'xtb';xp.mkdir()
    parents={};rows=[];coordinates={};calculations={};requests=[];assignments=[];generation=[];head_calls=0;example_calls=0;tick=time.perf_counter()
    for name in ['fm','gaga']:
        src=input_sources[name];rfile=root/src['generation'];pfile=root/src['physical'];physical_hash=sha(pfile)
        assert sha(rfile)==src['generation_sha256'] and physical_hash==src['physical_sha256']
        report=json.loads(rfile.read_text());physical=json.loads(pfile.read_text());physical_rows={(v['condition_index'],v['sample_index']):v for v in physical['rows'] if v['method']==src['method']}
        native=[v for v in report['rows'] if v['method']==src['method']];assert len(native)==n_conditions
        for i,entry in enumerate(native):
            sourcefile=rfile.parent/entry['file'];assert sha(sourcefile)==entry['sha256'];record=torch.load(sourcefile,map_location='cpu',weights_only=False)
            c=record['condition'];x=record['positions'];assert c==conditions[i] and len(x)==count and record['strength']==spec['parent_strength']
            parents[name,i]=dict(file=str(sourcefile.relative_to(root)),sha256=sha(sourcefile));parent_graph=np.array([v['graph_supported'] for v in entry['records']])
            # Seed the physical cache with the original immutable calculations.
            for j,pos in enumerate(x):
                digest=coordinate_hash(pos);key=(name,i,j,digest);r=physical_rows[i,j]
                d=pfile.parent/'details'/r['task_id'];assert sha(d/'input.xyz')==r['input_xyz_sha256']
                np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in (d/'input.xyz').read_text().splitlines()[2:]],pos,atol=1e-12,rtol=0)
                calculations[key]=dict(reused_parent=True,result=r,details=str(d.relative_to(root)),source_results=str(pfile.relative_to(root)),source_results_sha256=physical_hash)
            for method in spec['methods']:
                label=name+'_'+method;start=time.perf_counter();y,info=readout(model,x,c,spec,method);torch.cuda.synchronize();seconds=time.perf_counter()-start
                if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite readout')
                # Protect every graph-valid parent; changing a previously invalid
                # output is not a certificate that its repaired geometry is valid.
                torch.testing.assert_close(y[parent_graph],x[parent_graph],atol=0,rtol=0)
                heavy=torch.tensor(c['numbers'])!=1
                torch.testing.assert_close((y[:,heavy]-y[:,heavy].mean(1,keepdim=True)),(x[:,heavy]-x[:,heavy].mean(1,keepdim=True)),atol=3e-6,rtol=1e-6)
                assessment=assess(y,c,list(range(count)))
                assert all(not b or v['graph_supported'] for b,v in zip(parent_graph,assessment['records']))
                out=raw/f'{label}_c{i}.pt';atomic_save(dict(positions=y,condition=c,method=label,protocol_sha256=ph,parent=parents[name,i],
                    decoder_state_sha256=model_hash,decoder_checkpoint_sha256=checkpoint_hash,decoder_info=info,decoder_seconds=seconds),out)
                item=dict(method=label,condition_index=i,file=str(out.relative_to(a.out)),sha256=sha(out),**assessment);generation.append(item)
                head_calls+=info['network_calls'];example_calls+=info['network_example_calls']
                for j,pos in enumerate(y):
                    digest=coordinate_hash(pos);key=(name,i,j,digest)
                    if key not in calculations:
                        task=dict(task_id=f'{label}_c{i}_s{j}',method=label,replica=si,parent_id=i*count+j,inversion_check=False,
                            positions=pos.tolist(),condition_index=i,sample_index=j)
                        calculations[key]=None;requests.append((key,task,c))
                    assignments.append(dict(method=label,condition_index=i,sample_index=j,coordinate_sha256=digest,key=key,
                        graph=assessment['records'][j]['graph_supported']))
                print(json.dumps(dict(family=name,condition=i,method=method,graph=assessment['graph_supported'],changed=int(info['changed'].sum()))),flush=True)
    write(a.out/'generation.json',dict(complete=True,protocol_sha256=ph,decoder_checkpoint_sha256=checkpoint_hash,decoder_state_sha256=model_hash,rows=generation,
        new_parent_trajectories=0,new_derived_outputs=2*n_conditions*count*(len(spec['methods'])-1),decoder_network_calls=head_calls,decoder_network_example_calls=example_calls,
        parents_manifest_sha256=sha(a.parents) if a.parents else None))
    write(xp/'tasks.json',dict(protocol_sha256=ph,tasks=[dict(key=k,task=t,condition=c) for k,t,c in requests]))
    settings=dict(accuracy=.1,timeout_seconds=90,restraint_eV_A2=0.)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(run_task,t,c,binary,xp,settings):key for key,t,c in requests}
        for n,future in enumerate(as_completed(futures),1):
            key=futures[future];r=future.result();calculations[key]=dict(reused_parent=False,result=r,details=str((xp/'details'/r['task_id']).relative_to(root)))
            if n%64==0:print(json.dumps(dict(phase='new_gfn2',completed=n,requested=len(requests))),flush=True)
    for item in assignments:
        value=calculations[tuple(item.pop('key'))];r=value['result'];force=float(np.sqrt(np.square(r['force_eV_A']).sum(-1).mean())) if r['success'] else None
        rows.append(dict(**item,physical=value,rms_force=force,joint=bool(item['graph'] and r['success'] and force<=5)))
    summary={}
    for method in sorted({r['method'] for r in rows}):
        values=[v for v in rows if v['method']==method];summary[method]=dict(attempted=len(values),graph=sum(v['graph'] for v in values),joint=sum(v['joint'] for v in values),
            gfn2_failures=sum(not v['physical']['result']['success'] for v in values))
    write(a.out/'physical.json',dict(complete=True,protocol_sha256=ph,generation_sha256=sha(a.out/'generation.json'),tasks_sha256=sha(xp/'tasks.json'),rows=rows,summary=summary,
        new_gfn2_attempts=len(requests),reused_parent_readout_records=sum(v['physical']['reused_parent'] for v in rows)))
    assert base.state_hash(model)==model_hash
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,seed_index=si,decoder_checkpoint_sha256=checkpoint_hash,summary=summary,
        generation_sha256=sha(a.out/'generation.json'),physical_sha256=sha(a.out/'physical.json'),new_parent_trajectories=0,
        new_derived_outputs=2*n_conditions*count*(len(spec['methods'])-1),new_gfn2_attempts=len(requests),decoder_network_calls=head_calls,
        decoder_network_example_calls=example_calls,new_esen_queries=0,new_optimizer_steps=0,seconds=time.perf_counter()-tick))


if __name__=='__main__':main()
