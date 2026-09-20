"""Replay conditional H readouts, including unchanged physical-score reuse."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.evaluate_hydrogen_completion import readout,coordinate_hash
from scripts.research.audit_generator_output_support import assess
from scripts.research.run_matched_generators import batches,write
from scripts.research.train_electronic_fm import sha
from scripts.research.confirm_gaga_feedback import bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--training',type=Path);p.add_argument('--stage',choices=['select','confirm'],default='select')
    a=p.parse_args();root=a.project.resolve();run=a.run.resolve();torch.set_num_threads(2);spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    assert spec['frozen'] and not a.out.exists();datafile=root/spec['data'];assert sha(datafile)==spec['data_sha256'];data=torch.load(datafile,map_location='cpu',weights_only=False)
    perseed=[];provenance={};new_queries=0;decoder_examples=0;fits={};parent_queries=0
    confirming=a.stage=='confirm';training_root=a.training.resolve() if a.training else run/'training'
    conditions=spec['test_rows'] if confirming else spec['validation_rows'];n_conditions=len(conditions)
    training_ph=spec.get('training_protocol_sha256',ph)
    for si in [0,1]:
        training=training_root/f's{si}';trained=json.loads((training/'complete.json').read_text());assert trained['complete'] and trained['protocol_sha256']==training_ph and trained['steps']==10000
        checkpoint=training/'step_10000.pt';assert sha(checkpoint)==trained['checkpoint_sha256'];saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==training_ph and saved['seed']==spec['seeds'][si] and saved['step']==10000
        config=saved['network_spec'];assert config['upstream_args']==spec['network_spec']['upstream_args'] and config['atomic_numbers']==spec['network_spec']['atomic_numbers'] and config['initialization_seed']==spec['seeds'][si]
        model=base.initialize(config,'cpu').eval();initial=base.state_hash(model);info=json.loads((training/'initialization.json').read_text())
        assert initial==info['initial_state_sha256']==saved['initial_state_sha256']
        expected=batches(data['training'],spec['steps'],spec['batch_size'],spec['batch_seeds'][si]);actual=np.load(training/'batch_indices.npy');np.testing.assert_array_equal(actual,expected)
        assert sha(training/'batch_indices.npy')==info['schedule_sha256'];logs=[json.loads(line) for line in (training/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in logs]==[1]+list(range(100,10001,100)) and all(np.isfinite([r['loss'],r['gradient_norm']]).all() for r in logs)
        for n,pv in model.named_parameters():
            if not pv.requires_grad:
                torch.testing.assert_close(saved['state_dict'][n],pv,atol=0,rtol=0);torch.testing.assert_close(saved['ema_state_dict'][n],pv,atol=0,rtol=0)
        for state in saved['optimizer_state_dict']['state'].values():assert int(state['step'])==10000
        model.load_state_dict(saved['ema_state_dict'],strict=True);model.requires_grad_(False);model_hash=base.state_hash(model)
        assert sum(pv.numel() for pv in model.parameters())==153790
        if confirming:
            outer=run/f's{si}';outer_done=json.loads((outer/'complete.json').read_text());manifest=outer/'parents.json'
            assert outer_done['complete'] and outer_done['protocol_sha256']==ph and sha(manifest)==outer_done['parents_manifest_sha256']
            inputs=json.loads(manifest.read_text());assert inputs['complete'] and inputs['protocol_sha256']==ph
            input_sources=inputs['sources'];folder=outer/'readouts';parent_queries+=outer_done['new_parent_trajectories']
            assert sha(folder/'complete.json')==outer_done['readout_complete_sha256']
        else:folder=run/'validation'/f's{si}';input_sources=spec['validation_sources'][si]
        counts={v['samples_per_condition'] for v in input_sources.values()};assert len(counts)==1;count=counts.pop()
        done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['decoder_checkpoint_sha256']==sha(checkpoint)
        gfile=folder/'generation.json';pfile=folder/'physical.json';assert sha(gfile)==done['generation_sha256'] and sha(pfile)==done['physical_sha256']
        generation=json.loads(gfile.read_text());physical=json.loads(pfile.read_text());assert generation['complete'] and physical['complete'] and generation['decoder_state_sha256']==model_hash
        sources={};parents={};records={};positions={};calls=0;example_calls=0
        for name in ['fm','gaga']:
            origin=input_sources[name];f=root/origin['generation'];pf=root/origin['physical']
            assert sha(f)==origin['generation_sha256'] and sha(pf)==origin['physical_sha256'];sources[name]=json.loads(pf.read_text())
            parent_report=json.loads(f.read_text());rows=[r for r in parent_report['rows'] if r['method']==origin['method']];assert len(rows)==n_conditions
            if confirming:assert parent_report['protocol_sha256']==ph and parent_report['model_state_sha256']==spec['physical_heads'][str(si)][name]['model_state_sha256']
            for i,r in enumerate(rows):
                src=f.parent/r['file'];assert sha(src)==r['sha256'];parents[name,i]=torch.load(src,map_location='cpu',weights_only=False)
        for row in generation['rows']:
            label=row['method'];name,method=label.split('_',1);i=row['condition_index'];parent=parents[name,i];x=parent['positions'];c=parent['condition']
            f=folder/row['file'];assert sha(f)==row['sha256'];v=torch.load(f,map_location='cpu',weights_only=False);y=v['positions']
            assert v['protocol_sha256']==ph and v['decoder_checkpoint_sha256']==sha(checkpoint) and v['decoder_state_sha256']==model_hash and v['condition']==c
            assert sha(root/v['parent']['file'])==v['parent']['sha256']
            expected,counts=readout(model,x,c,spec,method)
            torch.testing.assert_close(y,expected,atol=3e-5 if method not in ['base','radial'] else 0,rtol=1e-5 if method not in ['base','radial'] else 0)
            for key in ['network_calls','network_example_calls']:assert counts[key]==v['decoder_info'][key]
            torch.testing.assert_close(counts['changed'],v['decoder_info']['changed'],atol=0,rtol=0)
            calls+=counts['network_calls'];example_calls+=counts['network_example_calls']
            quality=assess(y,c,list(range(count)))
            for key,value in quality.items():assert value==row[key]
            old=assess(x,c,list(range(count)))
            for j,r in enumerate(old['records']):
                if r['graph_supported']:torch.testing.assert_close(y[j],x[j],atol=0,rtol=0);assert quality['records'][j]['graph_supported']
            records[label,i]=quality;positions[label,i]=y
        assert len(generation['rows'])==2*n_conditions*len(spec['methods'])
        assert calls==generation['decoder_network_calls']==done['decoder_network_calls'] and example_calls==generation['decoder_network_example_calls']==done['decoder_network_example_calls']
        observed=set();seen_physical={};arrays={name+'_'+m:dict(graph=np.zeros((n_conditions,count),bool),joint=np.zeros((n_conditions,count),bool),
            energy=np.full((n_conditions,count),np.nan),success=np.zeros((n_conditions,count),bool),coordinate_sha256=np.full((n_conditions,count),'',dtype='U64')) for name in ['fm','gaga'] for m in spec['methods']}
        requests=json.loads((folder/'xtb/tasks.json').read_text());assert sha(folder/'xtb/tasks.json')==physical['tasks_sha256'];tasks={r['task']['task_id']:r for r in requests['tasks']}
        new_ids=set()
        for row in physical['rows']:
            label,i,j=row['method'],row['condition_index'],row['sample_index'];key=(label,i,j);assert key not in observed;observed.add(key)
            x=positions[label,i][j];assert coordinate_hash(x)==row['coordinate_sha256'];v=row['physical'];r=v['result'];d=root/v['details'];name=label.split('_',1)[0];c=conditions[i]
            if v['reused_parent']:
                f=root/v['source_results'];assert str(f)==str(root/input_sources[name]['physical']) and v['source_results_sha256']==input_sources[name]['physical_sha256']
                old=next(z for z in sources[name]['rows'] if z['task_id']==r['task_id']);assert old==r
                torch.testing.assert_close(x,parents[name,i]['positions'][j],atol=0,rtol=0)
            else:
                new_ids.add(r['task_id']);task=tasks[r['task_id']]['task'];np.testing.assert_array_equal(task['positions'],x)
            physical_key=str(d)
            if physical_key not in seen_physical:
                for filename,keyname in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(d/filename)==r[keyname]
                if r['gradient_sha256']:assert sha(d/'gradient')==r['gradient_sha256']
                assert r['command'][-1]=='--grad' and '--opt' not in r['command'] and r['original_charge']==c['charge'] and r['original_spin_multiplicity']==c['spin_multiplicity']
                if r['returncode'] is not None:
                    gradient=(d/'gradient').read_text() if (d/'gradient').exists() else '';parsed=parse_singlepoint((d/'stdout.txt').read_text(),(d/'stderr.txt').read_text(),r['returncode'],gradient,c['n_atoms'])
                    assert parsed['success']==r['success']
                    if r['success']:np.testing.assert_array_equal(parsed['force_eV_A'],r['force_eV_A']);assert parsed['energy_eV']==r['energy_eV']
                seen_physical[physical_key]=True
            np.testing.assert_allclose([[float(vv) for vv in line.split()[1:]] for line in (d/'input.xyz').read_text().splitlines()[2:]],x,atol=1e-12,rtol=0)
            graph=records[label,i]['records'][j]['graph_supported'];force=float(np.sqrt(np.square(r['force_eV_A']).sum(-1).mean())) if r['success'] else None
            joint=bool(graph and r['success'] and force<=5);assert graph==row['graph'] and joint==row['joint'] and force==row['rms_force']
            arrays[label]['graph'][i,j]=graph;arrays[label]['joint'][i,j]=joint
            arrays[label]['coordinate_sha256'][i,j]=row['coordinate_sha256'];arrays[label]['success'][i,j]=r['success']
            if r['success']:arrays[label]['energy'][i,j]=r['energy_eV']/c['n_atoms']
        assert len(observed)==2*n_conditions*count*len(spec['methods']) and len(new_ids)==len(tasks)==done['new_gfn2_attempts']==physical['new_gfn2_attempts']
        new_queries+=len(tasks);decoder_examples+=example_calls;perseed.append(arrays)
        provenance[str(gfile.relative_to(root))]=sha(gfile);provenance[str(pfile.relative_to(root))]=sha(pfile)
        fits[str(si)]=dict(checkpoint=str(checkpoint.relative_to(root)),sha256=sha(checkpoint),parameter_count=153790,steps=10000)
    summary={}
    for label in perseed[0]:
        graph=np.stack([v[label]['graph'] for v in perseed]);joint=np.stack([v[label]['joint'] for v in perseed])
        summary[label]=dict(attempted=int(joint.size),graph_rate=float(graph.mean()),joint_rate=float(joint.mean()),graph_by_seed=graph.mean((1,2)).tolist(),joint_by_seed=joint.mean((1,2)).tolist())
    if confirming:
        labels=list(perseed[0]);packed={key:np.stack([[r[m][key] for m in labels] for r in perseed]) for key in ['graph','joint','energy','success','coordinate_sha256']}
        contrasts={};rng=np.random.default_rng(63081)
        for name in ['fm','gaga']:
            left=labels.index(name+'_molecule_start0')
            for method in ['base','radial','atom_start0']:
                right=labels.index(name+'_'+method);same=packed['coordinate_sha256'][:,left]==packed['coordinate_sha256'][:,right]
                success=packed['success'][:,left]&packed['success'][:,right];difference=np.where(same,0.,packed['energy'][:,left]-packed['energy'][:,right]);missing=(~same)&(~success)
                key=name+'_learned_minus_'+method
                energy=dict(complete=not bool(missing.any()),missing_changed_pairs=int(missing.sum()),identity_zero_pairs=int(same.sum()),failed_identity_cancellations=int((same&~success).sum()))
                if not missing.any():energy.update(bootstrap(difference.mean(-1),rng,20000))
                contrasts[key]=dict(energy=energy,joint=bootstrap(packed['joint'][:,left].mean(-1)-packed['joint'][:,right].mean(-1),rng,20000),
                    graph=bootstrap(packed['graph'][:,left].mean(-1)-packed['graph'][:,right].mean(-1),rng,20000))
        for gaga in ['radial','molecule_start0']:
            li=labels.index('fm_molecule_start0');ri=labels.index('gaga_'+gaga)
            contrasts['fm_learned_minus_gaga_'+gaga]=dict(joint=bootstrap(packed['joint'][:,li].mean(-1)-packed['joint'][:,ri].mean(-1),rng,20000),
                graph=bootstrap(packed['graph'][:,li].mean(-1)-packed['graph'][:,ri].mean(-1),rng,20000))
        primary=contrasts['fm_learned_minus_radial'];e=primary['energy'];gate=bool(e['complete'] and e['ci95'][1]<0 and max(e['by_seed'])<0 and primary['graph']['mean']>=0 and primary['joint']['mean']>=0)
        file=a.out.with_suffix('.npz');np.savez_compressed(file,**packed)
        write(a.out,dict(complete=True,protocol_sha256=ph,summary=summary,contrasts=contrasts,physical_readout_gate=gate,original_yield_screen_passed=False,
            provenance=provenance,arrays_sha256=sha(file),methods=labels,new_parent_trajectories=parent_queries,new_derived_outputs=parent_queries*(len(spec['methods'])-1),
            new_gfn2_attempts=parent_queries+new_queries,decoder_network_example_calls=decoder_examples,new_optimizer_steps=0,new_esen_queries=0,scope=spec['scope']))
        print(json.dumps(dict(summary=summary,contrasts=contrasts,physical_readout_gate=gate)),flush=True);return
    selected={}
    for name in ['fm','gaga']:
        chosen=max(spec['methods'],key=lambda m:(summary[name+'_'+m]['joint_rate'],summary[name+'_'+m]['graph_rate'],-spec['methods'].index(m)))
        selected[name]=dict(method=chosen,**summary[name+'_'+chosen])
    left=selected['fm'];basevalue=summary['fm_base'];radial=summary['fm_radial']
    advance=bool(left['method'] not in ['base','radial'] and all(l>r for l,r in zip(left['joint_by_seed'],basevalue['joint_by_seed'])) and
        all(l>r for l,r in zip(left['joint_by_seed'],radial['joint_by_seed'])) and left['graph_rate']>=max(basevalue['graph_rate'],radial['graph_rate']))
    write(a.out,dict(complete=True,protocol_sha256=ph,selected=selected,advance=advance,summary=summary,fits=fits,provenance=provenance,
        new_parent_trajectories=0,new_derived_outputs=2560,new_gfn2_attempts=new_queries,new_optimizer_steps=20000,new_training_forward_examples=640000,
        decoder_network_example_calls=decoder_examples,new_esen_queries=0,fresh_test_outputs_used=False,
        scope='Both trained H flows replay on CPU within declared float tolerance. Original accepted coordinates and all reused physical records are checked. This is development selection, not a fresh-generator superiority claim. The learned decoder must improve both fits beyond the fixed geometric control.'))
    print(json.dumps(dict(selected=selected,advance=advance,summary=summary,new_gfn2_attempts=new_queries)),flush=True)


if __name__=='__main__':main()
