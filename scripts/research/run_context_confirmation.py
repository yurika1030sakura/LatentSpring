"""Fit a size-matched control and compare frozen full generators on a common panel."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol.physical_connection import make_physical_connection
from scripts.research.audit_generator_output_support import assess
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);a=p.parse_args();torch.set_num_threads(2)
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);si=a.seed_index;seed=spec['training_seeds'][si];es=spec['evaluation_seeds'][si];item=spec['sources'][si]
    assert spec['frozen'];a.out.mkdir(parents=True,exist_ok=False);wp=a.project/spec['wide_protocol'];assert sha(wp)==spec['wide_protocol_sha256']
    subprocess.run([sys.executable,'-u','-m','scripts.research.connection_capacity','--project',str(a.project),'--protocol',str(wp),'--out',str(a.out/'wide_fit'),'--seed-index',str(si)],check=True)
    parentfile=a.project/spec['checkpoint'];assert sha(parentfile)==spec['checkpoint_sha256'];parent=torch.load(parentfile,map_location='cpu',weights_only=False)
    conditions=a.out/'conditions.json';write(conditions,dict(rows=spec['test_rows']));sampling=a.out/'sampling';sampling.mkdir();checkpoints={};rows=[]
    for method in spec['methods']:
        folder=a.out/(method+'_raw')
        if not method.startswith('gaga'):
            if method=='base':
                torch.manual_seed(seed);config=spec['physical_connection'];head=make_physical_connection(**config);weights=head.state_dict();origin='zero_initialized'
            else:
                source=a.out/'wide_fit/wide/step_20000.pt' if method=='pair_wide' else a.project/item['heads'][method]['path']
                expected_ph=spec['wide_protocol_sha256'] if method=='pair_wide' else item['heads'][method]['protocol_sha256']
                if method!='pair_wide':assert sha(source)==item['heads'][method]['sha256']
                saved=torch.load(source,map_location='cpu',weights_only=False);assert saved['protocol_sha256']==expected_ph and saved['seed']==seed and saved['step']==20000
                config=saved['configuration'];weights=saved['state_dict'];origin=sha(source)
            states={**parent['state_dict'],**{'vector_field.physical_connection.'+n:v for n,v in weights.items()}}
            for n,v in parent['state_dict'].items():assert torch.equal(states[n],v)
            target=dict(mode='physical_head_confirmation',teacher_temperature_K=300.,model_kT_eV=1.,terminal_noise_std_A=0.,thermal_distribution_claim=False,method=method,protocol_sha256=ph)
            recipe={**parent['research_protocol'],'physical_connection':config,'direct_endpoint_training':target};checkpoint=sampling/(method+'.ckpt')
            torch.save(dict(state_dict=states,research_protocol=recipe,source_prior=parent['source_prior']),checkpoint)
            checkpoints[method]=dict(file=str(checkpoint.relative_to(a.out)),sha256=sha(checkpoint),head_origin=origin,head_parameters=sum(v.numel() for k,v in weights.items() if k!='radii'))
            subprocess.run([sys.executable,'-u','-m','scripts.research.generate_weighted_minima','--checkpoint',str(checkpoint),'--config',str(a.project/spec['config']),
                '--conditions',str(conditions),'--out',str(folder),'--device','cuda','--samples','16','--midpoint-steps','32','--seed',str(es),'--allow-legacy-pickle'],check=True)
        else:
            from cfm_mol import matched_egnn as base
            from scripts.research.run_gaga_feedback import evaluate
            data=item['gaga'];path=a.project/data['protocol'];assert sha(path)==data['protocol_sha256'];cfg=json.loads(path.read_text());checkpoint=a.project/data['checkpoint'];assert sha(checkpoint)==data['checkpoint_sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['global_step']==30000
            model=base.initialize(cfg,'cuda');model.load_state_dict(state['ema_state_dict'],strict=True);model.eval().requires_grad_(False);calls=128 if method=='gaga' else 651
            cfg.update(evaluation_calls=calls,evaluation_batch=8,gaga_max_t=650);source=base.HarmonicSource(cfg['edge_log_width']);native=a.out/(method+'_native')
            counter=[0];hook=model.dynamics.egnn.register_forward_pre_hook(lambda m,args:counter.__setitem__(0,counter[0]+1))
            report=evaluate(model,source,cfg,None,spec['test_rows'],es,16,native,method);assert counter[0]==32*calls;hook.remove()
            folder.mkdir();generated=[]
            for i,row in enumerate(report['rows']):
                f=native/f'{method}_c{i}.pt';assert sha(f)==row['sample_sha256'];v=torch.load(f,map_location='cpu',weights_only=False);path=folder/f'raw_condition_{i:04d}.npz'
                np.savez_compressed(path,raw_positions=v['positions'].numpy(),source_positions=v['initial_positions'].numpy(),atomic_numbers=np.asarray(row['condition']['atomic_numbers']))
                generated.append(dict(condition=row['condition'],file=path.name,sha256=sha(path),attempted=16,seconds=v['generation_seconds'],native_file=str(f.relative_to(a.out)),native_sha256=sha(f)))
            write(folder/'generation.json',dict(complete=True,rows=generated,checkpoint_sha256=sha(checkpoint),model_state_sha256=base.state_hash(model),backbone_calls_per_attempt=calls,
                connection_calls_per_attempt=0,primitive_network_calls_per_attempt=calls,geometry_refinement=False,energy_filter=False,physical_queries=0,
                terminal_noise_A='native_GAGA_observation_noise',native_report_sha256=sha(native/(method+'_results.json'))))
            checkpoints[method]=dict(file=str(checkpoint),sha256=sha(checkpoint),head_parameters=0);del model,state;torch.cuda.empty_cache()
        report=json.loads((folder/'generation.json').read_text());assert report['complete'] and not report['geometry_refinement'] and not report['energy_filter']
        for row in report['rows']:
            f=folder/row['file'];assert sha(f)==row['sha256']
            with np.load(f) as v:x=torch.from_numpy(v['raw_positions'])
            rows.append(dict(arm=method,condition=row['condition'],raw_sha256=row['sha256'],**assess(x,row['condition'],list(range(len(x))))))
    write(a.out/'audit.json',dict(complete=True,protocol_sha256=ph,seed=seed,rows=rows))
    xp=a.out/'physical_protocol.json';write(xp,dict(frozen=True,native_protocol_sha256=ph,methods=spec['methods'],replica=seed,condition_count=16,samples_per_condition=16,
        xtb_binary=spec['xtb_binary'],xtb_binary_sha256=spec['xtb_binary_sha256'],xtb=spec['xtb'],thresholds=spec['thresholds'],scope=spec['scope']))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,'-u','-m','scripts.research.score_native_minima','--project',str(a.project),'--run',str(a.out),'--protocol',str(xp),'--out',str(a.out/'xtb')],env=env,check=True)
    physical=json.loads((a.out/'xtb/audit.json').read_text());assert physical['complete']
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,seed=seed,sampling_checkpoints=checkpoints,raw_audit_sha256=sha(a.out/'audit.json'),
        xtb_audit_sha256=sha(a.out/'xtb/audit.json'),summary=physical['summary'],new_neural_outputs=1536,new_optimizer_steps=20000,new_gfn2_attempts=1536,new_esen_queries=0))


if __name__=='__main__':main()
