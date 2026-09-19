"""Evaluate the already-published physical-update weights on the common panel.

The noise-free output is captured from the same neural trajectory; the published
0.025-A terminal-noise sample is produced by the original evaluation function.
Derived output records and actual neural work are counted separately.
"""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from flowmol.model_utils.load import read_config_file
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import restore_model,evaluate
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def prepare(root,out):
    assert not out.exists();common=root/'research/evidence/context_confirmation_v1.json';s=json.loads(common.read_text());sources=[]
    for si in [0,1]:
        p=root/f'research/evidence/fresh_physics_s{si}_v1.json';spec=json.loads(p.read_text());ref=spec['checkpoints']['escort_delta'];assert sha(root/ref['path'])==ref['sha256']
        sources.append(dict(protocol=str(p.relative_to(root)),protocol_sha256=sha(p),checkpoint=ref,terminal_noise_A=spec['terminal_noise_std_A']))
    spec=dict(s,format='published_model_confirmation_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        common_protocol=str(common.relative_to(root)),common_protocol_sha256=sha(common),published_sources=sources,
        methods=['published','published_zero'],conditions=list(range(16)),evaluation_batch=8,
        scope='Previously designated paper method evaluated on the SAME common panel. The additional zero-terminal-noise control reuses its exact neural trajectories. No model selection from this panel. Full-system architecture/data/history differences remain explicit; not a shared-backbone GAGA comparison.',
        timing='Added before reading any common-panel GFN2 comparison, to include the already-published model rather than only new heads. No prior frozen experiment is rewritten.')
    write(out,spec)


def run(root,protocol,out,si):
    s=json.loads(protocol.read_text());ph=sha(protocol);source=s['published_sources'][si];ref=source['checkpoint'];path=root/ref['path'];assert sha(path)==ref['sha256']
    cfgfile=root/s['config'];assert sha(cfgfile)==s['config_sha256'];cfg=read_config_file(cfgfile);cfg['mol_fm'].pop('bgfm',None)
    assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    state=torch.load(path,map_location='cpu',weights_only=False);model=restore_model(cfg,state);prior=prior_from_checkpoint(state);torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False);native=out/'native';native.mkdir();manifest=out/'conditions.json';write(manifest,dict(rows=s['test_rows']))
    p=dict(s,evaluation_seed=s['evaluation_seeds'][si],terminal_noise_std_A=source['terminal_noise_A']);captured=[];counter=[0]
    def capture(model,graph,nbi,uem,**kwargs):
        y=sample_clamped_flow(model,graph,nbi,uem,**kwargs);batch=graph.batch_size;n=len(y)//batch
        z=y.reshape(batch,n,3).double();z=z-z.mean(1,keepdim=True);captured.append(z.cpu());return y
    hook=model.vector_field.register_forward_pre_hook(lambda m,args:counter.__setitem__(0,counter[0]+1))
    # Each observed field call executes the checkpoint's two-pass backbone.
    assert state['research_protocol']['geometry_self_conditioning']==dict(zero_init=True,deep_supervision=True,edge_feedback='clamped')
    evaluate(model,prior,'published',cfg,p,ph,native,ref['sha256'],manifest,sampler=capture);hook.remove()
    assert len(captured)==32 and counter[0]==32*64
    report=json.loads((native/'published_results.json').read_text());assert len(report['rows'])==16;raw=[]
    for method in s['methods'][:2]:
        directory=out/(method+'_raw');directory.mkdir();records=[]
        for i,row in enumerate(report['rows']):
            f=native/f'published_c{i}.pt';assert sha(f)==row['sample_sha256'];value=torch.load(f,map_location='cpu',weights_only=False);zero=torch.cat(captured[2*i:2*i+2])
            x=value['positions'] if method=='published' else zero;file=directory/f'raw_condition_{i:04d}.npz'
            np.savez_compressed(file,raw_positions=x.numpy(),source_positions=value['initial_positions'].numpy(),
                atomic_numbers=np.asarray(value['condition']['atomic_numbers']),zero_noise_positions=zero.numpy(),
                terminal_displacement=(value['positions']-zero).numpy())
            records.append(dict(condition=value['condition'],file=file.name,sha256=sha(file),native_sha256=sha(f),native_file=str(f.relative_to(out))))
            raw.append(dict(arm=method,condition=value['condition'],raw_sha256=sha(file),**assess(x,value['condition'],list(range(16)))))
        write(directory/'generation.json',dict(complete=True,rows=records,checkpoint_sha256=ref['sha256'],protocol_sha256=ph,
            backbone_calls_per_attempt=128,head_calls_per_attempt=0,terminal_noise_A=source['terminal_noise_A'] if method=='published' else 0.,
            geometry_refinement=False,energy_filter=False,physical_queries=0,shared_trajectories=True))
    # The short head is a prespecified earlier prefix of the already-tested long fit.
    short=s['short_sources'][si];shortfile=root/short['path'];assert sha(shortfile)==short['sha256']
    h=torch.load(shortfile,map_location='cpu',weights_only=False);assert h['step']==2000 and h['seed']==s['training_seeds'][si]
    parentfile=root/s['checkpoint'];assert sha(parentfile)==s['checkpoint_sha256'];parent=torch.load(parentfile,map_location='cpu',weights_only=False)
    states={**parent['state_dict'],**{'vector_field.physical_connection.'+k:v for k,v in h['state_dict'].items()}}
    target=dict(mode='short_head_ablation',teacher_temperature_K=300.,model_kT_eV=1.,terminal_noise_std_A=0.,protocol_sha256=ph)
    checkpoint=out/'pair_short.ckpt';torch.save(dict(state_dict=states,source_prior=parent['source_prior'],research_protocol={**parent['research_protocol'],
        'physical_connection':h['configuration'],'direct_endpoint_training':target}),checkpoint)
    del model,state;torch.cuda.empty_cache()
    directory=out/'pair_short_raw'
    subprocess.run([sys.executable,'-u','-m','scripts.research.generate_weighted_minima','--checkpoint',str(checkpoint),'--config',str(cfgfile),
        '--conditions',str(manifest),'--out',str(directory),'--device','cuda','--samples','16','--midpoint-steps','32','--seed',str(s['evaluation_seeds'][si]),'--allow-legacy-pickle'],check=True)
    generated=json.loads((directory/'generation.json').read_text());assert generated['complete']
    for row in generated['rows']:
        f=directory/row['file'];assert sha(f)==row['sha256']
        with np.load(f) as v:x=torch.from_numpy(v['raw_positions'].copy())
        raw.append(dict(arm='pair_short',condition=row['condition'],raw_sha256=row['sha256'],**assess(x,row['condition'],list(range(16)))))
    write(out/'audit.json',dict(complete=True,protocol_sha256=ph,rows=raw));xp=out/'physical_protocol.json' 
    write(xp,dict(frozen=True,native_protocol_sha256=ph,methods=s['methods'],replica=si,condition_count=16,samples_per_condition=16,
        xtb_binary=s['xtb_binary'],xtb_binary_sha256=s['xtb_binary_sha256'],xtb=s['xtb'],thresholds=s['thresholds'],scope=s['scope']))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,'-u','-m','scripts.research.score_native_minima','--project',str(root),'--run',str(out),'--protocol',str(xp),'--out',str(out/'xtb')],env=env,check=True)
    physical=json.loads((out/'xtb/audit.json').read_text());assert physical['complete']
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,replica=si,raw_audit_sha256=sha(out/'audit.json'),xtb_audit_sha256=sha(out/'xtb/audit.json'),
        summary=physical['summary'],new_neural_trajectories=512,additional_derived_outputs=256,total_scored_outputs=768,new_gfn2_attempts=768,new_training_steps=0,new_esen_queries=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--seed-index',type=int,choices=[0,1]);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.project,a.protocol)
    else:assert a.out is not None and a.seed_index is not None;run(a.project,a.protocol,a.out,a.seed_index)


if __name__=='__main__':main()
