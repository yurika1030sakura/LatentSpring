"""Check geometry reuse on the existing GAGA-comparison validation panel only."""
import argparse,datetime,json,hashlib
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base,connectivity_feedback as feedback
from scripts.research.run_gaga_feedback import evaluate
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def prepare(root,out):
    assert not out.exists();replicas=[]
    for si in [0,1]:
        protocol=root/f'research/evidence/gaga_feedback_distance_s{si}_v1.json';p=json.loads(protocol.read_text());path=root/f'runs/gaga_feedback_v1/training/s{si}/distance/last.ckpt'
        refs={k:f'runs/gaga_feedback_v1/training/s{si}/'+v for k,v in [('distance','distance/validation/distance_results.json'),('gaga','baselines/gaga_650_results.json')]}
        replicas.append(dict(protocol=str(protocol.relative_to(root)),protocol_sha256=sha(protocol),checkpoint=str(path.relative_to(root)),checkpoint_sha256=sha(path),reference_reports={k:dict(path=v,sha256=sha(root/v)) for k,v in refs.items()}))
    spec=dict(format='cached_feedback_v1',frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),replicas=replicas,calls=[64,128],
        scope='Internal validation only, using frozen matched-EGNN distance-feedback models. Stateful reuse changes the sampling process; it does not integrate the old two-pass ODE. No new training, no physical oracle and no fresh confirmation outcomes.',
        selection='Advance cached128 only if its pooled graph validity exceeds original distance128 by at least2 percentage points and both seeds improve. Cached64 is a cost diagnostic. A GAGA superiority claim requires separate fresh confirmation.')
    write(out,spec)


def run(root,protocol,out,si):
    spec=json.loads(protocol.read_text());ph=sha(protocol);item=spec['replicas'][si];p=root/item['protocol'];assert sha(p)==item['protocol_sha256'];old=json.loads(p.read_text())
    ckpt=root/item['checkpoint'];assert sha(ckpt)==item['checkpoint_sha256'];state=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert state['protocol_sha256']==item['protocol_sha256'] and state['global_step']==old['training_steps']
    panel=root/old['panel'];assert sha(panel)==old['panel_sha256'];rows=json.loads(panel.read_text())['validation_rows']
    torch.set_num_threads(2);model=feedback.install(base.initialize(old,'cuda'));model.load_state_dict(state['ema_state_dict'],strict=True);model.eval().requires_grad_(False)
    source=base.HarmonicSource(old['edge_log_width']);context=feedback.GeometryContext(source,old['context'],old['tree_regularization'])
    out.mkdir(parents=True,exist_ok=False);refs={}
    for name,ref in item['reference_reports'].items():
        p=root/ref['path'];assert sha(p)==ref['sha256'];r=json.loads(p.read_text());assert r['complete'] and r['attempted']==512
        assert r['settings']['seed']==old['validation_seed'] and r['settings']['count']==16 and r['settings']['calls']==128
        assert [x['condition']['composition_hex'] for x in r['rows']]==[c['composition_hex'] for c in rows]
        for x in r['rows']:assert sha(p.parent/(('distance' if name=='distance' else 'gaga_650')+f'_c{x["condition_index"]}.pt'))==x['sample_sha256']
        refs[name]=r
    counter=[0]
    def count(module,args):counter[0]+=1
    hook=model.dynamics.egnn.register_forward_pre_hook(count);results={}
    for calls in spec['calls']:
        cfg=dict(old,cached_feedback=True,evaluation_calls=calls);before=counter[0]
        result=evaluate(model,source,cfg,context,rows,old['validation_seed'],16,out/f'calls{calls}',f'cached{calls}')
        assert counter[0]-before==len(rows)*2*calls
        for i,r in enumerate(result['rows']):
            values=torch.load(out/f'calls{calls}/cached{calls}_c{i}.pt',map_location='cpu',weights_only=False)
            original=root/Path(item['reference_reports']['distance']['path']).parent/f'distance_c{i}.pt'
            baseline=torch.load(original,map_location='cpu',weights_only=False)
            torch.testing.assert_close(values['initial_positions'],baseline['initial_positions'],atol=0,rtol=0)
        results[str(calls)]=dict(attempted=512,graph_supported=result['graph_supported'],by_composition=[r['graph_supported']/16 for r in result['rows']],report_sha256=sha(out/f'calls{calls}/cached{calls}_results.json'))
    hook.remove()
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,replica=si,results=results,model_state_sha256=base.state_hash(model),
        reference_graph_counts={k:v['graph_supported'] for k,v in refs.items()},new_neural_outputs=1024,new_oracle_queries=0,new_training_steps=0,scope=spec['scope']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--seed-index',type=int,choices=[0,1]);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.project,a.protocol)
    else:assert a.out is not None and a.seed_index is not None;run(a.project,a.protocol,a.out,a.seed_index)


if __name__=='__main__':main()
