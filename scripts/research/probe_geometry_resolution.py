"""Paired integration-resolution diagnostic; unchanged weights and initial states."""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.matched_physical_connection import PhysicalFieldTransform
from scripts.research.run_matched_connection import load_parent,make_context,sample,score
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.run_geometry_recovery import review_geometry
from scripts.research.audit_generator_output_support import assess
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();cfg=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    assert cfg['frozen'];a.out.mkdir(parents=True,exist_ok=False)
    study=json.loads((root/cfg['source_protocol']).read_text());assert sha(root/cfg['source_protocol'])==cfg['source_protocol_sha256']
    spec=json.loads((root/study['parent_protocol']).read_text());spec.update(strength_limit=4.)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=base.HarmonicSource();arm=study['parents'][0];model=load_parent(root,arm);context=make_context(arm,source)
    head,_=restore_head(root,study['physical_heads'][0]);parent_hash=base.state_hash(model)
    baseline=root/cfg['baseline'];original=json.loads((baseline/'generation/generation.json').read_text())
    lookup={row['condition_index']:row for row in original['rows']};records=[];folder=a.out/'generation';folder.mkdir()
    for ci in cfg['condition_indices']:
        condition=study['conditions'][ci]['condition'];original_row=lookup[ci]
        file=baseline/'generation'/original_row['file'];assert sha(file)==original_row['sha256']
        saved=torch.load(file,map_location='cpu',weights_only=False);positions=[];starts=[];core=[0]
        transform=PhysicalFieldTransform(model,arm['spec'],head,4.,strength_limit=4.)
        hook=model.dynamics.egnn.register_forward_hook(lambda *_:core.__setitem__(0,core[0]+1))
        for begin in [0,8]:
            seed=study['evaluation_seeds'][0]*1000003+ci*100003+begin
            x,x0=sample(model,arm,source,context,condition,seed,8,cfg['backbone_calls'],transform)
            positions.append(x.cpu().double());starts.append(x0.cpu().double())
        hook.remove();initial=torch.cat(starts);torch.testing.assert_close(initial,saved['initial_positions'],atol=0,rtol=0)
        assert core[0]==2*cfg['backbone_calls'] and transform.calls==cfg['backbone_calls']
        x=torch.cat(positions);target=folder/f'steps512_c{ci}.pt'
        atomic_save(dict(positions=x,initial_positions=initial,condition=condition,protocol_sha256=ph,
            model_state_sha256=parent_hash,head_state_sha256=base.state_hash(head),backbone_calls=cfg['backbone_calls'],
            head_calls=cfg['backbone_calls']//2,source_replayed_exactly=True),target)
        quality=assess(x,condition,list(range(16)))
        records.append(dict(method='steps512',condition_index=ci,file=target.name,sha256=sha(target),**quality))
        print(json.dumps(dict(condition=ci,graph=quality['graph_supported'],attempted=16)),flush=True)
    assert base.state_hash(model)==parent_hash
    report=dict(complete=True,protocol_sha256=ph,rows=records,new_neural_outputs=128,
        model_state_sha256=parent_hash,head_state_sha256=base.state_hash(head),new_oracle_queries=0)
    write(folder/'generation.json',report)
    geometry=review_geometry(folder,report,a.out/'geometry.json')
    score(spec,ph,0,[(folder,report)],a.out/'xtb')
    new=json.loads((a.out/'xtb/results.json').read_text());old=json.loads((baseline/'xtb/results.json').read_text())
    old_geometry=json.loads((baseline/'geometry.json').read_text())['rows']
    def summarize(rows,physical):
        subset=[r for r in rows if r['condition'] in cfg['condition_indices']]
        scores={(r['condition_index'],r['sample_index']):r for r in physical['rows']}
        return dict(attempted=len(subset),graph=sum(r['graph'] for r in subset),
            geometry=sum(r['closed_shell_geometry_pass'] for r in subset),
            geometry_force=sum(r['closed_shell_geometry_pass'] and scores[r['condition'],r['sample']]['success']
                and scores[r['condition'],r['sample']]['rms_force']<=5 for r in subset),
            heavy_disconnected=sum(r['heavy_components']>1 for r in subset),
            planarity_failures=sum(r['graph'] and not r.get('planar_groups_pass',False) for r in subset))
    summary={'128_calls':summarize(old_geometry,old),'512_calls':summarize(geometry,new)}
    result=dict(complete=True,protocol_sha256=ph,summary=summary,source_replayed_exactly=True,
        unchanged_parent_and_head=True,new_generation_outputs=128,new_gfn2_attempts=128,new_esen_queries=0,
        reused_baseline_outputs=128,geometry_optimized=False,scope=cfg['scope'])
    write(a.out/'complete.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
