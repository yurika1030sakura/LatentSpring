"""Verify verbatim head transfer and evaluate it against independently audited controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.physical_connection import make_physical_connection
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_seed_replication import verify_physics
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha
from scripts.research.summarize_seed_replication import paired_intervals

def audit_one(root,protocol,run,si,out):
    assert not out.exists();proposal=json.loads(protocol.read_text());ph=sha(protocol);folder=run/f's{si}';done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['fit']==si
    source=root/done['source_study'];original=json.loads((source/'complete.json').read_text());assert original['complete'] and sha(source/'complete.json')==done['source_study_sha256']
    spec=json.loads((folder/'resolved_protocol.json').read_text());resolved_hash=sha(folder/'resolved_protocol.json');assert resolved_hash==done['resolved_protocol_sha256'];conditions=spec['test_rows'];assert len(conditions)==64
    arrays=dict(graph=np.zeros((2,64,16),bool),success=np.zeros((2,64,16),bool),force=np.full((2,64,16),np.inf),energy=np.full((2,64,16),np.nan));positions={};provenance={};reports={};cache=set()
    for mi,family in enumerate(['fm','gaga']):
        other='gaga' if family=='fm' else 'fm';info=done['heads'][family];expected=original['heads'][other]
        assert info['path']==expected['path'] and info['sha256']==expected['sha256'] and info['training_parent']==other
        file=root/info['path'];assert sha(file)==info['sha256'];saved=torch.load(file,map_location='cpu',weights_only=False);head=make_physical_connection(**saved['configuration']);head.load_state_dict(saved['state_dict'],strict=True);hh=base.state_hash(head);assert hh==info['head_state_sha256']
        path=folder/'parents'/family/'generation.json';report=json.loads(path.read_text());assert report['complete'] and report['head_state_sha256']==hh and report['protocol_sha256']==resolved_hash and len(report['rows'])==64
        parent=json.loads((source/'parents'/family/'generation.json').read_text());assert report['model_state_sha256']==parent['model_state_sha256'];provenance[str(path.relative_to(root))]=sha(path)
        for r in report['rows']:
            i=r['condition_index'];assert r['method']==family+'_cross_a0' and r['strength']==4.;file=path.parent/r['file'];assert sha(file)==r['sha256'];x=torch.load(file,map_location='cpu',weights_only=False)
            assert x['condition']==conditions[i] and x['protocol_sha256']==resolved_hash and x['head_state_sha256']==hh
            assert x['core_calls']==256 and x['head_calls']==(128 if family=='fm' else 256)
            control_row=next(v for v in parent['rows'] if v['condition_index']==i and v['method']==family+'_a0');control=torch.load(source/'parents'/family/control_row['file'],map_location='cpu',weights_only=False)
            torch.testing.assert_close(x['initial_positions'],control['initial_positions'],atol=0,rtol=0)
            quality=assess(x['positions'],conditions[i],list(range(16)));assert all(r[k]==v for k,v in quality.items());arrays['graph'][mi,i]=[v['graph_supported'] for v in quality['records']];positions[mi,i]=x['positions']
    xp=folder/'xtb';report=json.loads((xp/'results.json').read_text());assert report['complete'] and report['protocol_sha256']==resolved_hash and sha(xp/'results.json')==done['physical_results_sha256'];seen=set()
    for r in report['rows']:
        mi=['fm','gaga'].index(r['method'].split('_')[0]);i,j=r['condition_index'],r['sample_index'];assert (mi,i,j) not in seen;seen.add((mi,i,j));x=positions[mi,i][j]
        ok,energy,force=verify_physics(root,r,xp/'details'/r['task_id'],x,conditions[i],cache);graph=bool(arrays['graph'][mi,i,j]);assert r['graph']==graph and r['joint']==bool(graph and ok and force<=5)
        arrays['success'][mi,i,j]=ok;arrays['force'][mi,i,j]=force;arrays['energy'][mi,i,j]=energy
    assert len(seen)==2048==report['new_gfn2_attempts']==done['new_gfn2_attempts'];provenance[str((xp/'results.json').relative_to(root))]=sha(xp/'results.json')
    np.savez_compressed(out.with_suffix('.npz'),**arrays);write(out,dict(complete=True,fit=si,protocol_sha256=ph,source_study_sha256=done['source_study_sha256'],heads=done['heads'],provenance=provenance,arrays_sha256=sha(out.with_suffix('.npz')),new_parent_trajectories=2048,new_gfn2_attempts=2048,new_esen_queries=0,new_optimizer_steps=0))

def summarize(root,protocol,run,audits,out):
    assert not out.exists();ph=sha(protocol);cross=[];controls=[];proof={}
    for si in range(5):
        p=audits/f's{si}.json';d=json.loads(p.read_text());assert d['complete'] and d['fit']==si and d['protocol_sha256']==ph and sha(p.with_suffix('.npz'))==d['arrays_sha256'];cross.append(dict(np.load(p.with_suffix('.npz'))));proof[str(p.relative_to(root))]=sha(p)
        q=root/f'runs/seed_replication_v1/audits/s{si}.json';b=json.loads(q.read_text());assert b['complete'] and b['fit']==si and sha(q.with_suffix('.npz'))==b['arrays_sha256'];controls.append(dict(np.load(q.with_suffix('.npz'))));proof[str(q.relative_to(root))]=sha(q)
    cross={k:np.stack([r[k] for r in cross]) for k in cross[0]};control={k:np.stack([r[k] for r in controls]) for k in ['graph','force','success','energy']}
    joint=cross['graph']&cross['success']&(cross['force']<=5);cj=control['graph']&control['success']&(control['force']<=5);summary={};comparisons={}
    panel=json.loads((root/'runs/seed_replication_v1/panel/panel.json').read_text());strata=[0 if c['n_atoms']<=28 else 1 for c in panel['rows']]
    for mi,family in enumerate(['fm','gaga']):
        summary[family]=dict(attempted=5120,graph=float(cross['graph'][:,mi].mean()),joint=float(joint[:,mi].mean()),graph_by_fit=cross['graph'][:,mi].mean((1,2)).tolist(),joint_by_fit=joint[:,mi].mean((1,2)).tolist(),failures=int((~cross['success'][:,mi]).sum()))
        for target in ['parent','physical']:
            ci=b['methods'].index(family+'_'+target);name=family+'_cross_minus_'+target;comparisons[name]={}
            for tag,ids in [('all_five',[0,1,2,3,4]),('new_three',[2,3,4])]:comparisons[name][tag]=dict(graph=paired_intervals((cross['graph'][ids,mi].astype(float)-control['graph'][ids,ci]).mean(-1),strata=strata),joint=paired_intervals((joint[ids,mi].astype(float)-cj[ids,ci]).mean(-1),strata=strata))
    primary=comparisons['gaga_cross_minus_parent']['new_three']['joint'];gate=primary['composition_ci95'][0]>0 and min(primary['by_fit'])>0
    np.savez_compressed(out.with_suffix('.npz'),**cross);write(out,dict(complete=True,protocol_sha256=ph,summary=summary,contrasts=comparisons,primary_fm_head_to_gaga_transfer_gate=bool(gate),new_fit_crossed_ci_positive=bool(primary['crossed_fit_composition_ci95'][0]>0),provenance=proof,arrays_sha256=sha(out.with_suffix('.npz')),new_parent_trajectories=10240,new_gfn2_attempts=10240,new_esen_queries=0,new_optimizer_steps=0,scope='Verbatim head transfer within each fit pair. No target-generator force labels train the transferred head. Shared architecture and vocabulary; no all-architecture transfer claim. All directions and fits retained.'))
    print(json.dumps(dict(summary=summary,contrasts=comparisons,primary_gate=bool(gate))),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=range(5));p.add_argument('--audits',type=Path);a=p.parse_args();torch.set_num_threads(2)
    a.project=a.project.resolve();a.run=a.run.resolve()
    if a.audits:a.audits=a.audits.resolve()
    if a.fit is None:
        if a.audits is None:raise ValueError('Summary requires per-fit audits')
        summarize(a.project.resolve(),a.protocol,a.run,a.audits,a.out)
    else:audit_one(a.project.resolve(),a.protocol,a.run,a.fit,a.out)

if __name__=='__main__':main()
