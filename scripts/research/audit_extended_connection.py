"""Audit extended strength calibration and its independently gated confirmation."""
import argparse,datetime,json
from pathlib import Path
import numpy as np
import torch
from scripts.research.audit_matched_connection import audit_outputs
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['select','audit'],required=True);p.add_argument('--selection',type=Path);a=p.parse_args();torch.set_num_threads(2)
    a.project=a.project.resolve();a.run=a.run.resolve()
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'] and not a.out.exists()
    oldfile=a.project/spec['original_selection'];assert sha(oldfile)==spec['original_selection_sha256'];old=json.loads(oldfile.read_text())
    for file,digest in old['validation_provenance'].items():assert sha(a.project/file)==digest
    if a.stage=='select':
        assert not (a.run/'test').exists();phase='validation';strengths={n:spec['new_strengths'] for n in ['fm','gaga']}
    else:
        selection=json.loads(a.selection.read_text());assert selection['complete'] and selection['advance'] and selection['protocol_sha256']==ph
        for file,digest in selection['validation_provenance'].items():assert sha(a.project/file)==digest
        phase='test';strengths={n:[1.,selection['strengths'][n]] for n in ['fm','gaga']}
    values=[];provenance={}
    for si in [0,1]:
        result,origin=audit_outputs(a.project,a.run/f'{phase}/s{si}',spec,ph,si,phase,spec['heads'],strengths);values.append(result);provenance.update(origin)
    methods=list(values[0]['graph']);arrays={k:np.stack([[r[k][m] for m in methods] for r in values]) for k in ['graph','success','force']}
    joint=arrays['graph']&arrays['success']&(arrays['force']<=5)
    summary={m:dict(graph_rate=float(arrays['graph'][:,mi].mean()),graph_by_seed=arrays['graph'][:,mi].mean((1,2)).tolist(),
        joint_rate=float(joint[:,mi].mean()),joint_by_seed=joint[:,mi].mean((1,2)).tolist(),attempted=joint[:,mi].size,
        joint_counts_by_seed=joint[:,mi].sum((1,2)).tolist(),failures=int((~arrays['success'][:,mi]).sum())) for mi,m in enumerate(methods)}
    if a.stage=='select':
        curves={name:{} for name in ['fm','gaga']}
        for name in curves:
            for i,alpha in enumerate(spec['all_strengths'][:4]):
                method=f'{name}_a{i}';r=dict(old['summary'][method]);graphs=[]
                for si in [0,1]:
                    path=a.project/f'runs/matched_connection_v1/validation/s{si}/{name}/generation/generation.json'
                    assert sha(path)==old['validation_provenance'][str(path.relative_to(a.project))]
                    report=json.loads(path.read_text());rows=[v for v in report['rows'] if v['method']==method]
                    assert len(rows)==8 and all(v['strength']==alpha for v in rows);graphs.append(sum(v['graph_supported'] for v in rows)/128)
                r['graph_by_seed']=graphs;curves[name][str(alpha)]=r
            for i,alpha in enumerate(spec['new_strengths']):curves[name][str(alpha)]=summary[f'{name}_a{i}']
        selected={};eligibility={}
        for name,curve in curves.items():
            reference=curve['1.0'];eligibility[name]={}
            for alpha,r in curve.items():eligibility[name][alpha]=bool(r['graph_rate']>=reference['graph_rate']-.01-1e-12 and
                min(np.array(r['graph_by_seed'])-reference['graph_by_seed'])>=-.03-1e-12)
            eligible=[key for key,v in eligibility[name].items() if v]
            chosen=max(eligible,key=lambda key:(curve[key]['joint_rate'],-float(key)));selected[name]=float(chosen)
        fm=curves['fm'][str(selected['fm'])];ref=curves['fm']['1.0'];change=fm['joint_rate']-ref['joint_rate'];seedchanges=np.array(fm['joint_by_seed'])-ref['joint_by_seed']
        advance=bool(change>=.02-1e-12 and (seedchanges>0).all())
        value=dict(complete=True,protocol_sha256=ph,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),strengths=selected,curves=curves,
            graph_guard_eligibility=eligibility,advance=advance,validation_fm_gain_over_strength1=change,validation_fm_gain_by_seed=seedchanges.tolist(),
            validation_provenance=provenance,original_selection_sha256=sha(oldfile),fresh_test_outputs_used=False,new_neural_outputs=1024,
            new_gfn2_attempts=1024,new_esen_queries=0,new_optimizer_steps=0,scope=spec['scope'])
    else:
        rng=np.random.default_rng(61081);means={m:joint[:,mi].mean(-1) for mi,m in enumerate(methods)};contrasts={}
        for name,left,right in [('fm_improvement','fm_a1','fm_a0'),('gaga_improvement','gaga_a1','gaga_a0'),('baseline_fm_minus_gaga','fm_a0','gaga_a0'),('selected_fm_minus_gaga','fm_a1','gaga_a1')]:
            contrasts[name]=bootstrap(means[left]-means[right],rng,20000)
        for mi,m in enumerate(methods):
            valid=arrays['graph'][:,mi]&arrays['success'][:,mi];force=arrays['force'][:,mi]
            summary[m].update(median_valid_force=float(np.median(force[valid])) if valid.any() else None,
                joint_curve={str(t):float((valid&(force<=t)).mean()) for t in [.5,1.,2.,5.,10.]})
        graphdiff=arrays['graph'][:,methods.index('fm_a1')].mean(-1)-arrays['graph'][:,methods.index('gaga_a1')].mean(-1)
        contrasts['selected_fm_minus_gaga_graph']=bootstrap(graphdiff,rng,20000)
        path=a.out.with_suffix('.npz');np.savez_compressed(path,**arrays)
        positive=lambda key:contrasts[key]['ci95'][0]>0 and min(contrasts[key]['by_seed'])>0
        value=dict(complete=True,protocol_sha256=ph,selection_sha256=sha(a.selection),methods=methods,summary=summary,contrasts=contrasts,
            fm_improvement_gate=positive('fm_improvement'),fm_over_gaga_joint_gate=positive('selected_fm_minus_gaga'),
            fm_over_gaga_graph_gate=positive('selected_fm_minus_gaga_graph'),provenance=provenance,arrays_sha256=sha(path),
            new_neural_outputs=4096,new_gfn2_attempts=4096,new_esen_queries=0,new_optimizer_steps=0,scope=spec['scope'])
    write(a.out,value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
