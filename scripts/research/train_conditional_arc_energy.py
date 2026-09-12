#!/usr/bin/env python3
"""Frozen same-data comparison of conditional work and work-plus-force learning."""
import argparse
import copy
import json
from pathlib import Path
from statistics import mean
import time

import torch
import torch.nn.functional as F

from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def pack(rows,force=False):
    length=max(len(r['directions']) for r in rows)
    directions=[];work=[];gradient=[];mask=[];present=[]
    for r in rows:
        n=len(r['directions']);pad=length-n
        directions.append(torch.cat([r['directions'],r['directions'][0:1].expand(pad,-1)]))
        work.append(F.pad(r['work_eV'],(0,pad)))
        gradient.append(F.pad(r['angular_energy_gradients_eV'],(0,0,0,pad)))
        mask.append(F.pad(r['training_mask'],(0,pad)))
        present.append(torch.arange(length)<n)
    return dict(x=torch.stack([r['positions'] for r in rows]),bonds=torch.stack([r['bonds'] for r in rows]),
        numbers=rows[0]['numbers'],electronic=torch.stack([r['electronic'] for r in rows]),roots=torch.stack([r['root'] for r in rows]),
        directions=torch.stack(directions).requires_grad_(force),work=torch.stack(work),gradient=torch.stack(gradient),
        mask=torch.stack(mask),present=torch.stack(present))


def predictions(model,batch,force=False,create_graph=False):
    context=model.encode(batch['x'],batch['bonds'],batch['numbers'],batch['electronic'],batch['roots'])
    energy=model.energy(batch['directions'],context)
    work=energy-energy[:,0:1]
    gradient=None
    if force:
        raw=torch.autograd.grad(energy.sum(),batch['directions'],create_graph=create_graph,retain_graph=True)[0]
        gradient=raw-(raw*batch['directions']).sum(-1,keepdim=True)*batch['directions']
    return work,gradient


def objective(model,rows,force_weight,scale):
    assert all(r['role']=='fit' for r in rows)
    batch=pack(rows,force=force_weight>0)
    work,grad=predictions(model,batch,force=force_weight>0,create_graph=force_weight>0)
    work_mask=batch['mask'].clone();work_mask[:,0]=False
    loss=F.smooth_l1_loss(work/scale,batch['work']/scale,reduction='none')
    work_loss=((loss*work_mask).sum(1)/work_mask.sum(1)).mean()
    force_loss=work_loss*0
    if force_weight:
        loss=F.smooth_l1_loss(grad/scale,batch['gradient']/scale,reduction='none').mean(-1)
        force_loss=((loss*batch['mask']).sum(1)/batch['mask'].sum(1)).mean()
    return work_loss+force_weight*force_loss,dict(work=float(work_loss.detach()),force=float(force_loss.detach()))


def evaluate(model,records,batch_size=16):
    model.eval();values=[]
    for index in sorted({r['index'] for r in records}):
        selected=[r for r in records if r['index']==index]
        for begin in range(0,len(selected),batch_size):
            rows=selected[begin:begin+batch_size];batch=pack(rows,force=True)
            work,gradient=predictions(model,batch,force=True)
            work=work.detach();gradient=gradient.detach()
            for j,row in enumerate(rows):
                for family in ['local_check','arc']:
                    ids=[i for i,name in enumerate(row['sample_roles']) if name==family or name.startswith(family+'_')]
                    if not ids:continue
                    values.append(dict(index=index,parent=row['parent'],context=row['context'],role=row['role'],family=family,
                        work_MAE_eV=float((work[j,ids]-row['work_eV'][ids]).abs().mean()),
                        angular_gradient_MSE_eV2=float((gradient[j,ids]-row['angular_energy_gradients_eV'][ids]).square().mean())))
    summary={}
    for role in ['fit','withheld_parent','withheld_composition']:
        for family in ['local_check','arc']:
            group=[v for v in values if v['role']==role and v['family']==family]
            per_condition={}
            for index in sorted({v['index'] for v in group}):
                rows=[v for v in group if v['index']==index];parents=sorted({v['parent'] for v in rows})
                per_condition[str(index)]={key:mean(mean(v[key] for v in rows if v['parent']==parent) for parent in parents)
                    for key in ['work_MAE_eV','angular_gradient_MSE_eV2']}
            summary[role+'/'+family]=dict(contexts=len(group),parents=len({(v['index'],v['parent']) for v in group}),
                per_condition=per_condition,**{key:mean(v[key] for v in per_condition.values()) for key in ['work_MAE_eV','angular_gradient_MSE_eV2']})
    return dict(summary=summary,context_metrics=values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replica',type=int,choices=[0,1],required=True)
    p.add_argument('--objective',choices=['work','work_force'],required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/conditional_arc_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    dp=args.project/protocol['data_run'];header=json.loads((dp/'results.json').read_text())
    assert header['complete'] and sha(dp/'results.json')==protocol['data_results_sha256']
    assert sha(dp/'data.pt')==protocol['data_sha256']==header['data_sha256']
    records=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    assert len(records)==438 and sum(r['role']=='fit' for r in records)==206
    training=[r for r in records if r['role']=='fit']
    assert {(r['index'],r['parent']) for r in training}=={(int(i),p) for i,ids in protocol['fit_parent_ids'].items() for p in ids}
    for row in records:
        expected=torch.tensor([row['role']=='fit' and name!='local_check' for name in row['sample_roles']])
        assert torch.equal(row['training_mask'],expected)
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    seed=protocol['seeds'][args.replica];torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+1)
    model=ConditionalArcEnergy(**protocol['model']).double()
    initialization=copy.deepcopy(model)
    baseline_site=ConditionalArcEnergy(**dict(protocol['model'],restraint=0.)).double()
    report=dict(complete=False,objective=args.objective,replica=args.replica,seed=seed,protocol_sha256=sha(pp),
        data_sha256=protocol['data_sha256'],curves=[],new_physical_queries=0,scientific_submission_ready=False,
        architecture='Masked context predicts root-to-passive radial energy curves; exact confinement and site-prior energy included.',
        scope='Frozen800-step TRAINING pilot. All held-out parent/composition and FIT local-check labels are excluded from fitting. No early selection of a favorable checkpoint.')
    write(output,report);start=time.monotonic()
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    force_weight=protocol['force_weights'][args.objective]
    groups={i:{p:[r for r in training if r['index']==i and r['parent']==p] for p in sorted({r['parent'] for r in training if r['index']==i})}
        for i in protocol['fit_conditions']}
    try:
        report['initialization']=evaluate(initialization,records)
        report['physical_site']=evaluate(baseline_site,records)
        write(output,report)
        for step in range(1,protocol['steps']+1):
            index=protocol['fit_conditions'][int(torch.randint(len(groups),(1,),generator=rng))]
            parents=list(groups[index]);rows=[]
            for _ in range(protocol['batch_contexts']):
                parent=parents[int(torch.randint(len(parents),(1,),generator=rng))]
                contexts=groups[index][parent];rows.append(contexts[int(torch.randint(len(contexts),(1,),generator=rng))])
            model.train();optimizer.zero_grad(set_to_none=True)
            loss,parts=objective(model,rows,force_weight,protocol['energy_scale_eV'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite conditional learning loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite conditional learning gradient')
            optimizer.step()
            if step in protocol['checkpoints']:
                metric=evaluate(model,records)
                report['curves'].append(dict(step=step,loss=float(loss),last_batch=parts,metrics=metric['summary']))
                write(output,report)
                print(json.dumps(dict(step=step,objective=args.objective,replica=args.replica,
                    withheld_arc_work_MAE_eV=metric['summary']['withheld_parent/arc']['work_MAE_eV'],
                    withheld_composition_arc_work_MAE_eV=metric['summary']['withheld_composition/arc']['work_MAE_eV'])),flush=True)
        final=evaluate(model,records)
        checkpoint=dict(state_dict=model.state_dict(),configuration=model.configuration,protocol_sha256=sha(pp),
            data_sha256=protocol['data_sha256'],objective=args.objective,replica=args.replica,seed=seed,step=protocol['steps'],
            stream='fresh_training',force_weight=force_weight)
        torch.save(checkpoint,args.out/'model.pt')
        report.update(complete=True,final=final,model_sha256=sha(args.out/'model.pt'),elapsed_seconds=time.monotonic()-start,
            parameter_count=sum(p.numel() for p in model.parameters()),optimizer_steps=protocol['steps'])
        write(output,report)
    except Exception as exc:
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration),args.out/'failed_model.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write(output,report);raise


if __name__=='__main__':main()
