#!/usr/bin/env python3
"""Bounded masked angular force regression, without loading development coordinates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from cfm_mol.masked_angular_data import training_angular_examples
from cfm_mol.masked_angular_guide import MaskedAngularGuide,angular_surface_score


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--table',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--replica',type=int,choices=[0,1],required=True)
    p.add_argument('--model',choices=['tensor','vector'],required=True);args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/masked_angular_protocol_v1.json';protocol=json.loads(pp.read_text())
    header=json.loads((args.table/'results.json').read_text());tp=args.table/'training.pt'
    if (not header['complete'] or sha(args.table/'results.json')!=protocol['table_results_sha256']
            or sha(tp)!=protocol['training_artifact_sha256']):raise ValueError('Changed training source')
    data=torch.load(tp,map_location='cpu',weights_only=False);d=training_angular_examples(data)
    torch.manual_seed(protocol['model_seeds'][args.replica])
    model=MaskedAngularGuide(**protocol['architecture'],tensor=args.model=='tensor').double()
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    generator=torch.Generator().manual_seed(protocol['data_seeds'][args.replica]);history=[];start=time.perf_counter()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise FileExistsError(args.out/'results.json')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    for step in range(protocol['training_steps']):
        parents=torch.randint(len(d['parent_groups']),(protocol['batch_size'],),generator=generator)
        chosen=torch.tensor([int(d['parent_groups'][i][int(torch.randint(len(d['parent_groups'][i]),(1,),generator=generator))])
            for i in parents.tolist()])
        examples=d['examples'][chosen];sid=examples[:,0];roots=examples[:,1:3]
        eta,matrix,_=model(d['x'][sid],d['bonds'][sid],d['numbers'],d['electronic'],roots)
        predicted=angular_surface_score(d['directions'][chosen],eta,matrix)
        target=d['targets'][chosen];norm=target.norm(dim=1,keepdim=True)
        target=target*torch.clamp(protocol['score_clip']/norm.clamp_min(1e-300),max=1)
        error=(predicted-target).square().mean()/protocol['score_clip']**2
        penalty=(eta.square().sum(1)+matrix.square().sum((1,2))).mean()/protocol['architecture']['bound']**2
        loss=error+protocol['coefficient_regularization']*penalty
        if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss')
        optimizer.zero_grad();loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        if step%50==0 or step==protocol['training_steps']-1:
            row=dict(step=step+1,normalized_batch_mse=float(error.detach()),coefficient_penalty=float(penalty.detach()))
            history.append(row);print(json.dumps(row),flush=True)
    # Complete in-sample force audit; no development coordinates or labels used.
    total=0.;zero=0.;count=0
    with torch.no_grad():
        for begin in range(0,len(d['examples']),256):
            ex=d['examples'][begin:begin+256];sid=ex[:,0]
            eta,a,_=model(d['x'][sid],d['bonds'][sid],d['numbers'],d['electronic'],ex[:,1:3])
            prediction=angular_surface_score(d['directions'][begin:begin+len(ex)],eta,a)
            y=d['targets'][begin:begin+len(ex)];y=y*torch.clamp(protocol['score_clip']/y.norm(dim=1,keepdim=True).clamp_min(1e-300),max=1)
            total+=float((prediction-y).square().sum());zero+=float(y.square().sum());count+=y.numel()
    artifact=dict(configuration=model.configuration,state_dict=model.state_dict(),initial_state_dict=initial,
        protocol_sha256=sha(pp),training_sha256=sha(tp),model=args.model,replica=args.replica,
        model_seed=protocol['model_seeds'][args.replica],data_seed=protocol['data_seeds'][args.replica],
        data_generator_state=generator.get_state(),parent_ids=d['parent_ids'])
    torch.save(artifact,args.out/'model.pt')
    report=dict(complete=True,model=args.model,replica=args.replica,protocol_sha256=sha(pp),training_sha256=sha(tp),
        checkpoint_sha256=sha(args.out/'model.pt'),parameters=sum(p.numel() for p in model.parameters()),
        training_parents=len(d['parent_groups']),scored_states=len(d['x']),angular_examples=len(d['examples']),
        training_mse=total/count,zero_predictor_mse=zero/count,seconds=time.perf_counter()-start,history=history,
        inherited_training_raw_queries=header['streams']['training']['raw_queries'],new_physical_queries=0,
        development_coordinates_loaded=False,reference_coordinates_loaded=False,scientific_submission_ready=False)
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
