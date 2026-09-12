#!/usr/bin/env python3
"""Internal TRAINING-parent pilot: distill identified angular distributions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from cfm_mol.angular_distillation import vmf_teacher_kl
from cfm_mol.local_site_guide import StiffnessSiteGuide
from scripts.research.evaluate_chemical_policy import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replica',type=int,choices=[0,1],required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/stiffness_pilot_protocol_v1.json';protocol=json.loads(pp.read_text())
    directory=args.project/'runs/angular_curvature_probe_v1'
    report=json.loads((directory/'results.json').read_text());audit=json.loads((directory/'audit.json').read_text())
    assert report['complete'] and audit['complete'] and sha(directory/'results.json')==protocol['probe_results_sha256']==audit['results_sha256']
    assert sha(directory/'audit.json')==protocol['probe_audit_sha256'] and sha(directory/'trace.pt')==report['trace_sha256']
    source=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
    conditions=report['condition'];numbers=torch.tensor(conditions['numbers'],dtype=torch.long)
    electronic=torch.tensor([conditions['charge'],conditions['spin_multiplicity'],protocol['kT_eV']],dtype=torch.float64)
    ids=[r['context'] for r in report['diagnostics'] if r['full_rank']]
    assert len(ids)==32
    x=torch.stack([source['states'][i]['positions'] for i in ids])
    bonds=torch.stack([source['states'][i]['graph']['bond_orders'] for i in ids])
    roots=torch.tensor([[source['selected'][i]['leaf'],source['selected'][i]['anchor']] for i in ids])
    teacher=torch.tensor([report['diagnostics'][i]['fitted_parameter'] for i in ids],dtype=torch.float64)
    order=sorted(ids,key=lambda i:hashlib.sha256((str(protocol['split_seed'])+':'+str(source['selected'][i]['parent_id'])).encode()).hexdigest())
    train_ids=order[:protocol['fit_contexts']];check_ids=order[protocol['fit_contexts']:]
    torch.manual_seed(protocol['model_seeds'][args.replica]);model=StiffnessSiteGuide(**protocol['architecture']).double()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    generator=torch.Generator().manual_seed(protocol['data_seeds'][args.replica])
    @torch.no_grad()
    def assess():
        eta=model(x,bonds,numbers,electronic,roots)[0][:,0]
        kl=vmf_teacher_kl(teacher,eta);answer={}
        for label,subset in [('fit',train_ids),('withheld_training_parents',check_ids)]:
            force=[];work=[]
            for i in subset:
                center=source['states'][i];leaf,anchor=roots[i].tolist()
                u0=center['positions'][leaf]-center['positions'][anchor];u0/=u0.norm()
                for probe in source['probes']:
                    if probe['context']!=i or probe['role']!='check' or not probe['valid']:continue
                    state=source['states'][probe['state_id']];v=state['positions'][leaf]-state['positions'][anchor];r=v.norm();u=v/r
                    full=state['force_eV_A']-protocol['restraint_eV_A2']*state['positions'];full-=full.mean(0)
                    target=r*(full[leaf]-(full[leaf]*u).sum()*u)/protocol['kT_eV']
                    prediction=eta[i]-(eta[i]*u).sum()*u
                    force.append(float((prediction-target).square().mean()))
                    actual=float((state['potential_eV']-center['potential_eV'])/protocol['kT_eV'])
                    work.append(abs(-float(eta[i]@(u-u0))-actual))
            answer[label]=dict(contexts=len(subset),teacher_kl=float(kl[subset].mean()),
                heldout_angle_force_mse=sum(force)/len(force),heldout_angle_work_mae_over_kT=sum(work)/len(work),
                predicted_concentrations=eta[subset].norm(dim=1).tolist())
        return answer
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    result=dict(complete=False,replica=args.replica,protocol_sha256=sha(pp),probe_results_sha256=sha(directory/'results.json'),
        fit_context_ids=train_ids,withheld_context_ids=check_ids,initial=assess(),history=[],new_physical_queries=0,
        scientific_submission_ready=False)
    write(output,result);started=time.perf_counter()
    fit_indices=torch.tensor(train_ids)
    for step in range(protocol['steps']):
        selected=fit_indices[torch.randint(len(fit_indices),(protocol['batch_size'],),generator=generator)]
        predicted=model(x[selected],bonds[selected],numbers,electronic,roots[selected])[0][:,0]
        loss=vmf_teacher_kl(teacher[selected],predicted).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite conditional-distillation loss')
        optimizer.zero_grad();loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite distillation gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        if step%200==0 or step+1==protocol['steps']:
            row=dict(step=step+1,teacher_kl=float(loss.detach()));result['history'].append(row);print(json.dumps(row),flush=True)
    result['fitting_seconds']=time.perf_counter()-started;result['final']=assess()
    torch.save(dict(configuration=model.configuration,state_dict=model.state_dict(),initial_state_dict=initial,
        fit_context_ids=train_ids,withheld_context_ids=check_ids,generator_state=generator.get_state(),protocol_sha256=sha(pp)),args.out/'model.pt')
    result.update(complete=True,checkpoint_sha256=sha(args.out/'model.pt'),parameters=sum(p.numel() for p in model.parameters()),
        scope='Internal learnability check using24 old TRAINING parents and8 withheld TRAINING parents; parameterization was chosen after inspecting the full probe summary. Not independent molecular validation, acceptance improvement or ICLR readiness.')
    write(output,result)


if __name__=='__main__':main()
