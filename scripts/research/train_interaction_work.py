#!/usr/bin/env python3
"""Learn four-state electronic interactions with parent-disjoint diagnostics."""
import argparse
import json
import time
from pathlib import Path
import torch
from cfm_mol.interaction_work_model import InteractionWorkModel
from cfm_mol.chemical_edit_interaction import four_graphs
from scripts.research.audit_masked_angular import equal,sha
from scripts.research.evaluate_chemical_policy import write


def load_data(project,protocol):
    path=project/protocol['plan'];assert sha(path)==protocol['plan_sha256']
    plan=torch.load(path,map_location='cpu',weights_only=False);groups=[];provenance={}
    for index in [1,3,5]:
        directory=project/protocol['label_run']/f'condition_{index:02d}'
        ap=project/protocol['label_audit']/f'condition_{index:02d}/results.json'
        report=json.loads((directory/'results.json').read_text());audit=json.loads(ap.read_text())
        assert report['complete'] and audit['complete'] and audit['full_replay']
        assert report['protocol_sha256']==audit['protocol_sha256']==protocol['label_protocol_sha256']
        assert sha(directory/'results.json')==audit['source_results_sha256']
        assert sha(directory/'trace.pt')==audit['trace_sha256']==report['trace_sha256']
        provenance[str(index)]=dict(results_sha256=sha(directory/'results.json'),audit_sha256=sha(ap),trace_sha256=sha(directory/'trace.pt'))
        assert provenance[str(index)]==protocol['label_sources'][str(index)]
        data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        specs=[p for p in plan['pairs'] if p['index']==index]
        assert len(specs)==len(data['rows'])
        for source in [s for s in plan['sources'] if s['index']==index]:
            selected=[(spec,row) for spec,row in zip(specs,data['rows']) if spec['source_id']==source['source_id'] and row['valid']]
            assert source['fit_only'] and selected
            for spec,row in selected:
                assert spec['valid'] and row['actions']==spec['actions'] and row['parent']==source['parent']
            condition=source['condition']
            groups.append(dict(source_id=source['source_id'],index=index,parent=source['parent'],
                positions=torch.stack([spec['positions'] for spec,row in selected]),
                graphs=torch.stack([four_graphs(source['state']['graph']['bond_orders'],*spec['actions']) for spec,row in selected]),
                numbers=torch.tensor(condition['numbers'],dtype=torch.long),
                electronic=torch.tensor([[condition['charge'],condition['spin_multiplicity'],protocol['kT_eV']]]*len(selected),dtype=torch.float64),
                actions=torch.tensor([spec['actions'] for spec,row in selected],dtype=torch.long),
                target=torch.tensor([row['electronic_interaction_eV'] for spec,row in selected],dtype=torch.float64),
                additive=torch.tensor([row['oracle_additive_joint_work_eV'] for spec,row in selected],dtype=torch.float64),
                joint=torch.tensor([row['joint_work_eV'] for spec,row in selected],dtype=torch.float64)))
    assert len(groups)==27 and sum(len(g['target']) for g in groups)==440
    return groups,provenance


def predict(model,group):
    return model(*(group[k] for k in ('positions','graphs','numbers','electronic','actions')))


@torch.no_grad()
def metrics(model,groups):
    rows=[]
    for group in groups:
        pred=predict(model,group) if model is not None else torch.zeros_like(group['target'])
        error=pred-group['target'];assert torch.isfinite(pred).all()
        rows.append(dict(source_id=group['source_id'],parent=group['parent'],index=group['index'],
            MAE_eV=float(error.abs().mean()),MSE_eV2=float(error.square().mean()),
            # Both joint-work estimates use TRUE single-edit energies: this is
            # a mechanism diagnostic, not a deployed-policy comparison.
            oracle_single_joint_sign_accuracy=float(((group['additive']+pred<0)==(group['joint']<0)).double().mean()),
            prediction_eV=pred.tolist(),target_eV=group['target'].tolist()))
    return dict(parent_balanced={k:sum(row[k] for row in rows)/len(rows) for k in ('MAE_eV','MSE_eV2','oracle_single_joint_sign_accuracy')},rows=rows)


def make_model(variant,protocol):
    return InteractionWorkModel(protocol['elements'],**protocol['model'],linear=variant=='linear',environment=variant=='environment').double()


def train(project,out,pp,replica):
    protocol=json.loads(pp.read_text());groups,provenance=load_data(project,protocol)
    held_ids=set(protocol['diagnostic_source_ids'])
    fit=[g for g in groups if g['source_id'] not in held_ids];held=[g for g in groups if g['source_id'] in held_ids]
    assert len(fit)==18 and len(held)==9
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    write(out/'data.json',dict(sources=provenance,zero_fit=metrics(None,fit),zero_diagnostic=metrics(None,held),zero_all=metrics(None,groups)))
    for phase in ['diagnostic','full_fit']:
        training=fit if phase=='diagnostic' else groups
        seed=protocol['seeds'][replica]+(10000 if phase=='full_fit' else 0)
        for variant in protocol['variants']:
            start=time.monotonic();torch.manual_seed(seed);model=make_model(variant,protocol)
            optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate']);generator=torch.Generator().manual_seed(seed+1000)
            directory=out/phase/variant;directory.mkdir(parents=True);trace=[]
            for step in range(protocol['steps']):
                group=training[int(torch.randint(len(training),(1,),generator=generator))]
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.smooth_l1_loss(predict(model,group),group['target'],beta=protocol['huber_beta_eV'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite interaction loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
                trace.append(dict(step=step,source_id=group['source_id'],loss=float(loss),gradient_norm=float(norm)))
                if step%100==0:print(json.dumps(dict(phase=phase,variant=variant,replica=replica,**trace[-1])),flush=True)
            torch.save(dict(configuration=model.configuration,state_dict=model.state_dict(),variant=variant,phase=phase,seed=seed,
                protocol_sha256=sha(pp),training_source_ids=[g['source_id'] for g in training]),directory/'model.pt')
            report=dict(complete=True,phase=phase,variant=variant,replica=replica,model_sha256=sha(directory/'model.pt'),protocol_sha256=sha(pp),
                fit=metrics(model,training),trace=trace,new_physical_queries=0,elapsed_seconds=time.monotonic()-start,scientific_submission_ready=False)
            if phase=='diagnostic':report['diagnostic']=metrics(model,held)
            write(directory/'results.json',report)
    write(out/'results.json',dict(complete=True,replica=replica,models=6,protocol_sha256=sha(pp),new_physical_queries=0,scientific_submission_ready=False))


def audit(project,run,out,pp):
    protocol=json.loads(pp.read_text());groups,provenance=load_data(project,protocol)
    held_ids=set(protocol['diagnostic_source_ids']);fit=[g for g in groups if g['source_id'] not in held_ids];held=[g for g in groups if g['source_id'] in held_ids]
    streams={};results=[];checks=0
    for replica in [0,1]:
        assert json.loads((run/f's{replica}/results.json').read_text())['complete']
        for phase in ['diagnostic','full_fit']:
            training=fit if phase=='diagnostic' else groups
            for variant in protocol['variants']:
                directory=run/f's{replica}'/phase/variant;report=json.loads((directory/'results.json').read_text())
                assert report['complete'] and report['protocol_sha256']==sha(pp) and sha(directory/'model.pt')==report['model_sha256']
                saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
                assert saved['training_source_ids']==[g['source_id'] for g in training]
                trace=[r['source_id'] for r in report['trace']];key=(replica,phase)
                assert len(trace)==protocol['steps'] and set(trace)<=set(saved['training_source_ids'])
                if key in streams:assert streams[key]==trace
                streams[key]=trace
                model=make_model(variant,protocol);model.load_state_dict(saved['state_dict']);model.eval()
                equal(metrics(model,training),report['fit'])
                if phase=='diagnostic':equal(metrics(model,held),report['diagnostic'])
                with torch.no_grad():
                    for group in groups:
                        expected=predict(model,group);actions=group['actions'].clone();actions[:,0,2:]=actions[:,0,[3,2]]
                        actual=model(group['positions'][:,[1,0,3,2]],group['graphs'][:,[1,0,3,2]],group['numbers'],group['electronic'],actions)
                        torch.testing.assert_close(actual,-expected,atol=1e-9,rtol=1e-9);checks+=len(actual)
                results.append(dict(replica=replica,phase=phase,variant=variant,model_sha256=sha(directory/'model.pt'),results_sha256=sha(directory/'results.json'),
                    fit=report['fit']['parent_balanced'],diagnostic=report.get('diagnostic',{}).get('parent_balanced')))
    if out.exists():raise FileExistsError(out)
    write(out,dict(complete=True,protocol_sha256=sha(pp),sources=provenance,models=results,zero_diagnostic=metrics(None,held),
        exact_metric_replay=True,identical_parent_streams=True,reversal_checks=checks,new_physical_queries=0,scientific_submission_ready=False,
        scope='Internal parent-disjoint interaction prediction only. Nine original FIT parents have no eligible independent pair and remain reported separately. No trained sampler or generation gain has been demonstrated.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--replica',type=int);p.add_argument('--phase',choices=['train','audit'],required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.phase=='train':train(a.project,a.out,a.protocol,a.replica)
    else:audit(a.project,a.run,a.out,a.protocol)


if __name__=='__main__':main()
