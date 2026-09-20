"""Fit normalization/mobility controls on cached TRAIN forces and validate them."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.physical_connection import make_physical_connection
from cfm_mol.atomwise_physical_connection import balanced_force_shift
from cfm_mol.trajectory_physical_teacher import velocity_target
from scripts.research.run_matched_connection import load_parent,make_context,fit,generate,score
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def prepare_targets(root,spec,ph,si,name,out):
    info=spec['banks'][si][name];file=root/info['path'];assert sha(file)==info['sha256'];bank=torch.load(file,map_location='cpu',weights_only=False)
    assert bank['protocol_sha256']==spec['original_protocol_sha256'] and len(bank['rows'])==512
    original=bank['rows'];balanced=[];diagnostics=[];folder=root/info['teacher_folder'];cfg=spec['head_configuration']
    for slot,index in enumerate(spec['training_rows']):
        recordfile=folder/'records'/f'c{slot}.pt';r=torch.load(recordfile,map_location='cpu',weights_only=False);digest=sha(recordfile)
        assert r['protocol_sha256']==spec['original_protocol_sha256'] and r['training_row']==index
        rows=[v for v in original if v['composition_slot']==slot];assert len(rows)==4 and all(v['record_sha256']==digest for v in rows)
        force=r['centered_even_force'];shift,mobility,scale=balanced_force_shift(force,r['shift'],force_floor=spec['force_floor_eV_A'])
        t=torch.cat([v['progress'] for v in r['observed']]).double();target,cap=velocity_target(shift,t,velocity_scale=cfg['velocity_scale'],gate_power=cfg['gate_power'])
        torch.testing.assert_close(shift.flatten(1).norm(dim=-1),r['shift'].flatten(1).norm(dim=-1),atol=1e-12,rtol=1e-10)
        assert ((force*shift).sum((1,2))>=-1e-12).all() and (cap==1).all()
        for k,row in enumerate(rows):
            assert row['local_state']==k;new=dict(row,force=target[k].float());balanced.append(new)
            torch.testing.assert_close(new['force'].norm(),row['force'].norm(),atol=5e-7,rtol=1e-5)
            z=torch.tensor(row['condition']['numbers']);h=z==1
            diagnostics.append(dict(composition_slot=slot,local_state=k,source_record_sha256=digest,
                force_dot_shift=float((force[k]*shift[k]).sum()),displacement_norm=float(shift[k].norm()),
                hydrogen_shift_norm=float(shift[k,h].norm()),original_hydrogen_shift_norm=float(r['shift'][k,h].norm()),
                heavy_shift_norm=float(shift[k,~h].norm()),original_heavy_shift_norm=float(r['shift'][k,~h].norm()),
                mobility=mobility[k].tolist(),scale=float(scale[k])))
    assert len(balanced)==512;out.mkdir(parents=True,exist_ok=False)
    for target,rows in [('global',original),('balanced',balanced)]:
        atomic_save(dict(protocol_sha256=ph,source_bank_sha256=info['sha256'],target=target,rows=rows),out/(target+'.pt'))
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,source_bank_sha256=info['sha256'],
        banks={kind:sha(out/(kind+'.pt')) for kind in ['global','balanced']},rows=diagnostics,new_oracle_queries=0,
        identical_displacement_norms=True,positive_linearized_descent=True))
    return dict(global_=original,balanced=balanced)


def restore_head(root,info):
    file=root/info['path'];assert sha(file)==info['sha256'];saved=torch.load(file,map_location='cpu',weights_only=False)
    head=make_physical_connection(**saved['configuration']).cuda().float();head.load_state_dict(saved['state_dict'],strict=True);head.eval()
    return head,saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);p.add_argument('--selection',type=Path);a=p.parse_args();torch.set_num_threads(2)
    root=a.project.resolve();a.out=a.out.resolve();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);si=a.seed_index;assert spec['frozen']
    assert sha(root/spec['original_protocol'])==spec['original_protocol_sha256'] and sha(root/spec['original_selection'])==spec['original_selection_sha256']
    a.out.mkdir(parents=True,exist_ok=False);source=base.HarmonicSource();reports=[];heads={};tick=time.perf_counter();new_steps=0
    if a.selection:
        selection=json.loads(a.selection.read_text());assert selection['complete'] and selection['advance'] and selection['protocol_sha256']==ph
        phase='test';rows=spec['test_rows'];count=spec['test_samples'];seed=spec['evaluation_seeds'][si]
    else:phase='validation';rows=spec['validation_rows'];count=spec['validation_samples'];seed=spec['validation_seeds'][si]
    for name,arm in spec['parents'][si].items():
        model=load_parent(root,arm);context=make_context(arm,source);heads[name]={}
        if not a.selection:targets=prepare_targets(root,spec,ph,si,name,a.out/name/'targets')
        for variant,definition in spec['variants'].items():
            config=definition['configuration']
            if a.selection:
                info=selection['heads'][str(si)][name][variant];head,saved=restore_head(root,info)
                strengths=list(selection['test_strengths'][name][variant])
            elif definition['reuse']:
                old=spec['reused_heads'][str(si)][name];info={k:old[k] for k in ['path','sha256']};head,saved=restore_head(root,info)
                assert saved['protocol_sha256']==spec['original_protocol_sha256'];strengths=spec['strengths']
            else:
                bank=targets['global_' if definition['target']=='global' else 'balanced'];directory=a.out/name/variant/'fit'
                head=fit(dict(spec,head_configuration=config),ph,si,bank,directory);new_steps+=spec['steps']
                file=directory/f'step_{spec["steps"]}.pt';info=dict(path=str(file.relative_to(root)),sha256=sha(file));strengths=[v for v in spec['strengths'] if v>0]
            assert head.configuration==config and sum(p.numel() for p in head.parameters())==7106
            info=dict(info,configuration=config,target=definition['target'],head_state_sha256=base.state_hash(head));heads[name][variant]=info
            if strengths:
                directory=a.out/name/variant/'generation';report=generate(spec,ph,si,name,model,source,context,head,rows,seed,count,strengths,directory,
                    method_prefix=name+'_'+variant)
                report.update(variant=variant,configuration=config,target=definition['target'],head_checkpoint_sha256=info['sha256'],strengths=strengths)
                write(directory/'generation.json',report);reports.append((directory,report))
            del head;gc.collect();torch.cuda.empty_cache()
        assert base.state_hash(model)==spec['reused_heads'][str(si)][name]['model_state_sha256']
        del model,context;gc.collect();torch.cuda.empty_cache()
    summary=score(spec,ph,si,reports,a.out/'xtb');generated=sum(r['new_neural_outputs'] for f,r in reports)
    if not a.selection:assert generated==spec['expected_new_validation_outputs']//2 and new_steps==120000
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,phase=phase,seed_index=si,heads=heads,summary=summary,
        generation_reports=[dict(path=str((f/'generation.json').relative_to(root)),sha256=sha(f/'generation.json')) for f,r in reports],
        new_neural_outputs=generated,new_gfn2_attempts=sum(v['attempted'] for v in summary.values()),new_optimizer_steps=new_steps,
        new_fit_outputs=0,new_esen_queries=0,seconds=time.perf_counter()-tick,selection_sha256=sha(a.selection) if a.selection else None))


if __name__=='__main__':main()
