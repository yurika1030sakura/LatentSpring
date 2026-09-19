"""Cache actual parent states and query local force/work targets on TRAIN only."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.weighted_endpoints import EndpointDraw
from cfm_mol.weighted_endpoint_fm import dgl_batch,checkpoint_source
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.trajectory_physical_teacher import candidates,targets,velocity_target
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from scripts.research.run_gaga_feedback import atomic_save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);conf=spec['trajectory'];torch.set_num_threads(2)
    assert spec['frozen'] and sha(a.project/spec['data'])==spec['data_sha256']
    assert sha(a.project/spec['checkpoint'])==spec['checkpoint_sha256'] and sha(a.project/spec['config'])==spec['config_sha256']
    data=torch.load(a.project/spec['data'],map_location='cpu',weights_only=False)['training']
    state=torch.load(a.project/spec['checkpoint'],map_location='cpu',weights_only=False);recipe=state['research_protocol']
    cfg=read_config_file(a.project/spec['config']);cfg['mol_fm'].pop('bgfm',None)
    assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True)
    patch_smooth_geometry(model,recipe.get('geometry_softening',0.));model.cuda().float().eval().requires_grad_(False)
    prior=prior_from_checkpoint(state);a.out.mkdir(parents=True,exist_ok=False);(a.out/'records').mkdir();start=time.perf_counter();rows=[]
    oc=spec['oracle'];worker=Path(__file__).resolve().parent/'oracle_worker.py'
    assert sha(worker)==oc['oracle_worker_sha256'] and sha(oc['oracle_checkpoint'])==oc['oracle_sha256']
    c=data[spec['training_rows'][0]]['condition']
    with EnergyOracle(oc['oracle_interpreter'],worker,oc['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],
            spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16,timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        write(a.out/'handshake.json',oracle.handshake)
        for slot,index in enumerate(spec['training_rows']):
            c=data[index]['condition'];n=c['n_atoms'];batch=conf['draws_per_composition'];seed=conf['seed']*1000003+slot*100003
            draws=[EndpointDraw('train_trajectory','none','none',c,np.zeros((n,3))) for _ in range(batch)]
            g,nbi,uem=dgl_batch(draws,cfg,torch.device('cuda'),model_kT_eV=1.);g.ndata['has_reference_geometry'].zero_()
            x0=checkpoint_source(prior,draws,seed);observed=[]
            def capture(step,x,v,t):
                if step in conf['capture_steps']:
                    observed.append(dict(step=step,x=x.reshape(batch,n,3).cpu(),velocity=v.reshape(batch,n,3).cpu(),t=t.cpu()))
            final=sample_clamped_flow(model,g,nbi,uem,x0=x0.cuda().float(),n_ode_steps=spec['midpoint_steps'],
                terminal_time=1.,parameterization='displacement',midpoint_observer=capture).reshape(batch,n,3).cpu()
            assert len(observed)==len(conf['capture_steps'])
            x=torch.cat([r['x'] for r in observed]);v=torch.cat([r['velocity'] for r in observed]);t=torch.cat([r['t'] for r in observed])
            # Float arithmetic matches the live head's provisional endpoint.
            anchor=(x+(1-t[:,None,None])*v).double();count=len(anchor)
            oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity']);before=oracle.evaluated
            ea,fa=oracle.evaluate_chunked(torch.cat([anchor,-anchor]),max_request=16)
            energy=(ea[:count]+ea[count:])/2;force=(fa[:count]-fa[count:])/2
            source,proposal,sigma,shift=candidates(anchor,force,**{k:conf[k] for k in ['kT','particles','max_sigma','max_shift']},seed=seed+1)
            flat=proposal.flatten(0,1);ep,fp=oracle.evaluate_chunked(torch.cat([flat,-flat]),max_request=16)
            ep_mean=((ep[:len(flat)]+ep[len(flat):])/2).reshape(count,conf['particles'])
            result=targets(anchor,source,proposal,sigma,shift,energy,ep_mean,kT=conf['kT'],radius=conf['radius'])
            label={};cap={}
            for kind in ['force','work']:
                label[kind],cap[kind]=velocity_target(result[kind+'_shift'],t.double(),
                    velocity_scale=spec['physical_connection']['velocity_scale'],gate_power=spec['physical_connection']['gate_power'])
            record=dict(condition=c,training_row=index,source_seed=seed,source_positions=x0,final_positions=final,
                observed=observed,x=x,velocity=v,t=t,anchor=anchor,source=source,proposal=proposal,sigma=sigma,shift=shift,
                anchor_raw_energy=ea,anchor_raw_force=fa,proposal_raw_energy=ep,proposal_raw_force=fp,
                targets=result,velocity_targets=label,cap_scales=cap,protocol_sha256=ph,oracle_queries=oracle.evaluated-before)
            file=a.out/'records'/f'c{slot}.pt';atomic_save(record,file)
            for k in range(count):
                rows.append(dict(condition=c,training_row=index,composition_slot=slot,local_state=k,
                    x=x[k],endpoint=anchor[k].float(),t=t[k],force=label['force'][k].float(),work=label['work'][k].float(),record_sha256=sha(file)))
            progress=dict(compositions=slot+1,states=len(rows),oracle_queries=oracle.evaluated,seconds=time.perf_counter()-start)
            write(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        total=oracle.evaluated;assert total==oracle.requested_evaluations==9216
    bank=a.out/'bank.pt';atomic_save(dict(protocol_sha256=ph,rows=rows),bank)
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,bank_sha256=sha(bank),states=len(rows),
        compositions=128,new_fit_generator_outputs=256,oracle_queries=total,seconds=time.perf_counter()-start,
        teacher_scope=spec['target'],all_parent_states_retained=True))


if __name__=='__main__':main()
