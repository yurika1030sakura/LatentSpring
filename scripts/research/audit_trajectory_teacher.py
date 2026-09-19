"""Replay the complete cached local-work teacher without new oracle queries."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.weighted_endpoints import EndpointDraw
from cfm_mol.weighted_endpoint_fm import checkpoint_source
from cfm_mol.trajectory_physical_teacher import candidates,velocity_target
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def audit(project,folder,spec,ph):
    done=json.loads((folder/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['bank_sha256']==sha(folder/'bank.pt')
    bank=torch.load(folder/'bank.pt',map_location='cpu',weights_only=False);assert bank['protocol_sha256']==ph and len(bank['rows'])==512
    parent=torch.load(project/spec['checkpoint'],map_location='cpu',weights_only=False);prior=prior_from_checkpoint(parent)
    data=torch.load(project/spec['data'],map_location='cpu',weights_only=False)['training'];conf=spec['trajectory'];queries=0;ess=[];fallback=0;clipped=dict(force=0,work=0)
    for slot,index in enumerate(spec['training_rows']):
        file=folder/'records'/f'c{slot}.pt';r=torch.load(file,map_location='cpu',weights_only=False);c=data[index]['condition'];n=c['n_atoms'];batch=conf['draws_per_composition']
        assert r['protocol_sha256']==ph and r['condition']==c and r['training_row']==index
        seed=conf['seed']*1000003+slot*100003;assert r['source_seed']==seed
        draws=[EndpointDraw('replay','none','none',c,np.zeros((n,3))) for _ in range(batch)]
        torch.testing.assert_close(r['source_positions'],checkpoint_source(prior,draws,seed),atol=0,rtol=0)
        assert r['final_positions'].shape==(batch,n,3) and torch.isfinite(r['final_positions']).all()
        assert [o['step'] for o in r['observed']]==conf['capture_steps']
        for obs in r['observed']:
            torch.testing.assert_close(obs['t'],torch.full((batch,),(obs['step']+.5)/spec['midpoint_steps']),atol=0,rtol=0)
        x=torch.cat([o['x'] for o in r['observed']]);v=torch.cat([o['velocity'] for o in r['observed']]);t=torch.cat([o['t'] for o in r['observed']])
        torch.testing.assert_close(r['x'],x,atol=0,rtol=0);torch.testing.assert_close(r['velocity'],v,atol=0,rtol=0);torch.testing.assert_close(r['t'],t,atol=0,rtol=0)
        h=(x+(1-t[:,None,None])*v).double();torch.testing.assert_close(r['anchor'],h,atol=0,rtol=0);m=len(h)
        ea,fa=r['anchor_raw_energy'],r['anchor_raw_force'];ep=r['proposal_raw_energy'];fp=r['proposal_raw_force']
        assert ea.shape==(2*m,) and fa.shape==(2*m,n,3) and ep.shape==(2*m*conf['particles'],) and fp.shape==(2*m*conf['particles'],n,3)
        assert all(torch.isfinite(q).all() for q in [ea,fa,ep,fp])
        energy=(ea[:m]+ea[m:])/2;force=(fa[:m]-fa[m:])/2
        source,proposal,sigma,shift=candidates(h,force,**{k:conf[k] for k in ['kT','particles','max_sigma','max_shift']},seed=seed+1)
        for key,value in [('source',source),('proposal',proposal),('sigma',sigma),('shift',shift)]:torch.testing.assert_close(r[key],value,atol=0,rtol=0)
        energy_y=((ep[:m*conf['particles']]+ep[m*conf['particles']:])/2).reshape(m,conf['particles'])
        valid=((proposal-h[:,None])**2).sum((-1,-2))<=conf['radius']**2
        lw=-(energy_y-energy[:,None])/conf['kT']+(((source-h[:,None])**2).sum((-1,-2))-((proposal-h[:,None])**2).sum((-1,-2)))/(2*sigma[:,None]**2)
        lw[~valid]=-torch.inf;eligible=valid.any(1);w=torch.zeros_like(lw);w[eligible]=lw[eligible].softmax(-1)
        moment=(w[:,:,None,None]*(proposal-h[:,None])).sum(1);moment[~eligible]=shift[~eligible];moment-=moment.mean(1,keepdim=True)
        torch.testing.assert_close(r['targets']['log_weights'],lw,atol=1e-10,rtol=1e-12)
        torch.testing.assert_close(r['targets']['weights'],w,atol=1e-10,rtol=1e-12)
        torch.testing.assert_close(r['targets']['work_shift'],moment,atol=1e-10,rtol=1e-12)
        ess.extend((1/w[eligible].square().sum(-1)).tolist());fallback+=int((~eligible).sum())
        for kind,delta in [('force',shift),('work',moment)]:
            label,cap=velocity_target(delta,t.double(),velocity_scale=spec['physical_connection']['velocity_scale'],gate_power=spec['physical_connection']['gate_power'])
            torch.testing.assert_close(r['velocity_targets'][kind],label,atol=1e-10,rtol=1e-12);torch.testing.assert_close(r['cap_scales'][kind],cap,atol=1e-10,rtol=1e-12)
            clipped[kind]+=int((cap<1.).sum())
        for k in range(m):
            row=bank['rows'][slot*m+k];assert row['condition']==c and row['training_row']==index and row['composition_slot']==slot and row['local_state']==k and row['record_sha256']==sha(file)
            for key,value in [('x',x[k]),('endpoint',h[k].float()),('t',t[k]),('force',r['velocity_targets']['force'][k].float()),('work',r['velocity_targets']['work'][k].float())]:torch.testing.assert_close(row[key],value,atol=0,rtol=0)
        assert r['oracle_queries']==2*m*(1+conf['particles']);queries+=r['oracle_queries']
    assert queries==done['oracle_queries']==9216 and done['new_fit_generator_outputs']==256
    result=dict(complete=True,protocol_sha256=ph,bank_sha256=sha(folder/'bank.pt'),all_states_retained=True,states=512,
        compositions=128,oracle_queries=queries,new_fit_generator_outputs=256,median_local_ess=float(np.median(ess)),
        force_fallback_states=fallback,clipped_target_states=clipped,scope=spec['target'])
    write(folder/'audit.json',result);return bank,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','teacher']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);_,result=audit(a.project,a.teacher,json.loads(a.protocol.read_text()),sha(a.protocol));print(json.dumps(result))


if __name__=='__main__':main()
