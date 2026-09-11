#!/usr/bin/env python3
"""Replay local/rotation/exchange composition including quaternion auxiliaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii,terminal_exchange_actions,exchange_terminal_sites
from cfm_mol.terminal_rotation import terminal_rotation_actions
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--protocol',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=args.protocol or root/'research/evidence/angular_chemical_protocol_v1.json';protocol=json.loads(pp.read_text())
    physical=json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    kT=physical['kT_eV'];restraint=physical['restraint_eV_A2'];result=[]
    for replica in [0,1]:
        directory=args.run/f"{protocol['methods'][0]}_s{replica}";report=json.loads((directory/'results.json').read_text())
        assert report['complete'] and report['evaluation_protocol_sha256']==sha(pp)
        assert report['trace_sha256']==sha(directory/'trace.pt')
        data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        numbers=report['condition']['numbers'];charge=report['condition']['charge'];n=len(numbers)
        radii=covalent_radii(numbers);basis=centered_orthonormal_basis(n);states=data['states'];queries=data['query_trace']
        def graph(x):
            try:return infer_chemical_graph(x,numbers,charge)
            except ValueError:return None
        for j,q in enumerate(queries):
            assert q['raw_queries_before']==(0 if j==0 else queries[j-1]['raw_queries_after'])
            assert q['raw_queries_after']-q['raw_queries_before']==2*len(q['positions'])
        assert queries[-1]['raw_queries_after']==report['new_raw_queries']==report['requested_raw_queries']
        for i,s in enumerate(states):
            assert s['state_id']==i;q=queries[s['query_batch']];j=s['query_row'];x=q['positions'][j]
            torch.testing.assert_close(x,s['positions'],atol=1e-12,rtol=0);assert float(x.mean(0).abs().max())<1e-8
            e=.5*(q['raw_energy_eV'][j]+q['inverted_energy_eV'][j]);f=.5*(q['raw_force_eV_A'][j]-q['inverted_force_eV_A'][j])
            torch.testing.assert_close(e,s['energy_eV']);torch.testing.assert_close(f,s['force_eV_A'])
            torch.testing.assert_close(e+restraint/2*x.square().sum(),s['potential_eV'])
            torch.testing.assert_close((basis.T@((f-restraint*x)/kT)).flatten(),s['score'])
            g=graph(x);assert g is not None and g['connectivity_smiles']==s['graph']['connectivity_smiles']
            torch.testing.assert_close(g['bond_orders'],s['graph']['bond_orders'])
            assert terminal_exchange_actions(numbers,g['bond_orders'])==s['actions']
        rng=torch.Generator().manual_seed(protocol['evaluation_seeds'][replica])
        srng=torch.Generator().manual_seed(protocol['scale_seed']+replica)
        ids=list(data['history_state_ids'][0]);chains=len(ids);steps=protocol['uniform_total_budget_steps']
        assert len(data['transitions'])==steps*chains and len(data['history_state_ids'])==steps+1
        def clip(s):return s*min(1.,100/kT/max(float(s.norm()),1e-300))
        def natural_parameter(state,leaf,anchor,label):
            v=state['positions'][leaf]-state['positions'][anchor];radius=v.norm();u=v/radius
            force=state['force_eV_A']-restraint*state['positions'];force=force-force.mean(0)
            tangent=(force[leaf]-torch.dot(force[leaf],u)*u)*radius/kT
            tangent=tangent*min(1.,4*math.sqrt(label)/max(float(tangent.norm()),1e-300))
            return label*u+.5*tangent,u,radius
        def angular_logp(u,eta):
            k=float(eta.norm())
            return float(torch.dot(u,eta))+math.log(k)-math.log(4*math.pi)-math.log(math.sinh(k))
        summary={kind:dict(attempts=0,valid=0,accepted=0,difficult_parent_accepted=0) for kind in protocol['schedule']}
        for step in range(steps):
            kind=protocol['schedule'][step%len(protocol['schedule'])]
            choices=torch.randint(len(protocol['local_scales']),(chains,),generator=srng)
            assert choices.tolist()==data['scale_choice_history'][step]
            scales=torch.tensor(protocol['local_scales'],dtype=torch.float64)[choices]*kT**.5
            rows=data['transitions'][step*chains:(step+1)*chains]
            for i,row in enumerate(rows):
                assert row['kind']==kind and row['old_state_id']==ids[i]
                old=states[ids[i]];x=old['positions'];volume=0.;inverse=None
                if kind=='local':
                    std=float(scales[i]);assert row['proposal_std']==std
                    z=(basis.T@x).flatten();mean=z+.5*std**2*clip(old['score'])
                    noise=torch.randn(z.shape,dtype=z.dtype,generator=rng)
                    torch.testing.assert_close(noise,row['noise'],atol=0,rtol=0)
                    y=basis@(mean+std*noise).reshape(n-1,3)
                    torch.testing.assert_close(mean,row['forward_mean'])
                else:
                    actions=terminal_rotation_actions(numbers,old['graph']['bond_orders']) if kind!='exchange' else old['actions']
                    assert len(actions)==row['forward_count'] and len(actions)>0
                    choice=int(torch.randint(len(actions),(1,),generator=rng))
                    assert choice==row['choice_index'] and actions[choice]==row['action'];action=actions[choice]
                    if kind=='exchange':
                        y,vol,inverse=exchange_terminal_sites(x,radii,action);volume=float(vol)
                        assert inverse==row['inverse_action'];torch.testing.assert_close(vol,row['log_volume'])
                    elif kind=='rotation':
                        raw=torch.randn(4,dtype=torch.float64,generator=rng)
                        torch.testing.assert_close(raw,row['raw_quaternion'],atol=0,rtol=0)
                        torch.testing.assert_close(raw*raw.new_tensor([1.,-1.,-1.,-1.]),row['reverse_quaternion'])
                        w=raw[0]/raw.norm();v=raw[1:]/raw.norm();leaf,anchor=action;u=x[leaf]-x[anchor]
                        # Independent quaternion-vector formula, not the producer's matrix.
                        rotated=u+2*w*torch.linalg.cross(v,u)+2*torch.linalg.cross(v,torch.linalg.cross(v,u))
                        y=x.clone();y[leaf]=x[anchor]+rotated;y-=y.mean(0)
                        torch.testing.assert_close((y[leaf]-y[anchor]).norm(),u.norm(),atol=1e-10,rtol=0)
                    else:
                        cindex=int(torch.randint(3,(1,),generator=rng));label=protocol['concentrations'][cindex]
                        assert cindex==row['concentration_index'] and label==row['concentration']
                        leaf,anchor=action;eta,u,radius=natural_parameter(old,leaf,anchor,label)
                        torch.testing.assert_close(eta,row['forward_natural_parameter'])
                        torch.testing.assert_close(u,row['old_direction']);torch.testing.assert_close(radius,row['radius'])
                        uniform=torch.rand(1,dtype=torch.float64,generator=rng);noise=torch.randn((1,3),dtype=torch.float64,generator=rng)
                        torch.testing.assert_close(uniform,row['angular_random']['uniform'],atol=0,rtol=0)
                        torch.testing.assert_close(noise,row['angular_random']['gaussian'],atol=0,rtol=0)
                        k=eta.norm();mu=eta/k
                        cosine=1+torch.logaddexp(uniform.log(),torch.log1p(-uniform)-2*k)/k
                        tangent=noise[0]-torch.dot(noise[0],mu)*mu;tangent=tangent/tangent.norm()
                        direction=cosine[0]*mu+(1-cosine[0]**2).clamp_min(0).sqrt()*tangent
                        torch.testing.assert_close(direction,row['new_direction'],atol=1e-10,rtol=0)
                        assert abs(angular_logp(direction,eta)-row['angular_log_forward'])<1e-8
                        y=x.clone();y[leaf]=x[anchor]+radius*direction;y-=y.mean(0)
                        torch.testing.assert_close((y[leaf]-y[anchor]).norm(),radius,atol=1e-10,rtol=0)
                torch.testing.assert_close(y,row['proposal_positions'],atol=1e-10,rtol=0)
                g=graph(y);valid=g is not None
                if kind in ('rotation','force_rotation') and valid:valid=torch.equal(g['bond_orders'],old['graph']['bond_orders'])
                if kind=='exchange' and valid:valid=inverse in terminal_exchange_actions(numbers,g['bond_orders'])
                assert row['valid']==valid
                if valid:
                    new=states[row['new_state_id']];torch.testing.assert_close(y,new['positions'],atol=1e-10,rtol=0)
                    ratio=-float(new['potential_eV']-old['potential_eV'])/kT+volume
                    if kind=='local':
                        zz=(basis.T@y).flatten();reverse=zz+.5*std**2*clip(new['score'])
                        ratio+=float(((zz-mean).square().sum()-(z-reverse).square().sum())/(2*std**2))
                        assert row['log_forward_policy']==row['log_reverse_policy']==0
                    else:
                        reverse_count=(len(terminal_rotation_actions(numbers,new['graph']['bond_orders'])) if kind!='exchange' else len(new['actions']))
                        assert reverse_count==row['reverse_count'];correction=math.log(len(actions)/reverse_count)
                        assert abs(correction-row['action_log_ratio'])<1e-12;ratio+=correction
                        if kind=='force_rotation':
                            eta_reverse,_,r_reverse=natural_parameter(new,leaf,anchor,label)
                            torch.testing.assert_close(radius,r_reverse,atol=1e-10,rtol=0)
                            torch.testing.assert_close(eta_reverse,row['reverse_natural_parameter'])
                            log_reverse=angular_logp(u,eta_reverse)
                            assert abs(log_reverse-row['angular_log_reverse'])<1e-8
                            ratio+=log_reverse-row['angular_log_forward']
                    assert abs(ratio-row['log_acceptance_ratio'])<1e-7
                else:assert not row['accepted']
                summary[kind]['attempts']+=1;summary[kind]['valid']+=int(valid)
            logu=torch.rand(chains,dtype=torch.float64,generator=rng).log()
            for i,row in enumerate(rows):
                assert row['log_uniform']==float(logu[i])
                expected=row['valid'] and float(logu[i])<min(0.,row['log_acceptance_ratio'])
                assert row['accepted']==expected
                if expected:
                    ids[i]=row['new_state_id'];summary[kind]['accepted']+=1
                    summary[kind]['difficult_parent_accepted']+=int(i==2)
            assert ids==data['history_state_ids'][step+1]
            assert report['history'][step+1]['energy_eV']==[float(states[i]['energy_eV']) for i in ids]
            assert report['history'][step+1]['smiles']==[states[i]['graph']['connectivity_smiles'] for i in ids]
        torch.testing.assert_close(rng.get_state(),data['generator_state'],atol=0,rtol=0)
        torch.testing.assert_close(srng.get_state(),data['scale_generator_state'],atol=0,rtol=0)
        reference='CS(F)(F)(F)(F)F'
        result.append(dict(replica=replica,raw_queries=report['new_raw_queries'],total_raw_queries=report['history'][-1]['total_raw_queries'],
            moves=summary,raw_pair_and_score_replay=True,all_proposals_and_decisions_replayed=True,all_random_streams_replayed=True,
            quaternion_inverse_and_unit_volume_contract_checked=protocol['methods'][0]=='uniform_angular',
            S2_conditional_density_and_force_replay=protocol['methods'][0]=='force_angular',trace_sha256=report['trace_sha256'],
            reference_connectivity_first_hit_step=[next((h['step'] for h in report['history'] if h['smiles'][i]==reference),None) for i in range(chains)],
            final_energy_eV=report['history'][-1]['energy_eV'],oracle_requeried=False))
    out=dict(complete=True,rows=result,physical_baseline_only=True,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
