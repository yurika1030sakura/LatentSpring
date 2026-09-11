#!/usr/bin/env python3
"""Independent replay of raw pairs, complete escort paths and all random streams."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from cfm_mol.chemical_moves import infer_chemical_graph,terminal_exchange_actions,covalent_radii,exchange_terminal_sites
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.chemical_path_guide import exchanged_bond_graph


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/escorted_chemical_protocol_v1.json';protocol=json.loads(pp.read_text())
    physical=json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    kT=physical['kT_eV'];restraint=physical['restraint_eV_A2'];summaries=[]
    for replica in [0,1]:
        directory=args.run/f'replica_{replica}';report=json.loads((directory/'results.json').read_text())
        assert report['complete'] and report['protocol_sha256']==sha(pp)
        assert report['trace_sha256']==sha(directory/'trace.pt')
        data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        condition=report['condition'];numbers=condition['numbers'];n=len(numbers);basis=centered_orthonormal_basis(n)
        radii=covalent_radii(numbers);states=data['states'];queries=data['query_trace'];graphs={}
        def graph(x):
            try:return infer_chemical_graph(x,numbers,condition['charge'])
            except ValueError:return None
        for i,q in enumerate(queries):
            assert q['raw_queries_before']==(0 if i==0 else queries[i-1]['raw_queries_after'])
            assert q['raw_queries_after']-q['raw_queries_before']==2*len(q['positions'])
        assert queries[-1]['raw_queries_after']==report['new_raw_queries']==report['requested_raw_queries']
        for i,s in enumerate(states):
            assert s['state_id']==i
            q=queries[s['query_batch']];j=s['query_row'];x=q['positions'][j]
            torch.testing.assert_close(x,s['positions'],atol=1e-12,rtol=0)
            assert float(x.mean(0).abs().max())<1e-8
            energy=.5*(q['raw_energy_eV'][j]+q['inverted_energy_eV'][j])
            force=.5*(q['raw_force_eV_A'][j]-q['inverted_force_eV_A'][j])
            torch.testing.assert_close(energy,s['energy_eV']);torch.testing.assert_close(force,s['force_eV_A'])
            torch.testing.assert_close(energy+restraint/2*x.square().sum(),s['potential_eV'])
            torch.testing.assert_close((basis.T@((force-restraint*x)/kT)).flatten(),s['score'])
            g=graph(x);graphs[i]=g
            if s['graph'] is not None:
                assert g is not None and g['connectivity_smiles']==s['graph']['connectivity_smiles']
                torch.testing.assert_close(g['bond_orders'],s['graph']['bond_orders'])
                assert terminal_exchange_actions(numbers,g['bond_orders'])==s['actions']
        chains=len(data['history_state_ids'][0]);dims=3*(n-1)
        rng=torch.Generator().manual_seed(protocol['transition_seeds'][replica])
        action_rng=torch.Generator().manual_seed(protocol['action_seed']+replica)
        scale_rng=torch.Generator().manual_seed(protocol['scale_seed']+replica)
        ids=list(data['history_state_ids'][0]);cursor=0;scale_cursor=0
        def clipped(score):
            norm=score.norm(dim=-1,keepdim=True)
            return score*torch.minimum(torch.ones_like(norm),(100/kT)/norm.clamp_min(1e-300))
        def local_batch(cycle,side):
            nonlocal ids,cursor,scale_cursor
            choices=torch.randint(len(protocol['local_scales']),(chains,),generator=scale_rng)
            assert data['scale_choices'][scale_cursor]==dict(cycle=cycle,side=side,indices=choices.tolist())
            scale_cursor+=1
            scales=torch.tensor(protocol['local_scales'],dtype=torch.float64)[choices]*kT**.5
            rows=data['local_transitions'][cursor:cursor+chains];cursor+=chains
            assert len(rows)==chains
            for i,r in enumerate(rows):
                assert r['old_state_id']==ids[i] and r['action_index']==0
                assert r['proposal_std']==float(scales[i]);std=float(scales[i]);old=states[ids[i]]
                z=(basis.T@old['positions']).flatten();mean=z+.5*std**2*clipped(old['score'][None])[0]
                noise=torch.randn((dims,),dtype=torch.float64,generator=rng)
                torch.testing.assert_close(noise,r['noise'],atol=0,rtol=0)
                y=basis@(mean+std*noise).reshape(n-1,3)
                torch.testing.assert_close(mean,r['forward_mean']);torch.testing.assert_close(y,r['proposal_positions'],atol=1e-10,rtol=0)
                assert (graph(y) is not None)==r['valid']
                assert r['log_forward_policy']==r['log_reverse_policy']==0.
                if r['valid']:
                    new=states[r['new_state_id']];w=(basis.T@y).flatten()
                    torch.testing.assert_close(y,new['positions'],atol=1e-10,rtol=0)
                    rm=w+.5*std**2*clipped(new['score'][None])[0]
                    gaussian=((w-mean).square().sum()-(z-rm).square().sum())/(2*std**2)
                    ratio=-(new['potential_eV']-old['potential_eV'])/kT+gaussian
                    torch.testing.assert_close(ratio,r['base_log_ratio'],atol=1e-7,rtol=1e-9)
                    assert abs(float(ratio)-r['log_acceptance_ratio'])<1e-7
                else:assert r['new_state_id']==-1 and math.isinf(r['log_acceptance_ratio']) and r['log_acceptance_ratio']<0
            logu=torch.rand(chains,dtype=torch.float64,generator=rng).log()
            assert logu.tolist()==[r['log_uniform'] for r in rows]
            for i,r in enumerate(rows):
                assert r['accepted']==(r['valid'] and float(logu[i])<min(0.,r['log_acceptance_ratio']))
                if r['accepted']:ids[i]=r['new_state_id']
        def gaussian(u,v,su,sv,std):
            mf=u+.5*std**2*clipped(su);mr=v+.5*std**2*clipped(sv)
            normalizer=dims*math.log(std*math.sqrt(2*math.pi))
            lf=-.5*((v-mf)/std).square().sum(1)-normalizer
            lr=-.5*((u-mr)/std).square().sum(1)-normalizer
            return lf,lr,mf,mr
        paths_outside=0
        for cycle,path in enumerate(data['paths']):
            local_batch(cycle,'pre');assert path['cycle']==cycle and path['vertex_state_ids'][0]==ids
            k=protocol['steps_per_side'];std=protocol['path_scale']*kT**.5
            assert path['std']==std and path['steps_per_side']==k and path['max_score_norm']==100/kT
            assert path['map_source']==k and path['map_destination']==k+1
            assert path['vertices'].shape==(2*k+2,chains,dims)
            actions=[]
            for i,d in enumerate(path['decisions']):
                aa=states[ids[i]]['actions'];chosen=int(torch.randint(len(aa),(1,),generator=action_rng))
                assert chosen==d['choice_index'] and aa[chosen]==d['action'] and len(aa)==d['forward_count']
                actions.append(aa[chosen])
            for vertex,vertex_ids in enumerate(path['vertex_state_ids']):
                for i,state_id in enumerate(vertex_ids):
                    s=states[state_id]
                    torch.testing.assert_close((basis.T@s['positions']).flatten(),path['vertices'][vertex,i],atol=1e-10,rtol=0)
                    torch.testing.assert_close(-s['potential_eV']/kT,path['smooth_log_values'][vertex,i])
                    torch.testing.assert_close(s['score'],path['smooth_scores'][vertex,i])
            paths_outside+=sum(any(graphs[v[i]] is None for v in path['vertex_state_ids'][1:-1]) for i in range(chains))
            volumes=[]
            for i,a in enumerate(actions):
                before=basis@path['vertices'][k,i].reshape(n-1,3)
                after,volume,inverse=exchange_terminal_sites(before,radii,a)
                torch.testing.assert_close((basis.T@after).flatten(),path['vertices'][k+1,i],atol=1e-10,rtol=0)
                recovered,rv,_=exchange_terminal_sites(after,radii,inverse)
                torch.testing.assert_close(recovered,before,atol=1e-10,rtol=0)
                torch.testing.assert_close(volume+rv,torch.zeros_like(volume),atol=1e-10,rtol=0)
                assert inverse==path['decisions'][i]['inverse_action'];volumes.append(volume)
            torch.testing.assert_close(torch.stack(volumes),path['log_volume'])
            pairs=[(i,i+1) for i in range(2*k+1) if i!=k]
            assert [(e['source'],e['destination']) for e in path['edges']]==pairs
            total=torch.zeros(chains,dtype=torch.float64)
            for e in path['edges']:
                i,j=e['source'],e['destination'];u,v=path['vertices'][i],path['vertices'][j]
                noise=torch.randn((chains,dims),dtype=torch.float64,generator=rng)
                torch.testing.assert_close(noise,e['noise'],atol=0,rtol=0)
                lf,lr,mf,mr=gaussian(u,v,path['smooth_scores'][i],path['smooth_scores'][j],std)
                torch.testing.assert_close(v,mf+std*noise,atol=1e-10,rtol=0)
                for actual,saved in [(lf,e['log_forward']),(lr,e['log_reverse']),(mf,e['forward_mean']),(mr,e['reverse_mean'])]:
                    torch.testing.assert_close(actual,saved,atol=1e-7,rtol=1e-9)
                total+=lr-lf
            torch.testing.assert_close(total,path['path_log_ratio'],atol=1e-7,rtol=1e-9)
            base=path['smooth_log_values'][-1]-path['smooth_log_values'][0]+total+path['log_volume']
            torch.testing.assert_close(base,path['smooth_log_acceptance_ratio'],atol=1e-7,rtol=1e-9)
            # Explicit reverse path action, including the opposite map volume.
            vertices=path['vertices'].flip(0);scores=path['smooth_scores'].flip(0);reverse_total=torch.zeros_like(total)
            for i,j in pairs:
                lf,lr,_,_=gaussian(vertices[i],vertices[j],scores[i],scores[j],std);reverse_total+=lr-lf
            torch.testing.assert_close(reverse_total,-total,atol=1e-7,rtol=1e-9)
            logu=torch.rand(chains,dtype=torch.float64,generator=rng).log()
            for i,d in enumerate(path['decisions']):
                end_id=path['vertex_state_ids'][-1][i];g=graphs[end_id]
                assert d['old_state_id']==ids[i] and d['proposed_state_id']==end_id
                reverse_actions=terminal_exchange_actions(numbers,g['bond_orders']) if g is not None else []
                valid=g is not None and d['inverse_action'] in reverse_actions
                assert valid==d['valid'] and d['log_uniform']==float(logu[i])
                if valid:
                    correction=math.log(d['forward_count']/len(reverse_actions));ratio=float(base[i])+correction
                    assert d['reverse_count']==len(reverse_actions) and abs(d['action_log_ratio']-correction)<1e-12
                    assert abs(d['log_acceptance_ratio']-ratio)<1e-7
                    assert d['accepted']==(float(logu[i])<min(0.,ratio))
                else:assert not d['accepted']
                if d['accepted']:ids[i]=end_id
            local_batch(cycle,'post');assert ids==data['history_state_ids'][cycle+1]
            assert report['history'][cycle+1]['smiles']==[graphs[i]['connectivity_smiles'] for i in ids]
            assert report['history'][cycle+1]['energy_eV']==[float(states[i]['energy_eV']) for i in ids]
        assert len(data['paths'])==protocol['cycles'] and cursor==len(data['local_transitions'])
        assert scale_cursor==len(data['scale_choices'])
        for g,key in [(rng,'generator_state'),(action_rng,'action_generator_state'),(scale_rng,'scale_generator_state')]:
            torch.testing.assert_close(g.get_state(),data[key],atol=0,rtol=0)
        reference='CS(F)(F)(F)(F)F'
        summaries.append(dict(replica=replica,raw_queries=report['new_raw_queries'],total_raw_queries=report['history'][-1]['total_raw_queries'],
            raw_pair_and_score_reconstruction=True,full_path_reverse_ratio_replay=True,all_random_streams_replayed=True,
            paths_with_outside_support_intermediate=paths_outside,escorted_attempts=report['escorted_attempts'],
            escorted_valid=report['escorted_valid'],escorted_accepted=report['escorted_accepted'],
            reference_connectivity_first_hit_cycle=[next((h['cycle'] for h in report['history'] if h['smiles'][i]==reference),None) for i in range(chains)],
            final_energy_eV=report['history'][-1]['energy_eV'],trace_sha256=report['trace_sha256'],
            oracle_requeried=False,independent_physical_accuracy_certified=False))
    result=dict(complete=True,rows=summaries,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
