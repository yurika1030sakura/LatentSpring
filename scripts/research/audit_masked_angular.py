#!/usr/bin/env python3
"""Replay frozen masked kernels with saved oracle outputs and check their MH ratios."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.terminal_rotation import uniform_internal_transition,terminal_rotation_actions
from cfm_mol.masked_angular_guide import MaskedAngularGuide,masked_angular_context
from cfm_mol.masked_angular_sampler import masked_angular_transition


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def equal(actual,saved):
    if isinstance(saved,torch.Tensor):
        torch.testing.assert_close(actual,saved,atol=1e-8 if saved.is_floating_point() else 0,rtol=1e-9 if saved.is_floating_point() else 0)
    elif isinstance(saved,dict):
        assert set(actual)==set(saved)
        for k in saved:equal(actual[k],saved[k])
    elif isinstance(saved,(list,tuple)):
        assert len(actual)==len(saved)
        for a,b in zip(actual,saved):equal(a,b)
    elif isinstance(saved,float):assert actual==saved or abs(actual-saved)<1e-7
    else:assert actual==saved


class ReplayOracle:
    def __init__(self,queries):
        self.queries=queries;self.index=0;self.evaluated=0;self.maximum_position_error=0.
    def evaluate_chunked(self,x,max_request):
        q=self.queries[self.index];expected=torch.cat([q['positions'],-q['positions']])
        self.maximum_position_error=max(self.maximum_position_error,float((x-expected).abs().max()))
        torch.testing.assert_close(x,expected,atol=1e-9,rtol=0)
        assert self.evaluated==q['raw_queries_before'];self.evaluated+=len(x);assert self.evaluated==q['raw_queries_after'];self.index+=1
        return torch.cat([q['raw_energy_eV'],q['inverted_energy_eV']]),torch.cat([q['raw_force_eV_A'],q['inverted_force_eV_A']])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=args.protocol or root/'research/evidence/masked_angular_evaluation_protocol_v1.json';protocol=json.loads(pp.read_text())
    physical=json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    summaries=[]
    for method in protocol['methods']:
        for replica in [0,1]:
            directory=args.run/f'{method}_s{replica}';report=json.loads((directory/'results.json').read_text())
            assert report['complete'] and report['evaluation_protocol_sha256']==sha(pp)
            assert report['trace_sha256']==sha(directory/'trace.pt')
            data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
            oracle=ReplayOracle(data['query_trace'])
            target=ChemicalTarget(oracle,report['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
            model=None
            if method!='masked_zero':
                name=method.removeprefix('masked_');path=args.models/f'{name}_s{replica}/model.pt'
                assert sha(path)==report['policy_sha256']==protocol['frozen_model_sha256'][name][replica]
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                model=MaskedAngularGuide(**checkpoint['configuration']).double();model.load_state_dict(checkpoint['state_dict']);model.eval()
            initial=[data['states'][i]['positions'] for i in data['history_state_ids'][0]]
            states=target.evaluate([target.coordinate_state(x) for x in initial],phase='initial')
            generator=torch.Generator().manual_seed(protocol['evaluation_seeds'][replica])
            sg=torch.Generator().manual_seed(protocol['scale_seed']+replica)
            search_index=0;chains=len(states);totals={k:dict(attempts=0,valid=0,accepted=0) for k in protocol['schedule']}
            for step in range(protocol['uniform_total_budget_steps']):
                assert [s['state_id'] for s in states]==data['history_state_ids'][step]
                choices=torch.randint(len(protocol['local_scales']),(chains,),generator=sg)
                assert choices.tolist()==data['scale_choice_history'][step]
                scales=torch.tensor(protocol['local_scales'],dtype=torch.float64)[choices]*target.kT**.5
                kind=protocol['schedule'][step%4];old=list(states)
                if kind=='local':
                    states,rows=target.transition(states,policy=None,generator=generator,proposal_std=scales,
                        phase=f'evaluation_{step}',local_only=True)
                    for row in rows:row['kind']='local'
                elif kind=='exchange':
                    states,rows=uniform_internal_transition(target,states,kind=kind,generator=generator,phase=f'evaluation_{step}')
                else:
                    states,rows,search=masked_angular_transition(target,states,model,max_trials=protocol['capped_direction_trials'],
                        generator=generator,phase=f'evaluation_{step}',proposal=protocol.get('proposal','uniform'))
                    equal(search,data['masked_searches'][search_index]);search_index+=1
                    # Independently check the physical and unnormalized angular
                    # score ratio; no hidden normalization value is supplied.
                    for i,row in enumerate(rows):
                        if not row['valid']:assert row['exhausted'] and not row['accepted'];continue
                        new=target.states[row['new_state_id']];leaf,anchor=row['action']
                        x=old[i]['positions'];y=new['positions'];roots=torch.tensor([[leaf,anchor]])
                        for before,after in zip(masked_angular_context(x[None],roots),masked_angular_context(y[None],roots)):
                            torch.testing.assert_close(before,after,atol=1e-10,rtol=0)
                        assert torch.equal(old[i]['graph']['bond_orders'],new['graph']['bond_orders'])
                        ux=x[leaf]-x[anchor];ux=ux/ux.norm();uy=y[leaf]-y[anchor];uy=uy/uy.norm()
                        eta=search['eta'][i];a=search['matrix'][i]
                        sx=float(torch.dot(eta,ux)+ux@a@ux);sy=float(torch.dot(eta,uy)+uy@a@uy)
                        count=len(terminal_rotation_actions(target.numbers,new['graph']['bond_orders']))
                        ratio=-float(new['potential_eV']-old[i]['potential_eV'])/target.kT+sx-sy+math.log(row['forward_count']/count)
                        assert abs(ratio-row['log_acceptance_ratio'])<1e-7
                        assert row['accepted']==(row['log_uniform']<min(0.,ratio))
                equal(rows,data['transitions'][step*chains:(step+1)*chains])
                assert [s['state_id'] for s in states]==data['history_state_ids'][step+1]
                assert report['history'][step+1]['smiles']==[s['graph']['connectivity_smiles'] for s in states]
                for row in rows:
                    totals[kind]['attempts']+=1;totals[kind]['valid']+=row['valid'];totals[kind]['accepted']+=row['accepted']
            assert oracle.index==len(oracle.queries) and oracle.evaluated==report['new_raw_queries']
            assert search_index==len(data['masked_searches'])
            equal(generator.get_state(),data['generator_state']);equal(sg.get_state(),data['scale_generator_state'])
            reference='CS(F)(F)(F)(F)F'
            summaries.append(dict(method=method,replica=replica,raw_queries=oracle.evaluated,total_raw_queries=report['history'][-1]['total_raw_queries'],
                full_producer_replay=True,independent_masked_MH_ratio_check=True,all_random_streams_replayed=True,
                maximum_replay_position_error_A=oracle.maximum_position_error,oracle_requeried=False,
                angular_search_trials=report['angular_search_trials'],angular_geometry_checks=report['angular_geometry_checks'],
                angular_search_exhaustions=report['angular_search_exhaustions'],moves=totals,
                reference_connectivity_first_hit_step=[next((h['step'] for h in report['history'] if h['smiles'][i]==reference),None) for i in range(chains)],
                final_energy_eV=report['history'][-1]['energy_eV'],seconds=report['seconds'],trace_sha256=report['trace_sha256']))
    result=dict(complete=True,rows=summaries,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
