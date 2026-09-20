"""Apply an independently learned head to the other generator without fitting."""
import argparse,copy,gc,json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from scripts.research.run_matched_connection import load_parent,make_context,generate,score
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=range(5),required=True);a=p.parse_args();root=a.project.resolve();si=a.fit;torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proposal=json.loads(a.protocol.read_text());assert proposal['frozen'] and sha(root/proposal['parent_campaign'])==proposal['parent_campaign_sha256']
    original=root/f'runs/seed_replication_v1/evaluation/s{si}';done=json.loads((original/'complete.json').read_text());assert done['complete'] and done['campaign_sha256']==proposal['parent_campaign_sha256']
    spec=json.loads((original/'resolved_protocol.json').read_text());spec=copy.deepcopy(spec);spec.update(format='cross_generator_head_resolved_v1',cross_protocol_sha256=sha(a.protocol),source_study_sha256=sha(original/'complete.json'))
    a.out.mkdir(parents=True,exist_ok=False);protocol=a.out/'resolved_protocol.json';write(protocol,spec);ph=sha(protocol);source=base.HarmonicSource();reports=[];heads={}
    for family in ['fm','gaga']:
        other='gaga' if family=='fm' else 'fm';arm=spec['parents'][si][family];model=load_parent(root,arm);context=make_context(arm,source)
        info=done['heads'][other];head,_=restore_head(root,info);head_hash=base.state_hash(head);directory=a.out/'parents'/family
        report=generate(spec,ph,si,family,model,source,context,head,spec['test_rows'],spec['evaluation_seeds'][si],16,[4.],directory,method_prefix=family+'_cross')
        assert report['head_state_sha256']==head_hash;reports.append((directory,report));heads[family]=dict(**info,training_parent=other,head_state_sha256=head_hash)
        old=json.loads((original/'parents'/family/'generation.json').read_text())
        for i,row in enumerate(report['rows']):
            original_row=next(r for r in old['rows'] if r['method']==family+'_a0' and r['condition_index']==i)
            left=torch.load(directory/row['file'],map_location='cpu',weights_only=False);right=torch.load(original/'parents'/family/original_row['file'],map_location='cpu',weights_only=False)
            torch.testing.assert_close(left['initial_positions'],right['initial_positions'],atol=0,rtol=0)
        del model,head,context;gc.collect();torch.cuda.empty_cache()
    score(spec,ph,si,reports,a.out/'xtb');write(a.out/'complete.json',dict(complete=True,protocol_sha256=sha(a.protocol),resolved_protocol_sha256=ph,fit=si,
        source_study=str(original.relative_to(root)),source_study_sha256=sha(original/'complete.json'),heads=heads,new_parent_trajectories=2048,new_gfn2_attempts=2048,new_optimizer_steps=0,new_esen_queries=0,
        physical_results_sha256=sha(a.out/'xtb/results.json')))

if __name__=='__main__':main()
