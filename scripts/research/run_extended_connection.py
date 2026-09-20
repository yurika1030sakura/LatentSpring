"""Evaluate stronger learned corrections without refitting either generator."""
import argparse,gc,json,time
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.physical_connection import make_physical_connection
from scripts.research.run_matched_connection import load_parent,make_context,generate,score
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);p.add_argument('--selection',type=Path);a=p.parse_args();torch.set_num_threads(2)
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);si=a.seed_index;assert spec['frozen']
    oldpath=a.project/spec['original_selection'];assert sha(oldpath)==spec['original_selection_sha256'];old=json.loads(oldpath.read_text())
    for file,digest in old['validation_provenance'].items():assert sha(a.project/file)==digest
    if a.selection:
        selection=json.loads(a.selection.read_text());assert selection['complete'] and selection['advance'] and selection['protocol_sha256']==ph
        phase='test';rows=spec['test_rows'];seed=spec['evaluation_seeds'][si];count=spec['test_samples']
    else:phase='validation';rows=spec['validation_rows'];seed=spec['validation_seeds'][si];count=spec['validation_samples']
    a.out.mkdir(parents=True,exist_ok=False);source=base.HarmonicSource();reports=[];tick=time.perf_counter()
    for name,arm in spec['parents'][si].items():
        model=load_parent(a.project,arm);assert base.state_hash(model)==spec['heads'][str(si)][name]['model_state_sha256'];context=make_context(arm,source)
        info=spec['heads'][str(si)][name];file=a.project/info['path'];assert sha(file)==info['sha256'];saved=torch.load(file,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==spec['original_protocol_sha256'];head=make_physical_connection(**saved['configuration']).cuda().float();head.load_state_dict(saved['state_dict'],strict=True);head.eval()
        strengths=[1.,selection['strengths'][name]] if a.selection else spec['new_strengths']
        folder=a.out/name/'generation';report=generate(spec,ph,si,name,model,source,context,head,rows,seed,count,strengths,folder);reports.append((folder,report))
        del model,head,context;gc.collect();torch.cuda.empty_cache()
    summary=score(spec,ph,si,reports,a.out/'xtb')
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,phase=phase,seed_index=si,summary=summary,
        new_neural_outputs=sum(r['new_neural_outputs'] for f,r in reports),new_gfn2_attempts=sum(v['attempted'] for v in summary.values()),
        new_fit_outputs=0,new_esen_queries=0,new_optimizer_steps=0,seconds=time.perf_counter()-tick,
        selection_sha256=sha(a.selection) if a.selection else None))


if __name__=='__main__':main()
