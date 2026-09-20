"""Generate untouched parents once, then compare all frozen hydrogen readouts."""
import argparse,gc,json,subprocess,sys
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from scripts.research.run_matched_connection import load_parent,make_context,generate,score
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','training','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);a=p.parse_args();torch.set_num_threads(2);root=a.project.resolve();a.out=a.out.resolve();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);si=a.seed_index
    assert spec['frozen'] and sha(root/spec['pilot_audit'])==spec['pilot_audit_sha256'] and sha(root/spec['energy_nomination'])==spec['energy_nomination_sha256']
    a.out.mkdir(parents=True,exist_ok=False);reports=[];source=base.HarmonicSource();sources={};parent_spec=dict(spec,strength_limit=4.)
    for name,arm in spec['parents'][si].items():
        model=load_parent(root,arm);context=make_context(arm,source);info=spec['physical_heads'][str(si)][name];head,_=restore_head(root,info)
        directory=a.out/'parents'/name;report=generate(parent_spec,ph,si,name,model,source,context,head,spec['test_rows'],spec['evaluation_seeds'][si],spec['test_samples'],[4.],directory)
        reports.append((directory,report));sources[name]=dict(generation=str((directory/'generation.json').relative_to(root)),generation_sha256=sha(directory/'generation.json'),method=name+'_a0',condition_count=len(spec['test_rows']),samples_per_condition=spec['test_samples'])
        del model,context,head;gc.collect();torch.cuda.empty_cache()
    xp=a.out/'parent_xtb';score(spec,ph,si,reports,xp)
    for v in sources.values():v.update(physical=str((xp/'results.json').relative_to(root)),physical_sha256=sha(xp/'results.json'))
    manifest=a.out/'parents.json';write(manifest,dict(complete=True,protocol_sha256=ph,seed_index=si,sources=sources))
    subprocess.run([sys.executable,'-u','-m','scripts.research.evaluate_hydrogen_completion','--project',str(root),'--protocol',str(a.protocol.resolve()),
        '--training',str(a.training.resolve()),'--out',str(a.out/'readouts'),'--seed-index',str(si),'--parents',str(manifest)],check=True)
    readout=json.loads((a.out/'readouts/complete.json').read_text());assert readout['complete'] and readout['protocol_sha256']==ph
    parent_outputs=2*len(spec['test_rows'])*spec['test_samples'];write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,seed_index=si,
        parents_manifest_sha256=sha(manifest),readout_complete_sha256=sha(a.out/'readouts/complete.json'),new_parent_trajectories=parent_outputs,
        new_derived_outputs=readout['new_derived_outputs'],new_gfn2_attempts=parent_outputs+readout['new_gfn2_attempts'],new_esen_queries=0,new_optimizer_steps=0))


if __name__=='__main__':main()
