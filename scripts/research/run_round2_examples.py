"""Generate a fixed small validation-composition illustration panel, keeping every draw."""
import argparse,json,hashlib
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from scripts.research.run_matched_connection import load_parent,make_context,generate,score
from scripts.research.run_atomwise_connection import restore_head


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.project.resolve();protocol=json.loads(a.protocol.read_text());ph=hashlib.sha256(a.protocol.read_bytes()).hexdigest();assert protocol['frozen'];a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    spec=json.loads((root/protocol['parent_protocol']).read_text());assert hashlib.sha256((root/protocol['parent_protocol']).read_bytes()).hexdigest()==protocol['parent_protocol_sha256']
    spec['test_rows']=protocol['conditions'];spec['strength_limit']=4.;source=base.HarmonicSource();arm=spec['parents'][0]['fm'];model=load_parent(root,arm);context=make_context(arm,source);head,_=restore_head(root,protocol['head'])
    report=generate(spec,ph,0,'fm',model,source,context,head,protocol['conditions'],protocol['seed'],32,[4.],a.out/'generation',method_prefix='illustration')
    summary=score(spec,ph,0,[(a.out/'generation',report)],a.out/'xtb')
    (a.out/'complete.json').write_text(json.dumps(dict(complete=True,protocol_sha256=ph,summary=summary,new_outputs=128,new_gfn2_attempts=128,new_training_steps=0,new_teacher_queries=0,scope=protocol['scope']),indent=2)+'\n')


if __name__=='__main__':main()
