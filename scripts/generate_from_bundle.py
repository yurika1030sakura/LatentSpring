"""Generate every requested sample from an exported LatentSpring EGNN bundle."""
import argparse,json
from pathlib import Path
import torch
from cfm_mol.latentspring_generator import LatentSpringGenerator

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['bundle','condition','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--upstream',type=Path);p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--family',default='fm',choices=['fm','gaga']);p.add_argument('--fit',type=int,default=0);p.add_argument('--samples',type=int,default=16);p.add_argument('--seed',type=int,default=0);p.add_argument('--stream',type=int,default=0);p.add_argument('--without-hydrogen-readout',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    c=json.loads(a.condition.read_text());generator=LatentSpringGenerator(a.bundle,device=a.device,upstream=a.upstream)
    result=generator.generate(c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],samples=a.samples,seed=a.seed,stream=a.stream,fit=a.fit,family=a.family,hydrogen=not a.without_hydrogen_readout)
    a.out.mkdir(parents=True);torch.save(result,a.out/'samples.pt');(a.out/'report.json').write_text(json.dumps(dict(costs=result['costs'],settings=result['settings']),indent=2)+'\n');print(json.dumps(result['costs']))

if __name__=='__main__':main()
