"""Fixed-budget native FlowMol pilot; three closed-set compositions, not GAGA evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from cfm_mol.weighted_endpoints import file_sha256
from scripts.research.audit_generator_output_support import assess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.project;out=args.out
    protocol_path=Path(__file__).resolve().parents[2]/'research/evidence/weighted_minima_native_v1.json'
    spec=json.loads(protocol_path.read_text());assert spec['frozen']
    for key,digest in spec['inputs'].items():assert file_sha256(root/key)==digest,key
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    checkpoint=root/spec['checkpoint'];config=root/spec['config'];rows=[]
    # The unadapted baseline shares the NEW no-terminal-noise sampling protocol.
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)
    parent['research_protocol']={**parent['research_protocol'],'mass_preserving_minima':
        dict(terminal_noise_std_A=0.,model_kT_eV=1.,teacher_temperature_K=300.,
             role='Unadapted parent, metadata only; no claim of having learned minima.')}
    base=out/'base.ckpt';torch.save(parent,base);del parent
    for arm in ['base','proposal_min','full_work_min']:
        model=base
        if arm!='base':
            subprocess.run([sys.executable,'-u','-m','scripts.research.train_weighted_minima',
                '--checkpoint',str(checkpoint),'--config',str(config),
                '--endpoints',str(root/spec['stores'][arm]),'--out',str(out/arm),
                '--device','cuda','--steps',str(spec['steps']),'--batch-size','1',
                '--seed',str(spec['seed']),'--allow-legacy-pickle'],check=True)
            model=out/arm/'last.ckpt'
        destination=out/(arm+'_raw')
        subprocess.run([sys.executable,'-u','-m','scripts.research.generate_weighted_minima',
            '--checkpoint',str(model),'--config',str(config),'--conditions',str(root/spec['conditions']),
            '--out',str(destination),'--device','cuda','--samples',str(spec['samples']),
            '--seed',str(spec['evaluation_seed']),'--allow-legacy-pickle'],check=True)
        report=json.loads((destination/'generation.json').read_text());assert report['complete']
        for row in report['rows']:
            path=destination/row['file'];assert file_sha256(path)==row['sha256']
            with np.load(path,allow_pickle=False) as values:x=torch.from_numpy(values['raw_positions'])
            result=assess(x,row['condition'],list(range(len(x))))
            rows.append(dict(arm=arm,condition=row['condition'],raw_sha256=row['sha256'],**result))
        (out/'progress.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2)+'\n')
    result=dict(complete=True,protocol_sha256=file_sha256(protocol_path),rows=rows,
        new_training_steps=2*spec['steps'],new_raw_outputs=3*3*spec['samples'],
        new_physical_queries=0,native_flowmol=True,held_out_generalization=False,
        scope=spec['scope'],basin_probabilities_evaluated=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({arm:sum(r['graph_supported'] for r in rows if r['arm']==arm) for arm in ['base','proposal_min','full_work_min']}))


if __name__=='__main__':main()
