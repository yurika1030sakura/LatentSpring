#!/usr/bin/env python3
"""Paired fresh confirmation of fixed nonlinear and matched linear refinements."""
import argparse
import json
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.linear_entropy_adapter import endpoint_kl_change
from cfm_mol.nonequilibrium import WeightedPaths
from molecular_tempered_pilot import sha, write_json


def summarize(x):return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'samples': len(x)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--base-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--nonlinear-run',default='species_entropy_adapter_1000_v1')
    p.add_argument('--linear-run',default='linear_entropy_adapter_typed_1000_v1');args=p.parse_args()
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    old=json.loads((args.base_run/'results.json').read_text())
    if not old['complete'] or old['configuration']['seed']!=9169 or old['configuration']['eval_particles']!=2048:
        raise ValueError('Require prescribed fresh2048-path stream')
    data=torch.load(str(args.base_run/'final_samples.pt'),map_location='cpu',weights_only=False)
    x=data['positions'].double();condition=data['condition'];recipe=old['configuration']
    report={'complete':False,'scope':__doc__,'condition':condition,'source_checkpoint_sha256':old['trained_checkpoint_sha256'],
        'base_sha256':sha(args.base_run/'final_samples.pt'),'seed':9169,'samples':len(x),'arms':{},
        'limitations':['One condition and one trained nonlinear seed; this is evaluation confirmation, not training replication.',
            'Unknown base entropy and target normalizer cancel only in relative endpoint-KL changes.',
            'Fresh weights use the original source reverse model; compare within this fresh panel.',
            'No general target calibration, chemical-validity certificate or ICLR novelty follows.']}
    write_json(output,report);outputs={}
    root=Path(__file__).resolve().parents[2];oracle_path=Path(recipe['oracle'])
    if sha(oracle_path)!=old['oracle_sha256']:raise ValueError('Physical oracle changed')
    with EnergyOracle(Path(recipe['oracle_python']),root/'scripts/research/oracle_worker.py',oracle_path,
        numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle,torch.no_grad():
        for label,name in [('linear',args.linear_run),('nonlinear',args.nonlinear_run)]:
            model,trained,digest=load_entropy_adapter(args.runs_root/name,args.device)
            if trained['source_checkpoint_sha256']!=old['trained_checkpoint_sha256'] or trained['condition']!=condition or trained['oracle_sha256']!=old['oracle_sha256']:
                raise ValueError('Adapter and source differ')
            if trained['kT_eV']!=recipe['kT'] or trained['restraint_eV_A2']!=recipe['restraint'] or trained['oracle_evaluations']!=16512:
                raise ValueError('Target or matched training budget differs')
            positions=[];volumes=[]
            for begin in range(0,len(x),64):
                y,volume=model(x[begin:begin+64].to(args.device))
                if volume.ndim==0:volume=volume.expand(len(y))
                positions.append(y.cpu());volumes.append(volume.cpu())
            y=torch.cat(positions);volume=torch.cat(volumes)
            inverse_check=None
            if label=='nonlinear':
                inverse,inv_volume,info=model.inverse(y[:32].to(args.device))
                error=float((inverse.cpu()-x[:32]).abs().max())
                if error>1e-8 or float((inv_volume.cpu()+volume[:32]).abs().max())>1e-8:raise RuntimeError('Fresh inverse/volume check failed')
                inverse_check={'position_error_A':error,**info}
            energy,_=oracle.evaluate(y)
            change=endpoint_kl_change(data['energy_eV'],energy,x,y,kT=recipe['kT'],restraint=recipe['restraint'],log_volume=volume)
            work=data['work']+change
            outputs[label]={'positions':y,'energy_eV':energy,'work':work,'log_volume':volume,
                'paired_endpoint_kl_change':change,'condition':condition}
            torch.save(outputs[label],args.out/f'{label}_samples.pt')
            report['arms'][label]={'paired_endpoint_kl_change':summarize(change),'weights':WeightedPaths(y,-work,{}).summary(),
                'inverse_check':inverse_check,'checkpoint_sha256':digest,'samples_sha256':sha(args.out/f'{label}_samples.pt')}
            write_json(output,report);del model
        report['new_adapter_oracle_queries']=oracle.evaluated
    difference=outputs['nonlinear']['paired_endpoint_kl_change']-outputs['linear']['paired_endpoint_kl_change']
    report.update(complete=True,nonlinear_minus_linear=summarize(difference),base_weights=WeightedPaths(x,-data['work'],{}).summary(),
        total_confirmation_oracle_queries=old['oracle_evaluations']+report['new_adapter_oracle_queries'],seconds=time.perf_counter()-start)
    write_json(output,report);print(json.dumps({'arms':report['arms'],'nonlinear_minus_linear':report['nonlinear_minus_linear']}),flush=True)


if __name__=='__main__':main()
