#!/usr/bin/env python3
"""Fresh, common-source molecular generation after the frozen FM pairing pilot."""
import argparse
import json
import os
from pathlib import Path
import time
import dgl
import torch
from flowmol.model_utils.load import model_from_config, read_config_file
from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.condition_systems import graph_from_condition, load_condition
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_chemical_policy import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','out','protocol']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    protocol=json.loads(args.protocol.read_text());assert protocol['frozen']
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2]
    config=root/protocol['config'];manifest=root/protocol['condition_manifest']
    assert sha(config)==protocol['config_sha256'] and sha(manifest)==protocol['condition_manifest_sha256']
    checkpoints={'warm':args.project/protocol['warm_checkpoint']}
    checkpoints.update({m:args.run/m/'last.ckpt' for m in protocol['methods']})
    cfg=read_config_file(config);cfg['mol_fm'].pop('bgfm',None)
    assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    args.out.mkdir(parents=True);rows=[];source_records={};orders=[]
    for method, checkpoint in checkpoints.items():
        digest=sha(checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);recipe=state['research_protocol']
        if method=='warm':assert digest==protocol['warm_checkpoint_sha256']
        else:
            report=json.loads((checkpoint.parent/'validation.json').read_text())
            assert report['complete'] and report['all_weights_finite'] and report['global_step']==protocol['training_steps']
            assert recipe['pairing_protocol_sha256']==sha(args.protocol) and recipe['pairing']==method
            assert recipe['warm_sha256']==protocol['warm_checkpoint_sha256'] and recipe['seed']==protocol['training_seed']
            orders.append(recipe['data_order_sha256'])
        assert recipe['prior_std']==1. and recipe['requested_kT']==1. and recipe['data_endpoint_time']==1.
        assert recipe['electronic_conditioning'] and recipe['position_parameterization']=='displacement'
        source_records[method]=dict(checkpoint=str(checkpoint),sha256=digest,
            protocol=recipe,training=None if method=='warm' else report)
        model=model_from_config(cfg);prepare_research_backbone(model,recipe)
        model.load_state_dict(state['state_dict'],strict=True)
        patch_smooth_geometry(model,recipe.get('geometry_softening',0.))
        model=model.cuda().float().eval();model.requires_grad_(False);del state
        for condition_index in protocol['conditions']:
            condition=load_condition(manifest,condition_index)
            condition['requested_kT_eV']=protocol['model_input_kT_eV'];condition['numbers']=condition['atomic_numbers']
            base=graph_from_condition(condition,cfg['dataset']['atom_map'],
                n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4)
            n=base.num_nodes();batch=protocol['evaluation_batch'];count=protocol['samples_per_condition']
            graph=dgl.batch([base]*batch).to('cuda');nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
            basis=centered_orthonormal_basis(n,device='cuda')
            samples=[];seeds=[];start=time.perf_counter();torch.cuda.synchronize()
            for offset in range(0,count,batch):
                seed=protocol['evaluation_seed']*1000003+condition_index*100003+offset//batch
                generator=torch.Generator(device='cuda').manual_seed(seed);seeds.append(seed)
                z=torch.randn((batch,n-1,3),generator=generator,device='cuda',dtype=torch.float64)
                x0=torch.einsum('nk,bkd->bnd',basis,z).reshape(batch*n,3).float()
                with torch.no_grad():
                    x=sample_clamped_flow(model,graph,nbi,uem,x0=x0,n_ode_steps=protocol['midpoint_steps'],
                        terminal_time=protocol['terminal_time'],parameterization='displacement').reshape(batch,n,3).double()
                x=x-x.mean(1,keepdim=True)
                noise=torch.randn((batch,n-1,3),generator=generator,device='cuda',dtype=torch.float64)
                x=x+protocol['terminal_noise_std_A']*torch.einsum('nk,bkd->bnd',basis,noise)
                if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite molecular outputs; retain failed job')
                samples.append(x.cpu())
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            positions=torch.cat(samples);assert len(positions)==count
            torch.save(dict(positions=positions,condition=condition,seeds=seeds),args.out/f'{method}_c{condition_index}.pt')
            support=assess(positions,condition,list(range(count)))
            row=dict(method=method,condition_index=condition_index,condition=condition,
                sample_sha256=sha(args.out/f'{method}_c{condition_index}.pt'),seeds=seeds,
                generation_seconds=seconds,neural_field_evaluations=count*2*protocol['midpoint_steps'],
                radius_of_gyration_mean_A=float(positions.square().sum(-1).mean(-1).sqrt().mean()),**support)
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['method','condition_index','attempted','graph_supported','geometrically_supported','distinct_connectivity','validator_errors','generation_seconds']}),flush=True)
        del model;torch.cuda.empty_cache()
    assert len(set(orders))==1
    totals={m:sum(r['graph_supported'] for r in rows if r['method']==m) for m in checkpoints}
    gate=totals['steric']>totals['independent'] and totals['steric']>totals['rotation']
    write(args.out/'results.json',dict(complete=True,protocol_sha256=sha(args.protocol),sources=source_records,rows=rows,
        graph_supported_totals=totals,attempts_per_method=len(protocol['conditions'])*protocol['samples_per_condition'],
        proceed_to_frozen_energy_check=gate,new_molecular_oracle_calls=0,scientific_submission_ready=False,
        scope='All fresh unconditional geometries at fixed compositions, with common Gaussian prior/noise draws. FM-only training; structural outcomes do not certify Boltzmann probabilities. Geometric support, RDKit graph support, and validator errors remain separate. Single training seed.'))


if __name__=='__main__':main()
