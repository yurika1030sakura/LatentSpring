#!/usr/bin/env python3
"""Inspect internal coordinate collisions along a fixed reverse midpoint solve."""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType
import time
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from cfm_mol.perturbation_loader import PerturbationLoader
from cfm_mol.clamped_density import center_by_graph,deterministic_field,position_velocity
from cfm_mol.smooth_geometry import patch_smooth_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--parent',type=int,default=1137)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();started=time.monotonic()
    cfg=read_config_file('configs/sweep/a1_fm_only_s2.yaml');cfg['mol_fm'].pop('bgfm',None)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    protocol=state['research_protocol'];T=protocol['data_endpoint_time']
    model=model_from_config(cfg);model.load_state_dict(state['state_dict'],strict=True)
    model.to(args.device).float().eval();patch_smooth_geometry(model,protocol.get('geometry_softening',0.))
    shard=Path('/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed/perturbation_val_n10000_s0.pt')
    loader=PerturbationLoader(shard,model.n_atom_types,b_parents=1,device=args.device,perturbation_indices=[0,1])
    loader._order=[args.parent];loader._ptr=0
    graph,energy,pid,nbi,uem=loader.next_batch();source,destination=graph.edges()
    original=model.vector_field.precompute_distances;calls=[];active=[]
    def inspected(self,g,node_positions=None):
        x=g.ndata['x_t'] if node_positions is None else node_positions
        distances=(x[source]-x[destination]).norm(dim=-1)
        active.append([float(distances[nbi[source]==i].min()) for i in range(g.batch_size)])
        return original(g,node_positions)
    model.vector_field.precompute_distances=MethodType(inspected,model.vector_field)
    try:
        with torch.no_grad(),graph.local_scope(),deterministic_field(model.vector_field):
            for key in ['a','c']:graph.ndata[f'{key}_t']=graph.ndata[f'{key}_1_true']
            graph.edata['e_t']=graph.edata['e_1_true']
            x=center_by_graph(graph.ndata['x_1_true'],nbi,graph.batch_size);h=T/args.steps
            for step in range(args.steps):
                t=x.new_full((graph.batch_size,),T-(step+.5)*h)
                velocities=[]
                for stage in range(2):
                    active.clear();current=x if stage==0 else x-.5*h*velocities[0]
                    velocities.append(position_velocity(model,graph,current,t,nbi,uem,
                        parameterization=protocol.get('position_parameterization','endpoint')))
                    calls.append({'step':step,'stage':stage,'time':float(t[0]),
                        'minimum_distance_by_internal_stage_A':[list(v) for v in active]})
                x=x-h*velocities[-1]
    finally:model.vector_field.precompute_distances=original
    # In this n_recycles=1 backbone the final distance computation feeds only
    # unused discrete logits, so report it separately from used geometry.
    if model.vector_field.n_recycles!=1:raise ValueError('Interpretation assumes one recycle')
    used=[v for call in calls for pair in call['minimum_distance_by_internal_stage_A'][:-1] for v in pair]
    inputs=[v for call in calls for v in call['minimum_distance_by_internal_stage_A'][0]]
    report={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'parent':args.parent,'terminal_time':T,'steps':args.steps,'device':args.device,'dtype':'float32',
        'geometry_softening':protocol.get('geometry_softening',0.),
        'minimum_input_pair_distance_A':min(inputs),'minimum_used_internal_pair_distance_A':min(used),
        'calls_with_used_internal_distance_below_1e_3_A':sum(min(v for pair in c['minimum_distance_by_internal_stage_A'][:-1] for v in pair)<1e-3 for c in calls),
        'calls':calls,'seconds':time.monotonic()-started,'complete':True,
        'scope':'coordinate-path diagnostic; CPU/GPU roundoff may give different trajectories; no trace or performance inference'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='calls'}))


if __name__=='__main__':main()
