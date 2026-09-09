#!/usr/bin/env python3
"""Two-environment work-correction pilot; a declared restrained eSEN target.

propose: envs/flowmol, fixed-condition stochastic paths using frozen FM.
score: envs/omol25, energy values and weights on exactly those endpoints.
This is a feasibility/ESS diagnostic, not a molecular advantage experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def propose(args):
    import dgl
    from flowmol.model_utils.load import read_config_file,model_from_config
    from flowmol.data_processing.dataset import MoleculeDataset
    from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask
    from cfm_mol.clamped_work import sample_clamped_work_proposal
    from cfm_mol.radial_reference import prepare_research_backbone
    from cfm_mol.smooth_geometry import patch_smooth_geometry
    cfg=read_config_file(args.config);cfg['mol_fm'].pop('bgfm',None)
    cfg['mol_fm']['prior_config']['x']['align']=False
    dataset=MoleculeDataset('val',dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
        explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False)),prior_config=cfg['mol_fm']['prior_config'])
    base=dataset[args.source_row]
    if not 2<=base.num_nodes()<=200:raise ValueError('Atom count outside supported range')
    charges=base.ndata['c_1_true'].argmax(-1)-2
    if ((charges<=-2)|(charges>=3)).any():raise ValueError('Legacy clipping-boundary charge excluded')
    state=torch.load(str(args.checkpoint),map_location='cpu',weights_only=False)
    protocol=state.get('research_protocol')
    if protocol is None:raise ValueError('Require an explicitly trained conditional checkpoint')
    model=model_from_config(cfg);prepare_research_backbone(model,protocol)
    model.load_state_dict(state['state_dict'],strict=True)
    model.to(args.device).float().eval();patch_smooth_geometry(model,protocol.get('geometry_softening',0.))
    graph=dgl.batch([base]*args.particles).to(args.device)
    nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
    start=time.perf_counter()
    result=sample_clamped_work_proposal(model,graph,nbi,uem,steps=args.steps,
        terminal_time=protocol['data_endpoint_time'],noise_scale=args.noise_scale,
        parameterization=protocol['position_parameterization'],
        generator=torch.Generator(device=args.device).manual_seed(args.seed))
    for key,value in list(result.items()):
        if isinstance(value,torch.Tensor):result[key]=value.cpu()
    result.update(complete=True,source_row=args.source_row,source_split='legacy validation',
        symbols=[cfg['dataset']['atom_map'][int(i)] for i in base.ndata['a_1_true'].argmax(-1)],
        total_charge=int(charges.sum()),spin_provenance='original spin unknown; must declare at scoring',
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),
        config_sha256=sha(args.config),research_protocol=protocol,seed=args.seed,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),
            'cfm_mol/clamped_work.py':sha(Path(__file__).resolve().parents[2]/'cfm_mol/clamped_work.py'),
            'cfm_mol/nonequilibrium.py':sha(Path(__file__).resolve().parents[2]/'cfm_mol/nonequilibrium.py')},
        seconds=time.perf_counter()-start)
    torch.save(result,args.out)
    print(json.dumps({'complete':True,'particles':args.particles,'source_row':args.source_row,
        'seconds':result['seconds'],'output':str(args.out)}),flush=True)


def score(args):
    import ase
    from ase.data import atomic_numbers
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from cfm_mol.nonequilibrium import WeightedPaths,normalized_weights
    proposal=torch.load(str(args.proposals),map_location='cpu',weights_only=False)
    if not proposal['complete']:raise ValueError('Incomplete proposal file')
    if not args.kT>0 or not args.restraint_strength>0:raise ValueError('Positive kT and explicit restraint required')
    numbers=[atomic_numbers[s] for s in proposal['symbols']]
    electrons=sum(numbers)-proposal['total_charge']
    if electrons<1:raise ValueError('Invalid electron count')
    if args.spin_policy=='declared-singlet' and electrons%2:raise ValueError('Singlet incompatible with electron parity')
    spin=1+electrons%2 if args.spin_policy=='minimum-parity' else 1
    calculator=FAIRChemCalculator(load_predict_unit(str(args.oracle),device=args.device))
    report={'complete':False,'scope':__doc__,'proposal_file':str(args.proposals.resolve()),
        'proposal_sha256':sha(args.proposals),'oracle_path':str(args.oracle.resolve()),'oracle_sha256':sha(args.oracle),
        'script_sha256':sha(__file__),'source_row':proposal['source_row'],'symbols':proposal['symbols'],
        'total_charge':proposal['total_charge'],'spin_multiplicity':spin,'spin_policy':args.spin_policy,
        'spin_provenance':'declared convention; original OMol electronic state unknown',
        'target':{'kT_eV':args.kT,'restraint_eV_A2':args.restraint_strength,
            'potential':'eSEN(x) + restraint_strength/2 * sum_i ||x_i||^2 on COM-free H',
            'normalization_assumption':'eSEN is bounded below; harmonic restraint confines the relative coordinates',
            'not_claimed':'unrestrained Boltzmann target, independent test data, physical kinetics or calibrated generation'},
        'rows':[]}
    start=time.perf_counter()
    for i,positions in enumerate(proposal['positions']):
        atoms=ase.Atoms(numbers=numbers,positions=positions.double().numpy())
        atoms.info.update(charge=proposal['total_charge'],spin=spin);atoms.calc=calculator
        try:
            energy=float(atoms.get_potential_energy())
            if not torch.isfinite(torch.tensor(energy,dtype=torch.float64)):raise FloatingPointError('Non-finite energy')
            restraint=args.restraint_strength/2*float(positions.double().square().sum())
            logw=float(proposal['proposal_log_factor'][i])-(energy+restraint)/args.kT
            if not torch.isfinite(torch.tensor(logw,dtype=torch.float64)):raise FloatingPointError('Non-finite work weight')
            row={'sample_id':i,'success':True,'energy_eV':energy,'restraint_eV':restraint,'log_weight':logw}
        except Exception as exc:
            row={'sample_id':i,'success':False,'error':f'{type(exc).__name__}: {str(exc)[:500]}'}
        report['rows'].append(row)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['all_energies_finite']=all(row['success'] for row in report['rows'])
    if report['all_energies_finite']:
        logw=torch.tensor([row['log_weight'] for row in report['rows']],dtype=torch.float64)
        report['weights']=WeightedPaths(proposal['positions'],logw,{}).summary()
        report['normalized_weights']=normalized_weights(logw).tolist()
    else:
        report['estimator_status']='unavailable: failed oracle calls retained; no selection of successful endpoints'
    report.update(complete=True,seconds=time.perf_counter()-start)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','normalized_weights']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('propose')
    a.add_argument('--config',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True)
    a.add_argument('--source-row',type=int,required=True);a.add_argument('--particles',type=int,default=32)
    a.add_argument('--steps',type=int,default=32);a.add_argument('--noise-scale',type=float,default=.1)
    a.add_argument('--seed',type=int,default=9021)
    b=sub.add_parser('score')
    b.add_argument('--proposals',type=Path,required=True);b.add_argument('--oracle',type=Path,required=True)
    b.add_argument('--kT',type=float,default=1.);b.add_argument('--restraint-strength',type=float,default=.1)
    b.add_argument('--spin-policy',choices=['minimum-parity','declared-singlet'],required=True)
    for parser in [a,b]:parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    (propose if args.stage=='propose' else score)(args)


if __name__=='__main__':main()
