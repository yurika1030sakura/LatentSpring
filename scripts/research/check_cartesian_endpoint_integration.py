#!/usr/bin/env python3
"""CPU integration diagnostic using native radial reference, not main FlowMol.

Fresh MMFF94s teacher queries; fixed 2-angle basin maps/minima/reference masses
are reused from the prior pilot and fully identified by hashes. The model reads
only atom/electronic composition and learns CARTESIAN coordinates through the
repository's clamped_fm_loss, harmonic source and symmetry pairing. No energy
or geometry refinement is used in generation. This is a capacity-limited,
closed-set integration check, not a benchmark or a new ICLR-quality result.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import time
import math
import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolTransforms
from cfm_mol.weighted_endpoints import EndpointGroup, WeightedEndpointStore, file_sha256
from cfm_mol.weighted_endpoint_fm import checkpoint_source, endpoint_fm_loss
from cfm_mol.endpoint_diagnostics import TensorGraph, radial_model
from cfm_mol.tree_prior_controls import TreePriorControl
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.chemical_moves import infer_chemical_graph, covalent_radii
from cfm_mol.geometric_domain import connected_nonoverlapping

PANEL=[('pentane','CCCCC'),('1_butanol','CCCCO'),('1_butanamine','CCCCN'),('diethyl_ether','CCOCC')]
TORSIONS=[(0,1,2,3),(1,2,3,4)]
R=0.00198720425864083

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi


def build_stores(inputs,out,seed,n):
    arms={k:[] for k in ['proposal_min','full_work_min']};stats=[]
    for name,smiles in PANEL:
        path=inputs/(name+'_base.sdf')
        mol=Chem.SDMolSupplier(str(path),removeHs=False)[0]
        if mol is None:raise RuntimeError('Cannot read cached molecule')
        conf=mol.GetConformer();base=conf.GetPositions().copy()
        refpath=inputs/('landscape_'+name+'.npz')
        with np.load(refpath,allow_pickle=False) as f:
            labels,minangles,reference=f['labels'].copy(),f['minima'].copy(),f['pref'].copy()
        props=AllChem.MMFFGetMoleculeProperties(mol,mmffVariant='MMFF94s')
        ff=AllChem.MMFFGetMoleculeForceField(mol,props)
        def coordinates(phi):
            for i,pos in enumerate(base):conf.SetAtomPosition(i,pos)
            for tor,v in zip(TORSIONS,phi):rdMolTransforms.SetDihedralRad(conf,*tor,float(v))
            x=conf.GetPositions().copy();return x-x.mean(0)
        minima=np.stack([coordinates(phi) for phi in minangles])
        rng=np.random.default_rng(seed);z=rng.uniform(-np.pi,np.pi,(n,2))
        y=wrap(z+.75*np.sin(z));logj=np.log1p(.75*np.cos(z)).sum(1)
        energy=np.array([ff.CalcEnergy(tuple(coordinates(phi).ravel())) for phi in y])
        if not np.isfinite(energy).all():raise FloatingPointError('Nonfinite teacher energies')
        logw=-(energy-energy.min())/(R*300.)+logj  # uniform torus source; constants cancel
        ind=np.floor((y+np.pi)*len(labels)/(2*np.pi)+.5).astype(int)%len(labels)
        basin=labels[ind[:,0],ind[:,1]]
        numbers=tuple(a.GetAtomicNum() for a in mol.GetAtoms())
        np.savez_compressed(out/f'teacher_{name}_{seed}.npz',source_angles=z,proposal_angles=y,
            energy_kcal_mol=energy,log_jacobian=logj,log_weight=logw,basin=basin)
        provenance=dict(source_sdf_sha256=file_sha256(path),reference_npz_sha256=file_sha256(refpath),
            teacher_seed=seed,fresh_MMFF_queries=n,teacher_particle_file=f'teacher_{name}_{seed}.npz',
            reused_reference=True,reference_scope='old fixed two-angle map; not full Cartesian relaxation')
        masses={}
        for arm in arms:
            weights=logw if arm=='full_work_min' else np.zeros(n)
            g=EndpointGroup(name,smiles,numbers,0,1,300.,.25,minima[basin],weights,
                np.ones(n,dtype=bool),tuple(map(str,basin)),tuple(f'{seed}:{j}' for j in range(n)),
                dict(oracle_id='MMFF94s',base_measure='dphi1 dphi2 on fixed internal-geometry torus',
                    support='two-angle panel, all other internal geometry fixed',
                    optimizer='cached deterministic 128x128 angle-basin map + restricted minima',
                    weight_scope='graph_global',split='train',
                    weighting=arm,thermal_correctness_claim=(arm=='full_work_min')),
                provenance)
            aggregate=g.aggregate_identical_minima()
            p=np.array([aggregate.masses().get(str(j),0.) for j in range(len(reference))])
            masserr=max(abs(g.masses()[b]-aggregate.masses()[b]) for b in g.masses())
            masses[arm]=dict(TV_reference=float(.5*abs(p-reference).sum()),mass_preservation_error=masserr,
                            probabilities=p.tolist())
            arms[arm].append(aggregate)
        stats.append(dict(name=name,teacher_particles=n,fresh_MMFF_queries=n,arms=masses))
    stores={k:WeightedEndpointStore(v) for k,v in arms.items()}
    for k,v in stores.items():v.save(out/f'endpoints_{k}_{seed}')
    return stores,stats


@torch.no_grad()
def evaluate(model,store,prior,out,seed,attempts,steps):
    results=[];rng=np.random.default_rng(seed+900000)
    for group in store.groups:
        single=WeightedEndpointStore([EndpointGroup(**{**group.__dict__,'mixture_mass':1.})])
        draws=single.draw(attempts,rng)
        graph=TensorGraph(draws);nbi=graph.nbi;uem=graph.edges()[0]<graph.edges()[1]
        x0=checkpoint_source(prior,draws,seed+700000).float()
        # Erase labels entirely: they specify only the tensor shape at sampling.
        graph.ndata['x_1_true']=torch.zeros_like(graph.ndata['x_1_true'])
        graph.ndata['has_reference_geometry'].zero_()
        begin=time.perf_counter()
        y=sample_clamped_flow(model,graph,nbi,uem,x0=x0,n_ode_steps=steps,
            terminal_time=1.,parameterization='displacement').reshape(attempts,len(group.numbers),3)
        seconds=time.perf_counter()-begin
        geometry=connected_nonoverlapping(y,covalent_radii(group.numbers))
        valid=[];errors=0;smiles=[]
        for j,x in enumerate(y):
            if not geometry[j]:valid.append(False);smiles.append(None);continue
            try:
                with rdBase.BlockLogs():g=infer_chemical_graph(x,group.numbers,0)
                valid.append(True);smiles.append(g['connectivity_smiles'])
            except ValueError:valid.append(False);smiles.append(None)
            except (RuntimeError,IndexError):errors+=1;valid.append(False);smiles.append(None)
        np.savez_compressed(out/f'raw_{group.group_id}.npz',raw_positions=y.numpy(),source=x0.numpy(),
            geometrically_supported=geometry.numpy(),graph_supported=np.array(valid))
        counts=Counter(s for s in smiles if s is not None)
        # These counts are not sufficient for basin frequencies without graph-
        # and symmetry-aware atom remapping, which this diagnostic does not do.
        results.append(dict(group=group.group_id,composition=group.composition_id,attempted=attempts,
            finite=int(torch.isfinite(y).all((1,2)).sum()),geometrically_supported=int(geometry.sum()),
            graph_valid=sum(valid),validator_errors=errors,connectivity_counts=dict(counts),seconds=seconds,
            intended_graph_count=counts.get(group.graph_id,0),student_basin_TV=None,
            basin_evaluation_status='not evaluated: full graph/symmetry correspondence not qualified'))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--particles',type=int,default=4096)
    p.add_argument('--seed',type=int,nargs='+',default=[260938,260939,260940])
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--attempts',type=int,default=64)
    p.add_argument('--sample-steps',type=int,default=32)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    protocol=dict(seeds=a.seed,steps=a.steps,batch_size=a.batch_size,teacher_particles=a.particles,
        attempts_per_group=a.attempts,sampling_steps=a.sample_steps,network_evaluations_per_output=2*a.sample_steps,
        model='unmodified repository RadialPairReference with displacement-head wrapper; NOT FlowMol',
        source='saved-law TreePriorControl harmonic_tree width .2',endpoint_sigma_A=.005,
        teacher_temperature_K=300.,input='composition only, no graph or molecule ID',
        geometry_target='two-angle restricted MMFF minima; NOT full Cartesian or DFT minima',
        original_six_million_model=False,checkpoint_initialized=False,self_conditioning=False,
        purpose='integration/optimization diagnostic; not confirmatory paper results',
        raw_generation='no external optimization, rejection, energy filter, or terminal noise',
        reference_sha256={f.name:file_sha256(f) for f in a.inputs.iterdir() if f.is_file()})
    dump(a.out/'protocol.json',protocol);all_stats=[];start=time.perf_counter()
    prior=TreePriorControl('harmonic_tree').double().requires_grad_(False)
    for seed in a.seed:
        stores,teacher_stats=build_stores(a.inputs,a.out,seed,a.particles)
        dump(a.out/f'teacher_summary_{seed}.json',teacher_stats)
        for arm,store in stores.items():
            run=a.out/f'{arm}_{seed}';run.mkdir()
            torch.manual_seed(seed);model=radial_model();opt=torch.optim.AdamW(model.vector_field.parameters(),lr=.001,weight_decay=1e-12)
            rng=np.random.default_rng(seed);losses=[];begin=time.perf_counter()
            for step in range(1,a.steps+1):
                draws=store.draw(a.batch_size,rng,sigma_A=.005);g=TensorGraph(draws)
                loss=endpoint_fm_loss(model,g,g.nbi,g.edges()[0]<g.edges()[1],draws,prior,seed*3000017+step*a.batch_size)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.vector_field.parameters(),1.,error_if_nonfinite=True);opt.step()
                losses.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
                if step%100==0:print(json.dumps(dict(arm=arm,seed=seed,**losses[-1])),flush=True)
            seconds=time.perf_counter()-begin
            torch.save(dict(state_dict=model.vector_field.state_dict(),steps=a.steps,seed=seed),run/'radial_diagnostic.pt')
            dump(run/'training.json',losses)
            result=evaluate(model,store,prior,run,seed,a.attempts,a.sample_steps)
            row=dict(arm=arm,seed=seed,trainable_parameters=sum(v.numel() for v in model.vector_field.parameters()),
                first_50_mean_loss=float(np.mean([x['loss'] for x in losses[:50]])),
                last_50_mean_loss=float(np.mean([x['loss'] for x in losses[-50:]])),
                train_seconds=seconds,evaluation=result)
            dump(run/'results.json',row);all_stats.append(row)
            dump(a.out/'progress.json',dict(complete=False,completed=len(all_stats),results=all_stats))
    dump(a.out/'summary.json',dict(complete=True,runs=all_stats,total_seconds=time.perf_counter()-start,
        fresh_teacher_MMFF_queries=len(a.seed)*len(PANEL)*a.particles,main_model_trained=False,
        improved_main_paper_performance_established=False))


if __name__=='__main__':main()
