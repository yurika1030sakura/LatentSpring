#!/usr/bin/env python3
"""Audit frozen-coordinate physical readouts and all-attempt quality yields."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ase.data import chemical_symbols

from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.evaluate_generator_quality import load,write
from scripts.research.train_electronic_fm import sha


def interval(values, rng):
    # The same resampled compositions are shared by the two fitted seeds.
    means=values.mean((0,2))
    idx=np.concatenate([rng.integers(0,16,size=(10000,16))+k for k in [0,16,32,48]],axis=1)
    draws=means[idx].mean(-1)
    return dict(mean=float(values.mean()),composition95=np.quantile(draws,[.025,.975]).tolist(),
        by_seed=values.mean((1,2)).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);assert not args.out.exists()
    methods=['gaussian_fm','harmonic_fm','edm','gaga'];shape=(2,4,64,32)
    graph=np.zeros(shape,bool);energy={k:np.full(shape,np.nan) for k in ['esen','xtb']}
    force={k:np.full(shape,np.nan) for k in ['esen','xtb']};success={k:np.zeros(shape,bool) for k in ['esen','xtb']}
    refs={k:np.full((2,64),np.nan) for k in ['esen','xtb']}
    artifacts=[];esen_queries=0;xtb_attempts=0;inversions=[]
    for seed in [0,1]:
        protocol=args.project/f'research/evidence/generator_quality_s{seed}_v1.json';spec=json.loads(protocol.read_text());ph=sha(protocol)
        for chunk in range(spec['chunks']):
            args.chunk=chunk;conditions,references,samples=load(args,spec)
            for (m,i),(x,row) in samples.items():graph[seed,methods.index(m),i]=row['graph_valid']
            for kind in ['esen','xtb']:
                folder=args.run/f's{seed}/{kind}/chunk{chunk}';file=folder/'results.json';report=json.loads(file.read_text())
                assert report['complete'] and report['protocol_sha256']==ph and report['chunk']==chunk and report['kind']==kind
                artifacts.append(dict(seed=seed,chunk=chunk,kind=kind,results_sha256=sha(file)))
                if kind=='esen':
                    assert len(report['rows'])==16*5
                    assert report['raw_queries']==16*258;esen_queries+=report['raw_queries']
                    for row in report['rows']:
                        i,m=row['condition_index'],row['method'];c=conditions[i]
                        path=folder/row['artifact'];assert sha(path)==row['artifact_sha256']
                        saved=torch.load(path,map_location='cpu',weights_only=False)
                        x=references[i][None] if m=='reference' else samples[m,i][0]
                        assert torch.equal(saved['positions'],x) and saved['condition']==c
                        if m!='reference':assert saved['source_sample_sha256']==samples[m,i][1]['sample_sha256']
                        n=len(x);e=(saved['raw_energy_eV'][:n]+saved['raw_energy_eV'][n:])/2
                        f=(saved['raw_force_eV_A'][:n]-saved['raw_force_eV_A'][n:])/2
                        assert torch.equal(e,saved['even_energy_eV']) and torch.equal(f,saved['even_force_eV_A'])
                        assert torch.isfinite(e).all() and torch.isfinite(f).all()
                        if m=='reference':refs[kind][seed,i]=float(e[0])/c['n_atoms']
                        else:
                            mi=methods.index(m);energy[kind][seed,mi,i]=e.numpy()/c['n_atoms']
                            force[kind][seed,mi,i]=f.square().sum(-1).mean(-1).sqrt().numpy();success[kind][seed,mi,i]=True
                else:
                    tasksfile=folder/'tasks.json';assert sha(tasksfile)==report['tasks_sha256']
                    tasks=json.loads(tasksfile.read_text());lookup={r['task']['task_id']:r for r in tasks['tasks']}
                    assert len(lookup)==len(report['rows'])==report['requested_attempts']==report['attempted']
                    by_id={}
                    for row in report['rows']:
                        item=lookup[row['task_id']];task=item['task'];i,m,j=task['condition_index'],task['method'],task['sample_index'];c=conditions[i]
                        assert item['condition']==c and row['condition_index']==i and row['sample_index']==j
                        expected=references[i] if m=='reference' else samples[m,i][0][j]
                        if task['inversion_check']:expected=-expected
                        np.testing.assert_array_equal(np.asarray(task['positions']),expected.numpy())
                        directory=folder/'details'/row['task_id']
                        for name,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(directory/name)==row[key]
                        lines=(directory/'input.xyz').read_text().splitlines()
                        assert int(lines[0])==c['n_atoms'] and len(lines)==c['n_atoms']+2
                        assert [s.split()[0] for s in lines[2:]]==[chemical_symbols[z] for z in c['numbers']]
                        xyz=np.asarray([[float(v) for v in s.split()[1:]] for s in lines[2:]])
                        np.testing.assert_allclose(xyz,expected.numpy(),atol=1e-12,rtol=0)
                        if row['returncode'] is None:assert not row['success'] and row['failure']=='single_point_timeout'
                        else:
                            gradient=(directory/'gradient').read_text() if (directory/'gradient').exists() else ''
                            if row['gradient_sha256'] is not None:assert sha(directory/'gradient')==row['gradient_sha256']
                            parsed=parse_singlepoint((directory/'stdout.txt').read_text(),(directory/'stderr.txt').read_text(),row['returncode'],gradient,c['n_atoms'])
                            assert parsed['success']==row['success']
                            if row['success']:
                                assert parsed['energy_eV']==row['energy_eV'];np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A'])
                            else:assert parsed['failure']==row['failure']
                        assert '--opt' not in row['command'] and row['command'][-1]=='--grad'
                        assert row['original_charge']==c['charge']==0 and row['original_spin_multiplicity']==c['spin_multiplicity']==1
                        by_id[row['task_id']]=row;xtb_attempts+=1
                        if m=='reference':
                            if row['success'] and not task['inversion_check']:refs[kind][seed,i]=row['energy_eV']/c['n_atoms']
                        else:
                            mi=methods.index(m);assert graph[seed,mi,i,j]
                            success[kind][seed,mi,i,j]=row['success']
                            if row['success']:
                                energy[kind][seed,mi,i,j]=row['energy_eV']/c['n_atoms']
                                f=np.asarray(row['force_eV_A']);force[kind][seed,mi,i,j]=np.sqrt(np.mean(np.sum(f*f,axis=-1)))
                    for i in conditions:
                        plus=by_id[f'reference_c{i}_s0_plus'];minus=by_id[f'reference_c{i}_s0_minus']
                        passed=plus['success'] and minus['success']
                        de=df=None
                        if passed:
                            de=abs(plus['energy_eV']-minus['energy_eV']);df=float(np.max(np.abs(np.asarray(plus['force_eV_A'])+np.asarray(minus['force_eV_A']))))
                            passed=de<=spec['xtb_energy_inversion_tolerance_eV'] and df<=spec['xtb_force_inversion_tolerance_eV_A']
                        inversions.append(dict(seed=seed,condition=i,passed=bool(passed),energy_error=de,force_error=df))
    assert esen_queries==33024 and xtb_attempts==2198
    summary={};contrasts={};rng=np.random.default_rng(41691);arrays=dict(graph=graph)
    for kind in ['esen','xtb']:
        arrays[kind+'_energy_per_atom']=energy[kind];arrays[kind+'_force_rms']=force[kind];arrays[kind+'_success']=success[kind];arrays[kind+'_reference_energy_per_atom']=refs[kind]
        excess=energy[kind]-refs[kind][:,None,:,None]
        summary[kind]={};contrasts[kind]={}
        for mi,m in enumerate(methods):
            valid=graph[:,mi]&success[kind][:,mi];e=excess[:,mi];f=force[kind][:,mi]
            summary[kind][m]=dict(attempted=int(valid.size),graph_valid=int(graph[:,mi].sum()),physical_success_on_valid=int(valid.sum()),
                graph_rate=float(graph[:,mi].mean()),valid_energy_excess_quantiles=np.quantile(e[valid],[.1,.5,.9]).tolist(),
                valid_force_quantiles=np.quantile(f[valid],[.1,.5,.9]).tolist(),
                energy_yield=[dict(threshold=t,rate=float((valid&(e<=t)).mean())) for t in spec['energy_thresholds_eV_atom']],
                force_yield=[dict(threshold=t,rate=float((valid&(f<=t)).mean())) for t in spec['force_thresholds_eV_A']],
                by_seed=[dict(graph_rate=float(graph[s,mi].mean()),physical_success_on_valid=int(valid[s].sum()),
                    energy_yield=[float((valid[s]&(e[s]<=t)).mean()) for t in spec['energy_thresholds_eV_atom']],
                    force_yield=[float((valid[s]&(f[s]<=t)).mean()) for t in spec['force_thresholds_eV_A']]) for s in [0,1]])
        for other in [0,2,3]:
            key='harmonic_fm minus '+methods[other];contrasts[kind][key]={}
            for metric,values,thresholds in [('energy',excess,spec['energy_thresholds_eV_atom']),('force',force[kind],spec['force_thresholds_eV_A'])]:
                entries=[]
                for threshold in thresholds:
                    qualified=graph&success[kind]&(values<=threshold)
                    diff=qualified[:,1].astype(float)-qualified[:,other].astype(float)
                    entries.append(dict(threshold=threshold,**interval(diff,rng)))
                contrasts[kind][key][metric+'_yield']=entries
    arrayfile=args.out.with_suffix('.npz');assert not arrayfile.exists();np.savez_compressed(arrayfile,**arrays)
    report=dict(complete=True,methods=methods,summary=summary,contrasts=contrasts,reference_inversions=inversions,
        reference_inversion_gate=all(r['passed'] for r in inversions),artifacts=artifacts,arrays_sha256=sha(arrayfile),
        new_raw_esen_queries=esen_queries,new_xtb_attempts=xtb_attempts,new_training_steps=0,new_neural_outputs=0,
        scope='Frozen matched-generator outputs. Joint quality yields use every attempted output as denominator; GFN2 scored graph-valid outputs only. Reference-relative composition energy is not graph-specific strain, equilibrium calibration or an energy-optimized generator. Threshold-wise intervals are descriptive, not simultaneous superiority tests.')
    write(args.out,report)
    print(json.dumps(dict(complete=True,reference_inversion_gate=report['reference_inversion_gate'],summary=summary),indent=2))


if __name__=='__main__':main()
