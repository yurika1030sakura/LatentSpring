#!/usr/bin/env python3
"""Replay all rotor singlepoints and independently check the work density ratio."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp
from ase.data import chemical_symbols
from cfm_mol.xtb_singlepoint import parse_singlepoint
from cfm_mol.rotor_work import rotate_methyl
from scripts.research.train_electronic_fm import sha
from scripts.research.run_rotor_work import analyze
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','run','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists()
    pp=a.project/'research/evidence/rotor_work_v1.json';spec=json.loads(pp.read_text())
    result=json.loads((a.run/'results.json').read_text());single=json.loads((a.run/'singlepoints.json').read_text());tasks=json.loads((a.run/'tasks.json').read_text())
    assert result['complete'] and result['qualified'] and result['protocol_sha256']==single['protocol_sha256']==sha(pp)
    assert result['singlepoints_sha256']==sha(a.run/'singlepoints.json') and result['tasks_sha256']==sha(a.run/'tasks.json')
    by_id={r['task_id']:r for r in single['rows']};assert len(tasks)==len(by_id)==1536
    for task in tasks:
        row=by_id[task['task_id']];rotor=spec['rotors'][task['rotor_index']];root=a.run/'details'/task['task_id']
        expected=rotate_methyl(rotor['positions'],rotor['carbon'],rotor['anchor'],rotor['hydrogens'],task['angle'])
        np.testing.assert_allclose(expected,task['positions'],atol=0,rtol=0)
        for file,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256'),('gradient','gradient_sha256')]:
            assert sha(root/file)==row[key]
        lines=(root/'input.xyz').read_text().splitlines()
        assert [line.split()[0] for line in lines[2:]]==[chemical_symbols[z] for z in rotor['condition']['numbers']]
        coords=np.array([[float(v) for v in line.split()[1:]] for line in lines[2:]])
        np.testing.assert_allclose(coords,expected,atol=1e-12,rtol=0)
        parsed=parse_singlepoint((root/'stdout.txt').read_text(),(root/'stderr.txt').read_text(),row['returncode'],(root/'gradient').read_text(),len(expected))
        assert parsed['success'] and parsed['energy_eV']==row['energy_eV']
        np.testing.assert_allclose(parsed['force_eV_A'],row['force_eV_A'],atol=0,rtol=0)
        assert row['original_charge']==rotor['condition']['charge']==0 and row['original_spin_multiplicity']==1 and row['uhf']==0
    density_checks=[];finite=[]
    for i,rotor in enumerate(spec['rotors']):
        energies=np.array([by_id[f'rotor{i}_angle{j}']['energy_eV'] for j in range(512)])
        replay=dict(rotor_index=i,reference_smiles=rotor['reference_smiles'],**analyze(energies,spec))
        assert replay==result['studies'][i]
        # Separate direct q_target / q_pushforward computation, without work helper.
        knots=np.linspace(-np.pi,np.pi,512,endpoint=False)
        energy=CubicSpline(np.r_[knots,np.pi],np.r_[energies,energies[0]],bc_type='periodic')
        x=np.linspace(-np.pi,np.pi,65536,endpoint=False);kT=spec['kT_eV'];kappa=spec['reference_concentration']
        q=lambda z:np.exp(kappa*np.cos(z))/(2*np.pi*np.i0(kappa))
        target=lambda z:q(z)*np.exp(-(energy(z)-energy(0.))/kT)
        truth=target(x).mean()*2*np.pi
        for amplitude in spec['escort_amplitudes']:
            y=x+amplitude*np.sin(x);jac=1+amplitude*np.cos(x)
            weight=target(y)/(q(x)/jac)
            normalizer=(q(x)*weight).mean()*2*np.pi
            error=abs(normalizer/truth-1)
            density_checks.append(dict(rotor=i,amplitude=amplitude,relative_error=float(error)))
        comparisons=replay['comparisons']
        for n in spec['sample_sizes']:
            zero=[r for r in comparisons if r['amplitude']==0 and r['n']==n and r['method']!='unweighted']
            assert all({k:v for k,v in r.items() if k!='method'}=={k:v for k,v in zero[0].items() if k!='method'} for r in zero[1:])
        for amplitude in [-.5,.5]:
            rows={r['method']:r for r in comparisons if r['amplitude']==amplitude and r['n']==2048}
            finite.append(dict(rotor=i,amplitude=amplitude,
                full_TV=rows['complete_work']['mean_total_variation'],
                energy_only_TV=rows['energy_only']['mean_total_variation'],
                missing_jacobian_TV=rows['without_jacobian']['mean_total_variation'],
                gate=all(rows['complete_work']['mean_total_variation']<rows[m]['mean_total_variation'] for m in ['energy_only','without_jacobian'])))
    theoretical=all(r['relative_error']<spec['quadrature_tolerance'] for r in density_checks) and all(
        max(r['normalizer_relative_error'],r['cos_moment_error'],r['sin_moment_error'])<spec['quadrature_tolerance']
        for study in result['studies'] for r in study['quadrature'])
    write(a.out,dict(complete=True,raw_singlepoints_reparsed=1536,independent_density_ratio_checks=density_checks,
        complete_work_quadrature_gate=theoretical,finite_sample_checks=finite,finite_sample_gate=all(r['gate'] for r in finite),
        result_sha256=sha(a.run/'results.json'),protocol_sha256=sha(pp),scientific_submission_ready=False,
        scope='Controlled constrained-rotor work identity and finite-sample diagnostic on interpolated GFN2 potentials. It does not establish new Jarzynski theory, a neural-training advantage or global3D equilibrium generation.'))


if __name__=='__main__':main()
