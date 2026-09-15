#!/usr/bin/env python3
"""GFN2 rotor table and fixed-budget escorted-work mechanism comparison."""
import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp
from cfm_mol.rotor_work import rotate_methyl,log_reference,escort,complete_work
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def analyze(energies,spec):
    kT=spec['kT_eV'];concentration=spec['reference_concentration'];size=spec['energy_grid']
    grid=np.linspace(-np.pi,np.pi,size,endpoint=False)
    full=CubicSpline(np.r_[grid,np.pi],np.r_[energies,energies[0]],bc_type='periodic')
    coarse=CubicSpline(np.r_[grid[::2],np.pi],np.r_[energies[::2],energies[0]],bc_type='periodic')
    offset=float(full(0.));potential=lambda x:full(x)-offset
    reference_grid=np.linspace(-np.pi,np.pi,spec['quadrature_grid'],endpoint=False)
    log_target=log_reference(reference_grid,concentration)-potential(reference_grid)/kT
    log_z=logsumexp(log_target)-np.log(len(reference_grid))+np.log(2*np.pi)
    probabilities=np.exp(log_target-logsumexp(log_target))
    bins=np.linspace(-np.pi,np.pi,spec['histogram_bins']+1)
    reference_hist=np.histogram(reference_grid,bins=bins,weights=probabilities)[0]
    half=reference_grid[::2];half_log=log_reference(half,concentration)-potential(half)/kT
    half_log_z=logsumexp(half_log)-np.log(len(half))+np.log(2*np.pi)
    truth=dict(log_normalizer=float(log_z),free_energy_eV=float(-kT*log_z),
        mean_energy_eV=float(probabilities@potential(reference_grid)),
        cos_moment=float(probabilities@np.cos(reference_grid)),sin_moment=float(probabilities@np.sin(reference_grid)),
        dense_quadrature_free_energy_error_eV=float(abs(kT*(half_log_z-log_z))),
        coarse_to_fine_max_energy_error_eV=float(np.max(np.abs(coarse(grid[1::2])-energies[1::2]))),
        coarse_to_fine_rms_energy_error_eV=float(np.sqrt(np.mean((coarse(grid[1::2])-energies[1::2])**2))),
        reference_histogram=reference_hist.tolist(),relative_energy_range_eV=[float(potential(reference_grid).min()),float(potential(reference_grid).max())])
    comparisons=[];quadrature=[]
    for amplitude in spec['escort_amplitudes']:
        y,logj=escort(reference_grid,amplitude)
        delta=potential(y)
        work=complete_work(reference_grid,y,logj,delta,kT,concentration)
        log_integrand=log_reference(reference_grid,concentration)-work/kT
        recovered_log_z=logsumexp(log_integrand)-np.log(len(reference_grid))+np.log(2*np.pi)
        masses=np.exp(log_integrand-logsumexp(log_integrand))
        quadrature.append(dict(amplitude=amplitude,normalizer_relative_error=float(abs(np.expm1(recovered_log_z-log_z))),
            cos_moment_error=float(abs(masses@np.cos(y)-truth['cos_moment'])),
            sin_moment_error=float(abs(masses@np.sin(y)-truth['sin_moment']))))
        # Same initial draws across escort settings; every method shares its Y.
        for n in spec['sample_sizes']:
            rng=np.random.default_rng(spec['random_seed']+n)
            x=rng.vonmises(0.,concentration,size=(spec['replicates'],n));y,logj=escort(x,amplitude)
            delta=potential(y);w=complete_work(x,y,logj,delta,kT,concentration)
            logs=dict(complete_work=-w/kT,energy_only=-delta/kT,without_jacobian=-(w+kT*logj)/kT,unweighted=np.zeros_like(w))
            for method,logw in logs.items():
                normalized=np.exp(logw-logsumexp(logw,axis=1,keepdims=True))
                ess=1/np.sum(normalized**2,axis=1)
                hist=np.stack([np.histogram(y[i],bins=bins,weights=normalized[i])[0] for i in range(len(x))])
                tv=.5*np.abs(hist-reference_hist).sum(1)
                free=-kT*(logsumexp(logw,axis=1)-np.log(n))
                cos=(normalized*np.cos(y)).sum(1);sin=(normalized*np.sin(y)).sum(1)
                comparisons.append(dict(amplitude=amplitude,n=n,method=method,
                    mean_total_variation=float(tv.mean()),tv_standard_error=float(tv.std(ddof=1)/np.sqrt(len(tv))),
                    mean_effective_sample_size=float(ess.mean()),
                    free_energy_bias_eV=None if method=='unweighted' else float((free-truth['free_energy_eV']).mean()),
                    free_energy_rmse_eV=None if method=='unweighted' else float(np.sqrt(np.mean((free-truth['free_energy_eV'])**2))),
                    mean_cos_error=float(np.abs(cos-truth['cos_moment']).mean()),
                    mean_sin_error=float(np.abs(sin-truth['sin_moment']).mean()),
                    mean_histogram=hist.mean(0).tolist()))
    return dict(reference=truth,quadrature=quadrature,comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    assert spec['frozen'] and not a.out.exists() and sha(spec['xtb_binary'])==spec['xtb_binary_sha256']
    assert sha(spec['source_data'])==spec['source_data_sha256']
    a.out.mkdir(parents=True);tasks=[]
    for rotor_index,rotor in enumerate(spec['rotors']):
        for i,angle in enumerate(np.linspace(-np.pi,np.pi,spec['energy_grid'],endpoint=False)):
            x=rotate_methyl(rotor['positions'],rotor['carbon'],rotor['anchor'],rotor['hydrogens'],angle)
            tasks.append(dict(task_id=f'rotor{rotor_index}_angle{i}',method='rotor',replica=rotor_index,parent_id=i,
                inversion_check=False,positions=x.tolist(),angle=float(angle),rotor_index=rotor_index))
    assert len(tasks)==spec['maximum_xtb_attempts']
    write(a.out/'tasks.json',tasks)
    report=dict(complete=False,protocol_sha256=ph,rows=[],scientific_submission_ready=False)
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        futures={pool.submit(run_task,t,spec['rotors'][t['rotor_index']]['condition'],Path(spec['xtb_binary']),a.out,spec['singlepoint']):t for t in tasks}
        for future in as_completed(futures):
            task=futures[future];row=future.result();row.update(angle=task['angle'],rotor_index=task['rotor_index'])
            report['rows'].append(row)
            if len(report['rows'])%64==0:
                write(a.out/'singlepoints.json',report)
                print(json.dumps(dict(completed=len(report['rows']),failures=sum(not r['success'] for r in report['rows']))),flush=True)
    report['rows'].sort(key=lambda r:(r['rotor_index'],r['parent_id']));report['complete']=True
    write(a.out/'singlepoints.json',report)
    failures=[r['task_id'] for r in report['rows'] if not r['success']]
    if failures:
        write(a.out/'results.json',dict(complete=True,qualified=False,failures=failures,raw_queries=len(tasks),
            protocol_sha256=ph,scope='Preserved failed energy grid; no spline or reference-distribution claim.',scientific_submission_ready=False))
        return
    studies=[]
    for rotor_index,rotor in enumerate(spec['rotors']):
        energies=np.array([r['energy_eV'] for r in report['rows'] if r['rotor_index']==rotor_index])
        studies.append(dict(rotor_index=rotor_index,reference_smiles=rotor['reference_smiles'],**analyze(energies,spec)))
    write(a.out/'results.json',dict(complete=True,qualified=True,raw_queries=len(tasks),protocol_sha256=ph,
        studies=studies,singlepoints_sha256=sha(a.out/'singlepoints.json'),tasks_sha256=sha(a.out/'tasks.json'),
        scope=spec['scope'],scientific_submission_ready=False))


if __name__=='__main__':main()
