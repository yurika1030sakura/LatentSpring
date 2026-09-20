"""Physical, connectivity and timing readouts from already audited raw outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','audit','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.project=a.project.resolve();a.run=a.run.resolve()
    spec=json.loads(a.protocol.read_text());audit=json.loads(a.audit.read_text());assert audit['complete'] and audit['protocol_sha256']==sha(a.protocol)
    assert not a.out.exists();file=a.audit.with_suffix('.npz');assert sha(file)==audit['arrays_sha256'];arrays=dict(np.load(file));methods=audit['methods']
    shape=arrays['graph'].shape;assert shape==(2,4,32,16)
    energies=np.full(shape,np.nan);seconds=np.zeros(shape[:-1]);distinct=np.zeros(shape[:-1]);distinct_joint=np.zeros(shape[:-1]);zero_radical=np.zeros(shape,bool);geometry=np.zeros(shape,bool)
    disconnected=np.zeros(shape,bool);overlap=np.zeros(shape,bool)
    for si in [0,1]:
        folder=a.run/f'test/s{si}';reportfile=folder/'xtb/results.json';assert sha(reportfile)==audit['provenance'][str(reportfile.relative_to(a.project))]
        physical=json.loads(reportfile.read_text())
        for r in physical['rows']:
            mi=methods.index(r['method']);i,j=r['condition_index'],r['sample_index']
            if r['success']:energies[si,mi,i,j]=r['energy_eV']/spec['test_rows'][i]['n_atoms']
        for name in ['fm','gaga']:
            path=folder/name/'generation/generation.json';assert sha(path)==audit['provenance'][str(path.relative_to(a.project))];report=json.loads(path.read_text())
            for r in report['rows']:
                mi=methods.index(r['method']);i=r['condition_index'];file=path.parent/r['file'];assert sha(file)==r['sha256']
                saved=torch.load(file,map_location='cpu',weights_only=False);seconds[si,mi,i]=saved['generation_seconds']/16
                x=saved['positions'];radii=covalent_radii(saved['condition']['numbers']);n=x.shape[1]
                d2=(x[:,:,None]-x[:,None,:]).square().sum(-1);length=radii[:,None]+radii[None,:]
                off=~torch.eye(n,dtype=torch.bool);over=((d2<(.6*length).square())&off).any((1,2))
                adjacency=d2<=(1.25*length).square();reached=torch.zeros(x.shape[:2],dtype=torch.bool);reached[:,0]=True
                for _ in range(n-1):reached=reached|(adjacency&reached[:,:,None]).any(1)
                disconnected[si,mi,i]=(~reached.all(1)).numpy();overlap[si,mi,i]=over.numpy()
                identities=[];qualified=[]
                for j,record in enumerate(r['records']):
                    geometry[si,mi,i,j]=record['geometrically_supported']
                    if not record['graph_supported']:continue
                    zero_radical[si,mi,i,j]=record['radical_electrons']==0
                    mol=Chem.MolFromSmiles(record['smiles']);assert mol is not None
                    key=Chem.MolToSmiles(mol,isomericSmiles=False,canonical=True);identities.append(key)
                    if arrays['success'][si,mi,i,j] and arrays['force'][si,mi,i,j]<=5:qualified.append(key)
                distinct[si,mi,i]=len(set(identities))/16;distinct_joint[si,mi,i]=len(set(qualified))/16
                np.testing.assert_array_equal(geometry[si,mi,i],~disconnected[si,mi,i]&~overlap[si,mi,i])
    summary={};joint=arrays['graph']&arrays['success']&(arrays['force']<=5)
    for mi,m in enumerate(methods):
        summary[m]=dict(**audit['summary'][m],geometry_rate=float(geometry[:,mi].mean()),zero_radical_joint_rate=float((joint[:,mi]&zero_radical[:,mi]).mean()),
            distinct_connectivity_yield=float(distinct[:,mi].mean()),distinct_joint_connectivity_yield=float(distinct_joint[:,mi].mean()),
            mean_generation_seconds_per_output=float(seconds[:,mi].mean()),generation_seconds_by_seed=seconds[:,mi].mean(-1).tolist(),
            mean_joint_outputs_per_sampling_second=float(joint[:,mi].sum()/(16*seconds[:,mi].sum())),
            disconnected_outputs=int(disconnected[:,mi].sum()),overlapping_outputs=int(overlap[:,mi].sum()),
            disconnected_and_overlapping_outputs=int((disconnected[:,mi]&overlap[:,mi]).sum()))
    rng=np.random.default_rng(61091);contrasts={}
    fm=methods.index('fm_a1');gaga=methods.index('gaga_a1')
    contrasts['fm_minus_gaga_distinct_joint_yield']=bootstrap(distinct_joint[:,fm]-distinct_joint[:,gaga],rng,20000)
    contrasts['fm_minus_gaga_seconds_per_output']=bootstrap(seconds[:,fm]-seconds[:,gaga],rng,20000)
    # All-output energy changes keep composition fixed but may change inferred
    # connectivity. They are not equilibrium free-energy differences.
    omitted={}
    for label,l,r in [('fm_energy_change','fm_a1','fm_a0'),('gaga_energy_change','gaga_a1','gaga_a0'),('selected_fm_minus_gaga_energy','fm_a1','gaga_a1')]:
        difference=energies[:,methods.index(l)]-energies[:,methods.index(r)]
        if np.isfinite(difference).all():contrasts[label]=bootstrap(difference.mean(-1),rng,20000)
        else:omitted[label]=dict(reason='Incomplete all-attempt energy cohort; no failed output silently dropped',missing_pairs=int((~np.isfinite(difference)).sum()))
    output=a.out.with_suffix('.npz');np.savez_compressed(output,energy_eV_per_atom=energies,seconds_per_output=seconds,
        distinct_connectivity_yield=distinct,distinct_joint_connectivity_yield=distinct_joint,zero_radical=zero_radical,geometry=geometry,
        disconnected=disconnected,overlap=overlap)
    write(a.out,dict(complete=True,source_audit_sha256=sha(a.audit),protocol_sha256=sha(a.protocol),summary=summary,contrasts=contrasts,
        arrays_sha256=sha(output),omitted_energy_contrasts=omitted,new_neural_outputs=0,new_physical_queries=0,
        scope='Descriptive additional readouts from the same unoptimized outputs. Timing includes native sampling and graph construction, excludes checkpoint loading, graph assessment and GFN2; each comparison is within the same allocated GPU for a seed. No end-to-end training-cost equality. Canonical connectivity counts exclude stereochemistry and measure16-draw duplication, not chemical-space coverage. All-output GFN2 energy changes are composition-paired diagnostics, not Boltzmann or free-energy claims.'))


if __name__=='__main__':main()
