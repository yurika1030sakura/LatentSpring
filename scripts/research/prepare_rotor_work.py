#!/usr/bin/env python3
"""Freeze three FIT-only methyl rotors using geometry, before energy queries."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.rotor_work import rotate_methyl
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();torch.set_num_threads(1)
    source=a.project/'runs/tree_manifold_v1/s0/data/data.pt'
    rows=torch.load(source,map_location='cpu',weights_only=False)['training']
    selected=[];seen=set();checked=0;decisions=[]
    for index,row in enumerate(rows[:128]):
        checked+=1;c=row['condition'];numbers=c['numbers'];x=row['positions'].double().numpy()
        if c['composition_hex'] in seen:continue
        graph=infer_chemical_graph(torch.tensor(x),numbers,c['charge']);b=graph['bond_orders'].numpy()
        candidates=[]
        for carbon,z in enumerate(numbers):
            neighbours=np.flatnonzero(b[carbon]>0).tolist()
            hydrogen=[j for j in neighbours if numbers[j]==1]
            heavy=[j for j in neighbours if numbers[j]!=1]
            if z==6 and len(hydrogen)==3 and len(heavy)==1 and all(b[carbon,j]==1 for j in neighbours):
                candidates.append((carbon,heavy[0],hydrogen))
        for carbon,anchor,hydrogens in candidates:
            accepted=True
            for angle in np.linspace(-np.pi,np.pi,64,endpoint=False):
                y=rotate_methyl(x,carbon,anchor,hydrogens,angle)
                try:g=infer_chemical_graph(torch.tensor(y),numbers,c['charge'])
                except (ValueError,RuntimeError,IndexError):accepted=False;break
                if not np.array_equal(g['bond_orders'].numpy(),b):accepted=False;break
            decisions.append(dict(training_row=index,carbon=carbon,anchor=anchor,hydrogens=hydrogens,geometry_qualified=accepted))
            if accepted:
                selected.append(dict(training_row=index,condition=c,positions=x.tolist(),carbon=carbon,anchor=anchor,
                    hydrogens=hydrogens,reference_bond_orders=b.tolist(),reference_smiles=graph['connectivity_smiles']))
                seen.add(c['composition_hex']);break
        if len(selected)==3:break
    assert len(selected)==3,'Insufficient predeclared geometry-qualified FIT rotors; do not select by energy outcomes'
    binary='/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/xtb'
    spec=dict(format='rotor_work_v1',frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_data=str(source.resolve()),source_data_sha256=sha(source),rows_scanned=checked,decisions=decisions,rotors=selected,
        xtb_binary=binary,xtb_binary_sha256=sha(binary),singlepoint=dict(accuracy=.1,timeout_seconds=90,restraint_eV_A2=0.),
        energy_grid=512,coarse_grid=256,kT_eV=.0258519998,reference_concentration=1.,escort_amplitudes=[-.5,0.,.5],
        sample_sizes=[8,32,128,512,2048],replicates=128,random_seed=38991,quadrature_grid=65536,histogram_bins=64,
        methods=['complete_work','energy_only','without_jacobian','unweighted'],workers=8,maximum_xtb_attempts=1536,
        reference='Periodic cubic interpolation of512 GFN2 energies;256-to512 interpolation and dense quadrature convergence are reported. This explicitly defined tabulated potential is the reference target.',
        target='Normalized q0(phi)*exp(-(U(phi)-U(0))/kT) with q0 a von Mises concentration1 reference on a fixed methyl-rotor curve.',
        primary='For every molecule and escort, complete work quadrature recovers normalizer and target moments. Monte Carlo error versus N and method is reported across all128 replicas; no choice of molecule, amplitude or replica by performance.',
        scope='Controlled low-dimensional molecular mechanism test of generalized escorted Jarzynski work; fixed internal coordinates, local restrained target, no neural generation or global3D Boltzmann claim.',
        scientific_submission_ready=False)
    write(a.out,spec)
    print(json.dumps(dict(rows_scanned=checked,rotors=[dict(row=r['training_row'],smiles=r['reference_smiles'],carbon=r['carbon'],anchor=r['anchor']) for r in selected])),flush=True)


if __name__=='__main__':main()
