"""Strict frozen-geometry GFN2 output parsing; no implicit spin or optimization."""
import math
import re

HARTREE_EV=27.211386245988
BOHR_A=.529177210903
NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?'
ENERGY=re.compile(r'TOTAL ENERGY\s+('+NUMBER+r')')


def parse_singlepoint(stdout,stderr,returncode,gradient,n_atoms):
    joined=(stdout+'\n'+stderr).lower()
    if returncode!=0:
        return dict(success=False,failure=f'exit_{returncode}')
    if ('normal termination of xtb' not in joined or
            'convergence criteria satisfied' not in joined or
            any(s in joined for s in ['scc not converged','scf not converged','failed to converge'])):
        return dict(success=False,failure='unqualified_scc_termination')
    matches=ENERGY.findall(stdout)
    if not matches:
        return dict(success=False,failure='missing_energy')
    energy=float(matches[-1].replace('D','E').replace('d','e'))*HARTREE_EV
    triples=[]
    for line in gradient.splitlines():
        parts=line.replace('D','E').replace('d','e').split()
        if len(parts)!=3:continue
        try:triples.append([float(p) for p in parts])
        except ValueError:continue
    if len(triples)<n_atoms:
        return dict(success=False,failure='missing_gradient')
    force=[[-v*HARTREE_EV/BOHR_A for v in row] for row in triples[-n_atoms:]]
    if not math.isfinite(energy) or not all(math.isfinite(v) for row in force for v in row):
        return dict(success=False,failure='nonfinite_energy_or_gradient')
    return dict(success=True,energy_eV=energy,force_eV_A=force,
        maximum_force_norm_eV_A=max(math.sqrt(sum(v*v for v in row)) for row in force))
