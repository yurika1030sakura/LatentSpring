"""Synthetic-molecule integration check; no production oracle or training data."""
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from cfm_mol.chemical_sampler import ChemicalTarget
from scripts.research.audit_masked_angular import ReplayOracle, equal
from scripts.research.audit_multicomposition_angular import independent_checks
from scripts.research.probe_multicomposition_angular import collect, diagnose


class QuadraticOracle:
    def __init__(self):self.evaluated=0
    def evaluate_chunked(self,x,max_request):
        self.evaluated+=len(x)
        return .1*x.square().sum((1,2)),-.2*x


def test_cached_warm_forces_and_all_new_queries_replay_against_known_energy():
    molecule=Chem.AddHs(Chem.MolFromSmiles('FCCF'))
    assert AllChem.EmbedMolecule(molecule,randomSeed=25231)==0
    x=torch.tensor(molecule.GetConformer().GetPositions(),dtype=torch.float64);x-=x.mean(0)
    condition=dict(numbers=[a.GetAtomicNum() for a in molecule.GetAtoms()],charge=0,spin_multiplicity=1)
    physical=dict(kT_eV=.026,restraint_eV_A2=.1)
    target=ChemicalTarget(QuadraticOracle(),condition,.026,.1)
    warm_states=[]
    for i,y in enumerate([x,-x]):
        state=target.coordinate_state(y)
        state.update(state_id=i,energy_eV=.1*y.square().sum(),force_eV_A=-.2*y,
            potential_eV=.15*y.square().sum(),score=(target.basis.T@(-.3*y/.026)).flatten())
        warm_states.append(state)
    warm=dict(parent_ids=[100,200],history_state_ids=[[0,1]],states=warm_states)
    protocol=dict(probe_seed=25211,warm_roots_per_parent=1,exchange_trials_per_parent=2,
        site_concentration=64.,radial_width=.05,fit_angle_rad=.15,check_angle_rad=.1,
        rank_threshold=1e-5,maximum_contexts_per_parent=5,maximum_new_raw_queries_per_parent=64)
    saved={};collect(target,warm,protocol,0,saved)
    assert all(s['query_batch'] is None for s in saved['states'][:2])
    calls=2*(sum(r['valid'] for r in saved['exchange_trials'])+sum(r['valid'] for r in saved['probes']))
    assert target.oracle.evaluated==calls and 0<calls<=128
    report=dict(diagnostics=diagnose(saved,physical,protocol))
    checks=independent_checks(saved,report,target,protocol)
    assert checks['identified_parameters']>0 and checks['joint_coordinate_densities']>0
    for row in report['diagnostics']:
        for check in row['checks']:
            assert check['force_score_mse']['fitted']<1e-16
            assert abs(check['actual_work_over_kT']-check['predicted_work_over_kT']['fitted'])<1e-8
    replay=ChemicalTarget(ReplayOracle(saved['query_trace']),condition,.026,.1)
    again={};collect(replay,warm,protocol,0,again);equal(again,saved)
    assert replay.oracle.index==len(replay.oracle.queries) and replay.oracle.evaluated==calls
