import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from cfm_mol.chemical_sampler import ChemicalTarget
from scripts.research.audit_masked_angular import ReplayOracle, equal
from scripts.research.fresh_reuse import run_budget
from scripts.research.joint_arc_budget import audit_ratios, diagnostics


@pytest.mark.parametrize('method,decoder', [('legacy_site64','site'),('arc_uniform','arc_uniform'),('arc_site','arc_site')])
def test_budgeted_chain_replays_mixed_moves_and_independent_joint_densities(method, decoder):
    mol=Chem.AddHs(Chem.MolFromSmiles('FCCF'))
    assert AllChem.EmbedMolecule(mol, randomSeed=25401)==0
    x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.float64);x-=x.mean(0)
    condition=dict(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],charge=0,spin_multiplicity=1)
    class Oracle:
        evaluated=70
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x)
            return .01*x.square().sum((1,2)),-.02*x
    target=ChemicalTarget(Oracle(),condition,.026,.1)
    protocol=dict(query_caps={method:24},evaluation_seeds=[25451],scale_seeds=[25461],maximum_microsteps=128,
        joint_decoders={method:decoder})
    shared=dict(local_scales=[.1,.03,.01],schedule=['local','force_rotation','joint_exchange','local'],
        radial_width=.05,site_concentration=64.)
    saved=dict(raw_query_offset=70)
    run_budget(target,[x,-x],[3,5],None,method,0,1,protocol,shared,saved)
    saved.update(states=target.states,query_trace=target.query_trace)
    assert saved['final_queries_per_parent']==[24,24]
    assert target.oracle.evaluated==118
    oracle=ReplayOracle(saved['query_trace']);oracle.evaluated=70
    replay=ChemicalTarget(oracle,condition,.026,.1);actual=dict(raw_query_offset=70)
    run_budget(replay,[x,-x],[3,5],None,method,0,1,protocol,shared,actual)
    actual.update(states=replay.states,query_trace=replay.query_trace)
    equal(actual,saved)
    assert oracle.index==len(oracle.queries)
    assert audit_ratios(saved,replay,decoder,shared)>0
    result=diagnostics(saved,[8,24,32])
    assert result['all_caps_reached']
    assert [r['reached'] for r in result['endpoints']]==[True,True,False]*2
    assert result['transition_counts']['joint_exchange']['attempted']>0


def test_final_readouts_use_declared_caps_and_never_convert_censoring_to_completion():
    data=dict(parent_ids=[3,5],query_caps_per_parent=[4,6],final_queries_per_parent=[4,2],query_cap_reached=[True,False],
        query_count_history=[[2,2],[4,2]],history_state_ids=[[0,1],[2,1]],transitions=[],
        states=[dict(potential_eV=torch.tensor(v),graph={'connectivity_smiles':s}) for v,s in [(0.,'A'),(1.,'B'),(-1.,'C')]])
    result=diagnostics(data,[2,'final'])
    final=[r for r in result['endpoints'] if r['readout']=='final']
    assert [r['raw_queries'] for r in final]==[4,6]
    assert [r['reached'] for r in final]==[True,False]
    assert final[0]['potential_change_eV']==-1.
    assert 'potential_eV' not in final[1]
