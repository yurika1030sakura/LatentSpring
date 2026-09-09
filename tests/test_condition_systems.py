import hashlib
import json

import pytest
import torch

from cfm_mol.condition_systems import load_condition,graph_from_condition
from cfm_mol.clamped_fm import clamped_fm_path
from test_clamped_density import Schedule


def condition():
    return {'atomic_numbers':[6,1,1,1,1],'n_atoms':5,'charge':0,'spin_multiplicity':1}


def test_condition_graph_has_no_target_geometry_and_preserves_state():
    g=graph_from_condition(condition(),['H','C'])
    assert g.num_nodes()==5 and g.num_edges()==20
    assert not g.ndata['has_reference_geometry'].any()
    assert torch.equal(g.ndata['electronic_state'],torch.tensor([[0.,1.,1.]],dtype=torch.float64).expand(5,3))
    assert not g.edata['e_1_true'].argmax(-1).any()
    with pytest.raises(ValueError,match='placeholder'):
        clamped_fm_path(g,torch.zeros(5,dtype=torch.long),Schedule())


def test_development_manifest_can_be_loaded_without_reserved_protocol(tmp_path):
    p=tmp_path/'development.json';p.write_text(json.dumps({'complete':True,'role':'new_development','rows':[condition()]}))
    row=load_condition(p,0)
    assert row['manifest_index']==0 and row['manifest_role']=='new_development'


def test_reserved_manifest_requires_matching_frozen_protocol(tmp_path):
    p=tmp_path/'reserved.json';p.write_text(json.dumps({'complete':True,'role':'reserved_evaluation','rows':[condition()]}))
    with pytest.raises(ValueError,match='frozen'):load_condition(p,0)
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'frozen':True,
        'condition_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'method_ids':['fixed_recipe']}))
    assert load_condition(p,0,evaluation_protocol=protocol,method_id='fixed_recipe')['charge']==0
    with pytest.raises(ValueError,match='does not match'):
        load_condition(p,0,evaluation_protocol=protocol,method_id='changed_recipe')


def test_graph_rejects_electron_parity_inconsistent_condition():
    row=condition();row['spin_multiplicity']=2
    with pytest.raises(ValueError,match='parity'):graph_from_condition(row,['H','C'])
