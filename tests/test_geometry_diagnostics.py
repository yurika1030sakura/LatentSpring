import numpy as np
import pytest
from cfm_mol.geometry_diagnostics import distance_profile,profile_rms,contact_summary


def test_profile_is_element_permutation_and_rigid_motion_invariant():
    x=np.array([[0.,0,0],[1.,0,0],[0,2.,0],[0,0,3.]])
    z=np.array([6,1,1,8]);order=np.array([2,0,3,1])
    rotation=np.array([[0.,1,0],[-1.,0,0],[0,0,1.]])
    first=distance_profile(x,z)
    second=distance_profile((x@rotation+3)[order],z[order])
    assert profile_rms(first,second)==pytest.approx(0.,abs=1e-14)
    assert profile_rms(first,distance_profile(2*x,z))>0
    with pytest.raises(ValueError):profile_rms(first,distance_profile(x,z+1))


def test_contact_diagnostic_retains_disconnected_and_overlapping_geometries():
    result=contact_summary([[0,0,0],[.1,0,0],[5,0,0]],[.5,.5,.5])
    assert result['overlap_pairs']==1 and result['contact_components']==2
    assert result['min_covalent_distance_ratio']==pytest.approx(.1)
    singleton=contact_summary([[0,0,0]],[.5])
    assert singleton['contact_components']==1 and singleton['overlap_pairs']==0
    assert singleton['min_covalent_distance_ratio'] is None
