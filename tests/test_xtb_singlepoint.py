import pytest
from cfm_mol.xtb_singlepoint import parse_singlepoint,HARTREE_EV,BOHR_A


def test_gradient_units_sign_and_fortran_exponents():
    out='*** convergence criteria satisfied after 5 iterations ***\nTOTAL ENERGY -1.000000D+01'
    grad='$grad\ncycle = 1\n0.0 0.0 0.0 h\n0.0 0.0 1.0 h\n1.0D-03 0.0 0.0\n-1.0D-03 0.0 0.0\n$end'
    result=parse_singlepoint(out,'normal termination of xtb',0,grad,2)
    assert result['success']
    assert result['energy_eV']==pytest.approx(-10*HARTREE_EV)
    assert result['force_eV_A'][0][0]==pytest.approx(-.001*HARTREE_EV/BOHR_A)
    assert result['force_eV_A'][1][0]==pytest.approx(.001*HARTREE_EV/BOHR_A)


def test_printed_energy_alone_does_not_qualify_convergence():
    out='TOTAL ENERGY -10.0\nnormal termination of xtb'
    assert not parse_singlepoint(out,'',0,'0 0 0',1)['success']
    out+='\nconvergence criteria satisfied\nSCC not converged'
    assert not parse_singlepoint(out,'',0,'0 0 0',1)['success']
    out='TOTAL ENERGY -10.0\nnormal termination of xtb\nconvergence criteria satisfied'
    assert not parse_singlepoint(out,'',0,'nan 0 0',1)['success']
    assert not parse_singlepoint(out,'',1,'0 0 0',1)['success']
