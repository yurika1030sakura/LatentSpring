from scripts.audit_iclr_evidence import group_scores, pearson, summary
from scripts.eval_xtb_relaxation import summarize


def test_negative_correlation_is_preserved_and_reference_excluded():
    records = {(0, i): {'log_p_theta': str(i), 'negE_kT': str(-i)} for i in range(5)}
    records[0, 0] = {'log_p_theta': '90000', 'negE_kT': '90000'}
    result = group_scores(records, {0})
    assert result['r'] == -1
    assert result['top1'] == 0
    assert result['median_nrv'] == 4


def test_constant_scores_are_not_successful_rankings():
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert summary([1])['sem'] is None


def test_relaxation_numerical_success_is_not_convergence():
    records = [{'failure_reason': None, 'converged': True, 'delta_E_kcalmol': 4,
                'delta_E_kcal_per_atom': 2, 'rmsd_A': 0.5},
               {'failure_reason': None, 'converged': False, 'delta_E_kcalmol': 16,
                'delta_E_kcal_per_atom': 8, 'rmsd_A': 1.0},
               {'failure_reason': 'single_point_failed'}]
    result = summarize(records)
    assert result['n_ok'] == 2
    assert result['n_converged'] == 1
    assert result['n_unconverged_with_energy'] == 1
    assert result['converged_delta_E_per_atom_median'] == 2
