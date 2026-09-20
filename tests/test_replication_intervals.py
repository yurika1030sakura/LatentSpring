import numpy as np
import pytest
from scripts.research.summarize_seed_replication import paired_intervals


def test_fitted_model_uncertainty_is_not_hidden_by_many_compositions():
    values=np.broadcast_to(np.array([-.2,0.,.3])[:,None],(3,64))
    result=paired_intervals(values,repetitions=4000)
    np.testing.assert_allclose(result['composition_ci95'],[1/30,1/30])
    assert result['crossed_fit_composition_ci95'][0]<0<result['crossed_fit_composition_ci95'][1]
    np.testing.assert_allclose(result['by_fit'],[-.2,0.,.3])


def test_equal_stratum_estimate_and_incomplete_data_guard():
    x=np.tile(np.r_[np.ones(20),np.zeros(44)],(5,1));strata=np.r_[np.zeros(20),np.ones(44)]
    result=paired_intervals(x,strata=strata,repetitions=1000)
    assert result['mean']==20/64 and result['equal_stratum_mean']==.5
    np.testing.assert_allclose(result['equal_stratum_crossed_ci95'],[.5,.5])
    x[0,0]=np.nan
    with pytest.raises(ValueError):paired_intervals(x)
