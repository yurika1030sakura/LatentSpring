import math

import pytest
from scipy.integrate import quad


@pytest.mark.parametrize('ratio',[.25,.7,1.,1.9])
def test_normalized_gaussian_auxiliary_weights_and_finite_second_moment(ratio):
    def log_weight(z):return -.5*math.log(ratio)+.5*(1-1/ratio)*z*z
    def integral(power):
        return quad(lambda z:math.exp(power*log_weight(z)-.5*z*z)/math.sqrt(2*math.pi),
            -math.inf,math.inf,epsabs=1e-10)[0]
    assert integral(1)==pytest.approx(1.,rel=1e-10)
    assert integral(2)==pytest.approx(1/math.sqrt(ratio*(2-ratio)),rel=1e-10)


def test_bounded_positive_variance_does_not_ensure_finite_weight_variance():
    # At ratio2 the squared-weight integrand is constant; above2 it grows.
    for ratio in [2.,4.]:
        coefficient=(1-1/ratio)-.5
        assert coefficient>=0
        def truncated(radius):
            return quad(lambda z:math.exp(coefficient*z*z)/(ratio*math.sqrt(2*math.pi)),
                -radius,radius)[0]
        assert truncated(6)>=2*truncated(3)-1e-10
