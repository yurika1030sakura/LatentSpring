# Independent bounded-domain cubature cross-check

Four fixed-recipe HMC-SMC populations disagree strongly with each other and the
shape-QMC reference. Before assigning the discrepancy solely to Monte Carlo
noise, cross-check the coordinate measure and normalizer with direct cubature.

Use labelled distances r01 and r02 in [2,3.2] A, and cosine c in [-1,1]. Integrate
canonical eSEN energies against

    dH = 8*pi^2/(3*sqrt(3)) * r01^2 * r02^2 dr01 dr02 dc.

No sorted radii, permutation quotient, density evaluation, mixture balance
weight, logarithmic radius or atanh(c) transform is involved. Product Gauss-
Legendre orders 20 and 32 require 8,000 + 32,768 = 40,768 new oracle queries.
Compare to the existing QMC arrays restricted to exactly this box, retaining
all original QMC sample counts in the denominator. Actual charge, spin, eSEN
checkpoint, kT and restraint remain unchanged. The simple box volume and a
six-dimensional Gaussian integral are independent analytic tests of the factors.

This checks a finite-domain integral and numerical order stability; it cannot
certify total normalization or exclude important mass outside the box. Order
agreement is not a rigorous error bound. Keep both orders and every SMC seed;
do not select a reference retrospectively because it agrees with a candidate.


The completed order-20 and order-32 results differ from the same-box QMC value
by +.04961 and +.005303 nat respectively. This does not support a factor-of-two
volume error, but the order change is still measurable. Freeze one additional
order 48 (110,592 new queries) with exactly the same domain, potential and QMC
comparison; retain both prior orders. Do not reinterpret this as a full-domain
normalizer certificate.


Order 48 completed with difference +.007148 nat to restricted QMC; the 32-to-48
change is +.001845 nat. This is consistent with the QMC relative SE of 1.82% and
does not support a factor-of-two volume discrepancy. All three orders cost
151,360 new oracle queries. Neither agreement nor the finite box certifies the
full target or resolves the high variance of four small HMC-SMC populations.
