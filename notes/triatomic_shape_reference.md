# Independent shape proposals for cold three-atom references

Reweighting the original independent Gaussian quadrature at 1000/300 K gives
minimum scramble ESS 8.10/1.88 and normalizer relative SE 16.5%/25.2%. Its
half-to-full resolution changes can reach .305/.916 nat. These points do not
provide an adequate cold reference. All original results remain retained.

For relative vectors a=x1-x0 and b=x2-x0, write r1=|a|, r2=|b| and c=cos(angle).
The Euclidean six-dimensional orthonormal COM-free measure satisfies

    dH = [8*pi^2/(3*sqrt(3))] r1^2 r2^2 dr1 dr2 dc

after integration over common orientation. The factor 3^(-3/2) comes from the
relative-vector to orthonormal-COM coordinate Jacobian, and 8*pi^2 from the
orientation integral. With u=(log r1, log r2, atanh c), the integrated volume is

    dH/du = [8*pi^2/(3*sqrt(3))] r1^3 r2^3 (1-c^2).

A normalized Gaussian mixture q_u therefore induces q_H=q_u/(dH/du). This is
ordinary change of variables, not a new identity. Density values refer to the
labelled Cartesian COM-free measure used by the existing reference. Canonical
representatives suffice only for rotationally invariant energies/observables;
the oracle rotation check remains required. An independent full six-variable
Jacobian test and known-Gaussian normalizer/moment integration check the factor.

The fixed pilot is the original independent quadrature, not generated FM data.
Choose its 64 lowest restrained-energy configurations, include both orders of
the identical outer Br atoms, and use an equal-weight mixture with standard
deviations (.05,.05,.5) in u. Combine it 50/50 with an isotropic Gaussian in H
of width sqrt(1/.1) A. Equal-size independent scrambled Sobol strata match this
mixture exactly. The Gaussian component retains support outside pilot regions;
positive support does not certify that every important mode was visited.

The first new run uses four scrambles and 2,048 points per stratum, totaling
16,384 new quadrature queries. Serial/batched/rotation checks add 24 queries;
the 12,304 original pilot queries are reported separately. Original electronic
metadata and checkpoint hashes are checked. The same new points evaluate both
300-K and 1000-K targets, so temperature estimates are correlated. Report all
resolution levels, between-scramble errors, weights and invariant moments.
Four-scramble agreement is statistical evidence, not a rigorous convergence
certificate or proof against a basin missed by all proposals.

The first four-scramble reference is complete, with normalizer relative SE4.18%
at300 K and1.45% at1000 K. The next prespecified refinement uses power12 and
eight scrambles, preserving all existing points and the same fixed proposal.
Nested coordinates must reproduce bitwise; cached arrays must reproduce their
published normalizers before energies are reused. Existing arrays lacked stored
output checksums, so this limitation is recorded; newly written arrays have
explicit SHA256 checksums. The extension reuses16,384 quadrature queries and
adds49,152 quadrature queries plus24 fresh consistency checks. Results from both
resolutions remain available; agreement does not by itself certify unseen modes.
