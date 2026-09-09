# Temperature, target definition and generation relevance

The existing computational target is explicitly

    pi(x|m,Q,S) proportional to exp[-(E_eSEN(x;m,Q,S) + .1/2 ||x||^2) / kT]

on orthonormal COM-free coordinates, with kT=1 eV. That energy scale corresponds
to about 11,604.5 K using kB=8.617333262145e-5 eV/K. OMol geometries are not
treated as samples at this temperature, and the restrained target is not an
unconfined gas-phase ensemble. The original scale was a numerical calibration.

New evidence shows why this scope matters. The independent four-scramble AgBr2
quadrature at the stated target has pooled mean closest/farthest Ag-Br distances
3.7143/7.0083 A, with delta-method standard errors about .0309/.0261 A across
scrambles. The stored OMol reference distances are 2.2825/2.4972 A. Mean radius
of gyration is 3.4852 A in quadrature versus 1.9369 A for that reference.
These means show an extended target; they do not determine dissociation/contact
probabilities, prove complete quadrature convergence, or establish the behavior
of the separate eight-atom condition. Source hashes and all moments are in
research/evidence/target_temperature_audit.json.

Likewise, the 1-eV FM-initialized MALA control worsens xTB relaxation strain and
convergence after its fixed budget. Finite-time MALA is not assumed equilibrated,
but this rules out attributing every loss of compact geometry specifically to
the learned work objective. xTB structure quality and agreement with this hot
eSEN Boltzmann target are different questions. Neither a lower eSEN energy nor
a higher xTB success rate alone establishes accurate target sampling.

## Prospective temperature diagnostic

Preserve all 1-eV runs. Add two explicitly different targets, at 300 K and 1000 K:
kT=.025851999786435 and .08617333262145 eV. Keep the same .1-eV/A2 restraint,
ordered composition, original charge/spin, eSEN checkpoint, 64 initial FM16
geometries, random seed 9061 and 132 MALA steps. Each target therefore uses 8,512
potential queries. Do not select the best-looking temperature post hoc as if it
were the original target, or call any of these finite-budget endpoints converged.

Set proposal standard deviation to .1*sqrt(kT/1eV) A and score-norm cap to
100*(1eV/kT) per A. The capped deterministic physical-force drift is then the
same across temperatures, while thermal proposal variance scales with kT. Both
directions use the actual proposal density in the MH correction. Report the
effective scales, acceptance, trajectory diagnostics and every xTB failure.
The harmonic restraint changes the physical ensemble even at 300 K and must
remain explicit. The same fixed 32 xTB indices and all-sample geometry checks
apply to every temperature.

FM initialization is held fixed: it comes from the checkpoint trained with its
constant requested-kT input of 1. No untrained temperature-conditioning feature
is changed merely by changing the MCMC target. Any future multi-temperature
neural training must separately specify this initialization/conditioning issue.
This diagnostic concerns target suitability and local thermalization; it is not
a claimed new sampling method, proof of correct mode masses or ICLR contribution.
