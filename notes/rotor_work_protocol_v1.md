# Controlled molecular experiment for escorted Jarzynski work

User explicitly requested an experiment demonstrating the role of Jarzynski.
The test is a low-dimensional mechanism study, separate from full3D generator
quality and the deferred global stochastic-path sampler.

Three methyl rotors are selected from the existing FIT data, before energy queries:
training rows0,1,4, found within the first5 rows. Each passes the same labeled graph
check on64 equally spaced angles. Selection is by fixed row/atom order and geometry,
not energy, observed weight variance or estimator success. The fixed molecules are
C=C(N)N(C)C(N)=O, CCOC(C)(C)N, and CC(=O)NCC=O. Atoms other than the methyl hydrogens
remain rigid except for removal of overall translation. This defines a constrained
one-dimensional molecular curve, not unrestricted conformational equilibrium.

On angle phi in[-pi,pi), use normalized von Mises q0 with concentration1. Its
reference Hamiltonian is H0=-kT log q0. The target is proportional to
q0(phi)*exp(-(U(phi)-U(0))/kT), at300K. U is explicitly the periodic cubic interpolant
of512 GFN2 samples per rotor; report256-to512 interpolation error and dense
quadrature convergence. Total new singlepoints are1536, all retained. Any SCC
failure disqualifies the complete curve rather than being interpolated away.

The deterministic nonequilibrium escorts are y=x+a sin(x), a in{-0.5,0,0.5},
with positive Jacobian J=1+a cos(x). Complete generalized work is
W=U(y)-U(0)+kT[log q0(x)-log q0(y)-log J].
Thus E_q0 exp(-W/kT)=integral q0(y) exp(-(U(y)-U(0))/kT)dy.
This follows by change of variables and is the escorted-work formulation, not a
claim that an arbitrary learned vector field is literal molecular dynamics.

Compare complete work, energy-only weights, weights omitting the Jacobian, and
unweighted endpoints. The identity escort provides an exact implementation
control: all three weighted methods coincide. Draw counts8,32,128,512,2048,
128 fixed random replicates each, sharing draws across weighting methods and
escort settings. Report total-variation error on64 bins, free-energy bias/RMSE,
trigonometric-moment errors and local ESS. Unweighted endpoints have no free-energy
estimator and are not assigned a fictitious estimate.

Prespecified criteria: full-work normalizer and moment quadrature errors<1e-6
for all molecules/escorts; at2048 draws full-work mean TV below energy-only and
missing-Jacobian controls for each nonzero escort and every molecule. All outcomes
remain reported even if the criteria fail. No molecule, amplitude, temperature,
replica or sample-budget selection from outcomes. This tests complete-work
correction of a real-molecule constrained energy distribution. It does not prove
an extra neural-training advantage, global generator ESS or global3D Boltzmann
sampling. The existing work-versus-force-update control remains separate.

Two analytic tests passed: known periodic-potential quadrature identity with
missing-Jacobian control, and map derivatives/rigid fragment geometry. Credit
Jarzynski1997 (https://doi.org/10.1103/PhysRevLett.78.2690) and Vaikuntanathan and
Jarzynski2008 (https://arxiv.org/abs/0804.3055). Scientific readiness is not inferred
from a theorem or one successful diagnostic.
