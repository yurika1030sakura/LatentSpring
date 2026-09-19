# Physical correction learned on parent trajectories

The FM-trained correction head did not improve its frozen parent. This follow-up
changes its supervised states: the actual parent trajectory supplies the current
coordinates and provisional endpoint. It is a bounded candidate study, not an
adopted manuscript method. The parent-generated states are collected once; this
is not iterative on-policy training of the changing residual model.

## Input, target and generation

For each of128 TRAIN compositions, generate two unoptimized parent outputs with
32 midpoint steps. Record the second velocity evaluation at steps23 and29, whose
flow times are0.734375 and0.921875. For centered current coordinates X and parent
velocity v, H=X+(1-t)v is the provisional endpoint seen by the correction head.
There are256 generated TRAIN outputs and512 recorded states. The recorder copies
already evaluated tensors and does not add neural calls or modify the trajectory.

Query the inversion-averaged eSEN energy E+(H) and force F+(H). Subtract the force
centroid. At kT=0.0258519998 eV, choose sigma as the smaller of0.03 Angstrom and
sqrt(0.1*kT/||F+||_F). Translate a centered Gaussian by delta=sigma^2 F+/kT.
The Frobenius norm of delta is at most0.1 Angstrom. Four Gaussian draws and their
negatives give eight antithetic particles. Their unweighted mean is exactly
H+delta, so the force control has no Monte Carlo displacement noise.

Write each unshifted particle as U and shifted particle as Y=U+delta. The local
target is proportional to N_COM(Y;H,sigma^2 I) exp[-(E+(Y)-E+(H))/kT], restricted
to ||Y-H||_F<=1 Angstrom. Conditional on H and its force, delta is fixed and the
translation has unit intrinsic Jacobian. The complete log weight is

  -(E+(Y)-E+(H))/kT + (||U-H||_F^2-||Y-H||_F^2)/(2*sigma^2).

The second term corrects the Gaussian restraint/proposal shift; energy-only
weights would describe a different target. Self-normalized finite-particle
weights estimate the target's displacement mean. They are not an unbiased
finite-particle moment or proof that the trained generator has a Boltzmann law.
The bounded target assumes the energy is finite on its support. No chemical
validity mask is imposed; all parent states remain in the bank. A zero-support
particle set uses the declared force fallback, with its count reported.

Convert each displacement mean into a desired velocity correction by dividing
by1-t. Recenter and, if necessary, scale the whole molecular vector so its largest
per-atom norm is at most2*t^2. This preserves the centroid. The desired vector
need not be exactly expressible by the small head; training fit error is retained.

## Matched architectural controls

All four inference models use the same parent and8178-parameter pair head:
parent plus zero head, physical-reference FM head, trajectory-force head and
trajectory-work head. The scale is2 for every arm, including the FM control;
old scale1 results are not used as a controlled comparison. Every parent tensor
is frozen. Head initialization has a dedicated shared seed in all trained arms.

Each trained arm uses2000 updates with head learning rate0.0003 and two training
seeds. Cached-state fits avoid backbone evaluation during regression; the FM
control evaluates it twice per update. Teacher collection adds256 parent rollouts
and9216 physical queries. These costs and diagnostic forwards are counted; there
is no matched-total-compute or equal-wall-time claim.

Every fresh generation uses128 backbone and64 head calls, zero terminal noise,
and no physical oracle, filtering or coordinate optimization. A frozen16-composition
panel excludes prior generator panels and processed training corpora. Each method
has16 draws per composition in each of two runs. Fixed-coordinate GFN2 evaluates
all2048 attempted outputs. Primary evidence is trajectory-force versus parent;
work versus force separately tests the incremental local-work benefit. All arms,
seeds, failures and denominators are retained. Work has no automatic advantage.

The pairwise equivariance construction and frozen-model residual adaptation are
established techniques. Any contribution would require useful molecular results
for the specific composition-only architecture and supervision, plus comparison
to relevant prior methods. Passing mathematical/software tests alone is not such
evidence. Current41 relevant tests pass, including exact recorder neutrality and
analytic linear/quadratic local-work cases.
