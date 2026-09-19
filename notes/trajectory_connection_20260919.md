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

## Completed comparison

Teacher47257342, fits47258410_0/47258728_1 and audit47258729 all completed.
Every parent tensor remains unchanged. All source draws, raw coordinate files,
local-work arithmetic, physical inputs/results and denominators are checked.
The joint metric is graph validity AND GFN2 force RMS<=5 eV/Angstrom.

| Model | Graph validity | Median valid force | Joint yield |
|---|---:|---:|---:|
| Parent |46.88%|2.737|42.38%|
| Physical-reference FM head |46.88%|2.822|41.21%|
| Parent-trajectory force head |47.46%|2.603|43.55%|
| Parent-trajectory work head |47.07%|2.624|43.55%|

The force head improves on the matched FM head by2.34 percentage points,
with composition-bootstrap95% interval[0.39,4.49] and positive changes in both
runs. Against the parent, its gain is1.17 points[-1.17,3.32], with run-specific
changes+0.78,+1.56. This is a positive point estimate, not an established parent
advantage. Work-minus-force is0.00 points[-0.78,0.78]; no incremental neural
work benefit is established. The primary gate is false. These intervals are
conditional on two head fits from one shared pretrained parent and this panel.

The teacher's median local particle ESS is1.37/8. No state requires the force
fallback and no target reaches the declared velocity cap. The force regression
reduces TRAIN error by only about5-6% from a zero head; work regression reduces
it by about2-3%. These are training diagnostics, not generator-quality evidence.
They motivate checking fit capacity/optimization before simply expanding data
or repeating the same small head. No change to the frozen evaluation is made.

The candidate is preserved as working model checkpoints in
runs/trajectory_connection_v1/s{0,1}/connection_{force,work}/last.ckpt. It is not
promoted into the manuscript's main method. The broad-label positive control
has already been published with its parent-comparison uncertainty. Full completed
costs are in registry v26, which carries forward the teacher costs from v25.
