# Exact-entropy linear refinement baseline

This baseline complements the score-estimation study and uses no unqualified
critic. It is a standard normalizing-flow objective, not a proposed new identity.
It does not relax or bypass the score-based actor gate: no learned score enters.

Let Q0 be the frozen generator law on labelled COM-free H. For a deterministic
invertible map T and fixed target pi proportional to exp(-U/kT),
KL(T#Q0||pi)-KL(Q0||pi)
=E_Q0[(U(Tx)-U(x))/kT-log|det_H DT(x)|].
The unknown source entropy and target normalizer cancel. This is also a known
escorted-work change. Under the corresponding inverse-transformed auxiliary,
its conditional approximation gap stays constant, so that gap cannot confound
this transport-only objective. No numerical CNF trace or marginal score is used.

Use K=(W-diag(W1))/N, where W_ij=.25*tanh(a_type(i,j)) is symmetric and W_ii=0.
T=exp(K) acts identically on the three Cartesian columns. K1=0, so COM and
translation are preserved. Atom-type parameters commute with joint atom
permutations; the map is rotation/reflection equivariant. Its eigenvalues lie
in[exp(-.25),exp(.25)] by the complete-graph quadratic-form bound, hence it is
globally invertible. logdet_H T=3 trace(K). Scalar control ties every a to one
parameter; typed control has13 parameters for the current8-atom condition.
This is limited linear refinement, not a flexible general molecular generator.

Use the4096 unweighted source samples from forward_work_teacher_5846_v1 for
training and256 independent samples from forward_work_student_source_v1 for
assessment. Source control must have zero forward updates and matching checkpoint
and target hashes. The old work values carry through as W_new=W_old+Delta,
with exact log-volume and freshly checked energies. Keep importance ESS separate
from mean marginal-KL change. Initial/final energy queries use the same worker.

Prespecified runs: typed and scalar,200 AdamW updates, batch16, lr .01, clip10,
selection seed9141, identity initialization. Before full-run submission, fix a
cosine learning-rate decay from .01 to .0001 across the200 updates to reduce
late minibatch fluctuations in this stiff low-dimensional fit. The two-update
engineering smoke used constant .01 and is not performance evidence. Each uses3200 training+512 assessment
oracle queries=3712. All inherited base-training/data costs remain additional.
A2-update16-assessment smoke costs64 queries before these runs. Retain all cases,
energy-replay checks, maps, exact volumes, paired KL changes, weights and failures.
A statistically negative paired KL change is a refinement result only, not a
Boltzmann-calibration or matched-HMC superiority claim. Geometry needs independent
assessment. No wider scaling before these results and controls are audited.


The exact-volume energy objective is established Boltzmann-generator/normalizing-
flow methodology; the existing bibliography entry noe2019boltzmann is relevant.
The256 assessment samples are independent of adapter training, but belong to an
existing development panel. A positive result still needs fresh confirmation.
This is intentionally a strong simple refinement control before interpreting
more complicated score-based actor updates as useful.
