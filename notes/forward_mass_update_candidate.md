# Candidate forward-mass update: explicit endpoint mixtures

This is a prospective learning experiment, not established AI novelty or an
ICLR-ready method. Additional fitting of the same Gaussian backward model did
not remove degeneracy at fixed forward samples. Test a direct weighted-CFM
update to the forward endpoint distribution instead of another force-only or
covariance-only update.

Energy-weighted flow matching and Markovian flow matching are prior work:
https://arxiv.org/abs/2509.03726 and
https://proceedings.neurips.cc/paper_files/paper/2024/file/bcd11db0b26d8fc2266b91d3ff982ed1-Paper-Conference.pdf.
Simply adding importance weights, tempering or a sample buffer is not new.

## Population identity and a relevant distinction

Let Q(x,z)=q(x)Q(z|x) be a frozen path proposal and R(x,z)=pi(x)L(z|x) its
normalized target path measure. The weight a=R/Q has E_Q[a|x]=pi(x)/q(x).
For a fixed eta in[0,1], affine weights (1-eta)+eta*a give endpoint law
(1-eta)q+eta*pi exactly in the population. In particular, q=pi is preserved
regardless of the auxiliary conditional L. This follows from linearity; it is
not a claim of a new importance-sampling identity.

In contrast, weights a^beta induce an endpoint law proportional to
q(x)^(1-beta) pi(x)^beta A_beta(x), with
A_beta(x)=integral Q(z|x)^(1-beta)L(z|x)^beta dz. This extra conditional-overlap
factor generally depends on x. Even q=pi can be distorted by power tempering
when the auxiliary conditional is imperfect. An exact four-state test exhibits
this effect; it is a mechanism check, not molecular evidence.

For N empirical normalized weights w, linear damping
v=(1-eta)/N+eta*w has sum(v^2)=1/N+eta^2*(sum(w^2)-1/N).
The largest eta retaining empirical ESS>=f*N is therefore available in closed
form. It is a training-step control, not an effective-sample certificate for pi.
Self-normalization and choosing eta from finite data introduce finite-pool
limitations. No unbiased finite-sample claim is made for the resulting student.

With an ideal endpoint projection, the population linear mixture contracts
Wasserstein-1 distance to pi by at most (1-eta). A learned student introduces
projection error; sampling discretization and stochastic-kernel mismatch are
part of that error. Neither perfect projection nor convergence is assumed here.
The final requested target remains pi, not an intermediate stabilized mixture.

## Frozen first experiment

Export4096 fresh paths from the original scalar-noise annealed-joint1500 forward
proposal, using the independently refitted backward mean and fitted variance
ratios in[.25,1.9]. Seed9084, batch64. Neither refit training nor its heldout
sample stream is reused. Compute full path weights at the original300-K eSEN
plus .1-restraint target. Budget4096 new oracle queries; source and refit costs
remain explicit. Keep full/uniform/linear/power weights. Linear and power arms
are matched at empirical ESS fraction .5; this does not make their targets equal.

The next student comparison must include uniform, linear-mixture and power
controls with shared initialization and training budget, independent Gaussian
starts, and fresh evaluation at the original full target. Do not claim the
stabilized teacher itself is a calibrated Boltzmann ensemble. A dedicated novelty
review and successful molecular/compute comparisons would still be required.

## Student protocol fixed before molecular outcomes

Source plus three independently fitted arms: uniform, linear and power.
All three students start from exactly the source forward checkpoint, run1000
AdamW updates at1e-5 with batch64, zero weight decay and gradient clipping1.
Sample endpoints from the frozen global4096 weights; do not renormalize within
minibatches. Use an independent intrinsic Gaussian prior with the original
standard deviation and a uniform linear interpolation time. Regress the actual
displacement velocity against x1-x0. Training seed9085 is shared across arms.
No reference coordinates, alignment, augmentation or score conversion is used.
This is projection onto an empirical target, with no finite-pool bias guarantee.

Evaluate256 new ODE paths with midpoint32 and64, sharing initial Gaussian draws
(seed9086). Report paired integration discrepancy and partial pair-marginal W1
to each arm's own empirical teacher; these are projection diagnostics only.
Query energies for64-step outputs. Separately generate256 Gaussian16-step paths
(seed9088) and score their full work at the original target. They are a different
sampler; an ODE improvement cannot be transferred to these weights by assumption.

Refit each arm's auxiliary reverse model identically:2048 independent fresh paths
(seed9087),500 updates at1e-5, batch16, selection seed9089, frozen forward state.
Fit time-dependent scalar variance on training paths within[.25,1.9]. Report
both before/after reverse-refit full-target work on the same256 heldout paths.
Heldout paths/energies never enter reverse fitting. Source control gets this
same evaluation/refit budget without forward updates. Each arm costs512 new
oracle calls plus explicit model evaluation/training costs. The teacher costs
4096 queries, prior source25024, and prior full reverse diagnosis256; earlier
smokes and historical method development remain additional, not hidden.

An end-to-end2-update smoke with128 reverse-training and64 evaluation paths
precedes these runs. Its128 oracle calls are engineering cost, not performance.
A single1000-update comparison is a screen, not replicated superiority. Stop
this frozen-pool recipe if full-target ESS remains below16/256 in every student,
or if geometry deteriorates materially. An ESS above that threshold would only
permit larger independent evaluation, not certify calibration or ICLR novelty.
Retain all arms, errors, failed denominators and costs.
