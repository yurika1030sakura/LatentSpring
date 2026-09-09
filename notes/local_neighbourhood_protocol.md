# Local neighbourhood development protocol

The first old-checkpoint and new-position panels both have two unstable
parents under the five-sigma protocol [0.03, 0.06, 0.1, 0.2, 0.4] Angstrom.
The largest errors concentrate in the 0.4-A geometries, but one new-position
parent is also unstable at 0.1 A. This is a diagnostic finding, not permission
to remove failed cases from that reported panel.

A separately identified development branch uses only predeclared sigma
indices [0,1], i.e. 0.03 and 0.06 A, for EVERY parent. It targets local energy
ratios, not the original broad perturbation domain or inter-basin masses.
The dataset split and the full-domain failures remain unchanged. We check
64/128/256 quadrature on the same eight composition-disjoint parents. A
maximum mean centred change below 0.1 nat at the final refinement is the
initial numerical gate; this does not bound stochastic error or certify all
training geometries. Exact/adaptive comparisons are still needed.

The 4-step high-resolution runtime smoke uses 128 integration steps, two
conditionally independent replicas with common probes within each parent,
full trajectory/prior gradients and non-reentrant step checkpointing. Each
step's probes are cached, so backward recomputation neither changes the
objective nor advances global or dedicated trace RNGs. Tests compare actual
FlowMol parameter gradients and exact analytical gradients with and without
checkpointing. This costs additional network evaluations to reduce memory.

The smoke uses lambda_energy=0.001 and records the weighted auxiliary gradient
norm and its cosine with the FM gradient. This is calibration on training
geometries, not a selected performance result or the final hyperparameter.
It starts from the 1,000-step conditional baseline and uses four new FM data
indices, seed 9010. The final comparison must start matched arms from a
sufficiently trained common conditional baseline and include squared/common,
product/common, independent probes, zero and shuffled labels, and multiple
seeds. The longer FM-only baseline is running separately.
