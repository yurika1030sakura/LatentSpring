# Method contribution decision — September10,2026

The preferred ICLR direction is a new learning method motivated and constrained
by molecular statistical mechanics. We do not need a new physical law, and the
AFM/Jarzynski link alone is not novelty. A strong physics analysis could support
a machine-learning contribution, but the current elementary identities and
single-condition failures are not sufficient as a standalone ICLR result.

The measured learning problem is allocating probability across configurations
at a cold, specified molecular target. Lower energy and plausible geometries
can coexist with weight collapse. The latest independent4096-path teacher has
raw ESS2.625; stabilizing its weights changes the training objective and does
not create missing target information. Same-family reverse refits failed.

The current three-student experiment is a bounded test of an existing weighted
CFM family, not a claimed novel algorithm. It asks whether direct endpoint
projection improves what mean-work training did not. Uniform, affine and power
controls, source control, independent full-target paths, equally budgeted
reverse refits, ODE resolution and all xTB attempts are retained. If all three
students remain ESS<16/256 after refitting, stop extending this frozen pool.

A future contribution must specify an actual learning intervention that
improves the reliability of target-mass information or the learned transport.
It must distinguish auxiliary path randomness from endpoint-distribution error,
state any stabilization bias, and retain probability accounting for the actual
sampler. Merely fitting a stronger reverse network, clipping weights, annealing,
adding diversity, or appending Gaussian noise does not meet the novelty bar.

Before promoting another algorithm:
- Demonstrate its predicted mechanism on a tractable distribution with a known
  answer, including a negative control and an auxiliary-model mismatch case.
- Pass numerical sampling/density checks for the real FlowMol interface.
- Beat direct relevant methods under matched total compute, including HMC and
  the pertinent EWFM/MFM/flow-perturbation components. The existing power arm is
  not an implementation of EWFM.
- Replicate distributional and geometric gains across predeclared conditions
  and seeds. Neither a toy nor AgBr2 alone establishes broad molecular sampling.

Primary prior art and exact scopes are recorded in
notes/forward_mass_update_candidate.md. The novelty assessment remains open.
The project is not scientifically submission ready; no acceptance probability
or broad target-calibration claim is justified by current evidence.
