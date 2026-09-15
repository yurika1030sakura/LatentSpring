# ICLR readiness checkpoint — refreshed September15

The internal manuscript has been rewritten around the supported harmonic-source
candidate. It compiles cleanly:6 main pages,11 total. See
`research/HARMONIC_PAPER_STATE_20260915.json` and `paper/tree_working.pdf`.
The research goal remains unachieved and scientific submission readiness is false.

Evidence now supporting the candidate:

- Harmonic source versus isotropic Gaussian: +5.86pp on12 development compositions;
  +8.83pp on10 additional compositions, conditional paired95[5.08,12.50]. Same two
  model seeds reused, with both improving. Additional compositions are17-28 atoms
  and absent from both verified processed corpora.
- Harmonic versus covariance Gaussian: +11.65pp and+15.39pp on the two panels;
  all four continuation signs positive and both conditional intervals above zero.
  The covariance control is approximate, with1.85–6.25% calibration error.
- A task-adapted original endpoint baseline is now measured and audited. Its
  different training history prevents a clean source-effect or SOTA claim.
- All5376 new control/native outputs replay. Jobs46504779/46506935 are complete.

The method-level contribution is a physics-informed, topology-marginalized source
for bond-free molecular FM. Existing random-tree theory, self-conditioning and
Jarzynski cannot be claimed as new. Whether this specific contribution is novel
and competitive enough for ICLR still needs stronger comparative evidence.

The measured physics application is the auxiliary Gaussian spring prior with
atomic scales. Jarzynski is retained as an optional correction theory with explicit
normalizability, invertibility and Jacobian/path-law assumptions. Current outputs
have not undergone that correction, and no Boltzmann-distribution or ESS claim
is supported. A new physical law is neither established nor a submission requirement.

Remaining substantive evidence: fair independent-generator comparisons, energy/
geometry quality and broader robustness in the stated task. Prior failed shell,
learned-source and neural-module experiments remain recorded. No new study is
currently queued; no reserved722 outcomes were queried. Human authors handle final
review and submission. No acceptance probability is assigned.
