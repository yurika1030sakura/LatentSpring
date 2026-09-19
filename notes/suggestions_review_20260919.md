# Suggestion review and integration

The files in `~/bgfm/suggestions` were treated as independent proposals, not a
chronological source of truth. The input archive hashes are retained in
`runs/suggestions_20260919/input_manifest.json`.

The prose revision is adopted after checking it against the implementation.
It explains the composition input, coordinate output, auxiliary source tree,
two-pass network, local force targets, and parameter subtraction before relying
on their names. The same-norm control uses the square root of summed squared
parameter differences, excludes buffers, and does not explicitly filter by
`requires_grad`; the wording now matches that code. The two sampler evaluations
use the same midpoint time. The curvature fit has a zero-curvature fallback.
The Overleaf collaborator commit `1b64efd` is preserved as publication history;
its substantive edits are retained and the accidental `Yul` BST prefix is corrected.
The AI-use statement is expanded to describe the actual research assistance, as
required by the verified 2027 policy; no claim of completed human verification is invented.

The completed five-run source study replaces the two-run abstract headline.
At 3,000 updates the mean source gain is 3.89 percentage points, four of five
runs are positive, and the between-run SD is 5.27 points. All three new runs
improve at 6,000 updates. Full results and the learning curves are included.
The completed GAGA feedback and corrected physical-transfer outcomes are also
reported. Neither establishes general superiority over GAGA.

## Direct weighted endpoints

The supplied ZIP and unified patch agree on `weighted_endpoint_fm.py` and
`train_weighted_minima.py`. The separate code-integration report describes an
earlier package with different filenames and a different set of tests; its
52-passed claim is not used as validation of the delivered patch.
The actual archive records a negative small radial-network result and a missing
FlowMol environment. These were interface limitations to resolve, not evidence
of a successful new primary method.

We imported the additive data store, native FM adapter, training/generation
drivers, diagnostics, and tests. In the actual torch2.2/DGL/FlowMol environment,
the two new test modules plus source-checkpoint tests pass: 27 passed.
The teacher weights are sampled once, not applied again to the loss. Graph IDs
are metadata; the model still receives atomic/electronic composition. Failed
positive-mass particles stop training. Per-anchor weights cannot be relabeled as
global weights. The original backbone, saved harmonic prior, typed pairing, and
two-pass FM remain in use, with no parameter subtraction in this branch.

The native pilot restores the exact 5.90M checkpoint. It compares the unadapted
parent, proposal-weighted restricted minima, and complete-work-weighted restricted
minima. Both adapted models use 500 updates, the same random seed and sampler,
and 0.005-Angstrom training endpoint smoothing. All three use zero terminal noise,
so noise removal is controlled. The three compositions correspond to four fixed
two-torsion MMFF graphs, including two isomers of C4H10O.

| Native model | Graph-valid / 384 | Raw validity |
|---|---:|---:|
| Unadapted parent | 283 | 73.70% |
| Proposal-weighted minima | 334 | 86.98% |
| Complete-work-weighted minima | 325 | 84.64% |

All raw coordinates and graph assays are retained. This is a closed-set pilot,
not held-out generalization, a full Cartesian Boltzmann result, or a GAGA win.
The complete-work branch does not beat the simpler endpoint control on validity.
Its relevant next test is graph- and symmetry-aware basin probability plus raw
geometry, followed by a full-dimensional training teacher if that passes. The
supplied angular-network TV results are not presented as results of our native
Cartesian model. The pilot is not added to the main paper's claimed improvements.

Reusable small endpoint fixtures are under
`examples/weighted_endpoints/restricted_two_torsion`. Their reference coordinates
are restricted two-angle minima, not full Cartesian or DFT minima. The supplied
teacher records are reused; this pilot makes no new physical queries.

The independent fixed-coordinate GFN2 check is also complete (1,152 attempts).
Median residual force among each model's graph-valid outputs is 1.3043 eV/A for
the parent, 0.9485 for proposal minima, and 0.8375 for complete-work minima.
At force RMS<=1 eV/A, all-attempt joint yields are13.80%,48.70%,55.21%, respectively.
The complete-work minus proposal contrast is positive in each of the three
compositions at this threshold. The entire frozen threshold grid is retained,
including zero yield at0.1 eV/A for every model. This is preliminary evidence
of useful raw physical learning, not quantum-minimum convergence or calibrated
basin probabilities. Two further training/evaluation seeds use the same fixed
teacher bank, parent,500-update schedule and all controls; job47226653 runs them
and their independent GFN2 evaluations. No seed or budget is selected by outcome.

## Fair GAGA follow-up

The corrected full-strength update still harms both independent EGNN families.
A separate frozen protocol, `physical_strength_calibration_v1.json`, tests
alpha in {0, 1/16, 1/8, 1/4, 1/2, 1} for both families on internal validation.
Each family selects one alpha shared across its two seeds. The selection maximizes
all-attempt graph/force joint yield subject to preserving each seed's graph rate
within two percentage points of its unadapted parent. Zero is always an option.
Confirmation uses 16 previously ungenerated compositions, four in each size bin,
with 64 draws per seed. All attempts are saved. Confirmation is not used for
selection. Independent GFN2 is required before any two-potential improvement claim.

## Figure and scope decisions

PyMOL 3.1 ray-traces the original recorded coordinates with fixed orthographic
cameras. Atom identities, inferred final bonds, sample selection, and trajectory
frames are unchanged. Intermediate frames have no invented chemical bonds.
The overview and gallery use vector typography, restrained colors, and larger
molecular renders. Source training and physical-quality plots use consistent
colors, units, typography, and actual statistical intervals.

We retain Jarzynski as a qualified local-distribution extension. We do not promote
standard importance weighting or a pushforward identity to a new fundamental
theorem. Energy-weighted FM already addresses importance weighting
(https://arxiv.org/abs/2509.03726); the GAGA comparison uses the published
Gaussianity-aware truncation method
(https://proceedings.iclr.cc/paper_files/paper/2026/hash/08de3c1eb4adc7ac3c1949048422d895-Abstract-Conference.html).
Learnable escorts, input-dependent metrics, multi-temperature training, and DFT
sampling are not added simultaneously before the direct-target chain is validated.
