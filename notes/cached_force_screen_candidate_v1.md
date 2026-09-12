# Candidate after the antisymmetric screen comparison

The frozen zero-initialized learned screens improve the internal per-query
proxy over no screening, but the fixed physical control has a higher point
estimate and the neural model is indistinguishable from four fitted coefficients.
This is not established AI benefit. Preserve the four-arm result and do not
scale those neural weights. Ordinary delayed acceptance is prior art; see
`delayed_acceptance_prior_art_20260912.md`.

## Additional available information

The current ChemicalTarget already stores an inversion-averaged force for every
scored state. Local MALA needs this force; it was paid for in the existing oracle
calls. The preceding screen used only geometry, perceived bonds and proposal
ratios. A source-force work estimate is therefore a distinct cheap input to test.
It is a first-order approximation, not exact finite-displacement work, and the
older local-work teacher failed as a proposal distribution. Do not assume that
forces guarantee an effective screen.

`runs/screen_force_pairs_v1` attaches cached forces to all existing 1,671 attempts
(1,597 scored). It preserves the 36 FIT / 12 internal-selection split and all
failures. All 3,233 distinct used states reconstruct the force and projected
score exactly from stored raw and inverted oracle forces. Original charge/spin,
coordinates and trace hashes are checked. No new physical queries are used.
The candidate endpoint force is available in the training/audit table but MUST
NOT enter the forward pre-query decision.

## Required generalization of the correction

A source-force screen need not be antisymmetric. Let g(x,y,a) in (0,1] be its
probability of passing the cheap test, computed using only x's cached force,
both geometries/graphs, electronic metadata and known proposal probabilities.
If this test passes, query the candidate energy/force, then compute the reverse
g(y,x,inverse(a)) using the newly available candidate force. With the complete
ordinary MH log ratio R, the correction is

    log alpha_2 = min(0, R + log g_reverse - log g_forward).

The total accepted probability is g_forward times alpha_2. This is ordinary MH
on the screened proposal q*g with rejected mass assigned to self transitions,
not a new theorem. The old R-s formula is valid only for its reciprocal-factor
special case. Using it with source-only work would be wrong.

A concrete physical input is

    w_source = sum[(F_even(x) - gamma*x) dot (y-x)] / kT.

Reverse work uses F_even(y) and is generally not minus forward work. Keep the
sign in F_even = (F_raw(x)-F_raw(-x))/2. Use labelled COM coordinates and the
fixed physical target. Force values are frozen inputs when differentiating the
learner; no gradient through an unavailable candidate oracle is permitted.

## Bounded execution and controls

1. First qualify the general two-stage identity with nonsymmetric gates on a
   finite-state system and actual molecular proposals. A zero-log gate must
   retain ordinary MH, RNG streams and query counts. Verify that candidate force
   is accessed only after a passing gate and that every rejection remains in
   denominators. Nonfinite reverse gates after a paid query must fail visibly.
2. Keep the base physical site-arc proposal and move-family schedule fixed. Compare
   no screening, the completed fixed physical screen, a source-work physical
   screen, a small fitted screen and any neural addition. Include constant
   random thinning calibrated from FIT data when comparing full chains, so a
   changed effective exchange frequency is not mistaken for selective learning.
3. Before fitting, choose the precise architecture, factor floor, objective and
   finite two-seed budget using only FIT data. Both forward and reverse gates
   enter accepted utility, while only the forward gate enters query cost. Report
   raw utility, acceptance, topology flow and whole-kernel costs, not only the
   conditional joint-query rate. No new force-screen model or protocol is frozen.
4. The existing `utility_onpolicy_v1` PHYSICAL proposals can provide a separate
   internal recorded-proposal check of frozen screens without new oracle calls.
   Freeze that analysis before evaluating the current models; retain all 1,152
   physical attempts and describe reuse explicitly. This is neither a fresh run
   of screened chains nor actual achieved savings. Never fit on those outcomes.
5. Only actual chain experiments with full timing/data costs, strong controls and
   an independent final cohort can qualify a useful method. Leave old evaluated
   molecular cohorts and the 722 reserved outcomes out of fitting. This candidate
   is an investigation, not proof that adding force features will create novelty.

## Implemented source-force experiment

`cfm_mol/source_force_screen.py` now implements positive nonreciprocal gates and
`joint_chemical_geometry.py` computes the reverse gate only after a successful
candidate query. It retains the reciprocal-factor branch unchanged. Tests verify
finite-state balance, sensitive parameter gradients, molecular/force symmetries,
zero-screen exact RNG/query equivalence, absence of candidate-force information
before querying, and a visible error after an invalid paid reverse gate.

The new paired-state neural encoder sees source/candidate geometry and graphs,
continuous atomic descriptors, source force/work/norm/displacement features and
symmetric leaf/anchor roles. A five-coefficient linear model supplies the simpler
learned control. Fixed controls include no screen, the previous physical screen,
and source-force work plus the EXACT known COM-restraint difference and proposal
ratio. Both learned models initialize at zero; old checkpoints are not loaded.

`runs/screen_prefix_accounting_v1/results.json` joins all 1,671 attempts to their
96 original physical trajectories, retaining initial queries and all nonjoint
work/costs. For the 72 FIT trajectories, mean cost is128 calls and mean expected
work is0.4947591 eV, giving a frozen baseline rate0.0038653057 eV/call. The new
objective weights complete trajectories equally, with fixed nonjoint terms,
rather than giving each variable-length joint-attempt population equal mass.
This is still a fixed-source empirical proxy, not a changed chain trajectory.

`research/evidence/source_force_screen_training_protocol_v1.json` freezes two
seeds each for linear/neural, 300 steps and batch eight. Initial/nonjoint costs
remain in evaluation. A constant thinning probability is calibrated from FIT
query use separately for each learned model, without selection outcomes. The
same bound, data and optimizer budgets are used; active parameter counts and
runtime are reported. No new physical queries are used. See NEXT for live jobs.
