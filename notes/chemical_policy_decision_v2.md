# Chemical-policy decision after held-out evaluation

The learned selector improves the first connectivity transition but fails the
overall sampling/cost gate. Do not scale this recipe on condition0. All six
arms of job46039029 and the extra two physical controls46040976 are complete.
The implementation, raw paired outputs, proposal densities, graph assignments,
acceptance histories and all random streams replay. This is implementation
evidence, not independent quantum accuracy or equilibrium qualification.

## Positive and negative evidence together

The learned policies reach the reference connectivity in step1 for the same
three of four generated development parents in both repetitions. Uniform
0.5-local first hits are[30,29,null,22] and[83,19,null,29]; uniform0.1-local hits
are[9,6,null,19] and[5,1,null,2]. The difficult third parent remains unresolved
in every arm. These reference-connectivity hits are diagnostic: the target is
not a declared point mass on that graph.

At256 steps, learned-minus-uniform0.5 mean energy is+0.3015/+0.2255 eV. Against
uniform0.1 it is+0.0599/-0.1192 eV, hence mixed. Uniform final endpoints cost
10862/10876 or11770/11812 total raw queries, versus12560/12580 for learning.
Their mean final energies are lower by0.237--0.377 eV. The baseline endpoints
are cheaper, not exactly matched full-cost endpoints. This is enough to reject
an established overall superiority claim; it is not a theorem ranking their
equilibrium sampling error. No endpoint density, normalizer or physical ESS
has been inferred from these chains.

Actual new evaluation queries are45644. The shared training table and development
warm-up add5968; source preparation is inherited and additional. Dataset costs
are physically paid once, while each algorithm's ledger includes the preparation
it requires. The learned model also requires source-generation neural computation
and33.6/34.5 seconds of offline CPU training. Raw-query comparisons are not
end-to-end wall-time comparisons.

## The measured geometry bottleneck

The third parent's fixed-scale local proposals pass support only6/766 and12/751
times under uniform0.5. A new classical mixture chooses each local scale from
[0.1,0.03,0.01]*sqrt(kT), independently and with equal probabilities. Each
component carries its own forward/reverse Gaussian density.

Both256-step multiscale controls complete:1848/1838 new raw queries,2764/2754
including their required source and warm-up. Local support passes67/138 and
62/132 times for the difficult parent;66/62 moves are accepted. Its connectivity
still does not change. Final energies are-25503.7255/-25503.5911 eV for that
parent, whereas the other three are around-25506.5 to-25506.8 eV. Smaller steps
improve local motion without resolving the observed structural barrier at this
bounded duration. This does not prove that longer local sampling can never work.

Next investigate a proposal that couples the chemical exchange to geometry
adaptation. Do not add capacity to an action selector and assume it can compensate
for inadequate coordinate moves. A finite nonequilibrium path with exact reverse
path probability is a concrete next physical control; it is not yet implemented
or evidence of AI novelty. See `nonequilibrium_chemical_candidate.md`.

## Broader chemistry coverage is also unresolved

The no-energy-outcome census checks all4096 training and512 development parents
in each of the eight original development conditions. Only conditions0 and4
have any eligible H/halogen exchanges. Training graph successes are
[34,2048,315,0,553,0,591,0]; these are successes of the old algorithmic support,
not quantum-validity labels. Re/Pt and some Al configurations trigger RDKit
algorithm errors, which are recorded separately from bond-assignment rejections.

The installed RDKit option `allowChargedFragments=True` assigns formal charges;
its alternative assigns radical electrons. This distinction is documented in
the [official RDKit API](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html).
The old mode systematically rejects the neutral odd-electron condition5.

`chemical_support_v2.py` adds an experimental neutral-radical fallback, total
electron/spin parity checks and a distinct validator-error class. It leaves the
old support and every old trajectory unchanged. Methyl-radical and ammonium tests
pass. A training-only prefix of32 geometrically supported condition5 parents
yields25 assignments, but all have3--7 radical electron flags. This is increased
graph representability, NOT a qualified chemistry or spin result. Do not adopt
this extended support in production merely because the count rises. Ionic
radicals, metal coverage and the installed validator's runtime limitations
remain unresolved. Bond supervision has not been introduced.

Evidence: `chemical_policy_decision_v2.json`, `chemical_policy_full_audit_v2.json`,
`multiscale_chemical_audit_v1.json`, `chemical_source_panel_audit_v2.json`, and
`radical_support_preview_v1.json` in `research/evidence/`.
