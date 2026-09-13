# Claim and benchmark scope — user clarification, September12

## User priority correction: framework and fast AI evidence first

On September12 local time the user explicitly said that work had become too
heavy on details and that AI novelty, the whole idea and a fast demonstration
should take priority over making every case perfect. This overrides the former
plan to require further optimizer recovery before learning. No additional
mobility/constraint-recovery campaigns are prerequisites for the AI pilot.
Focus on a differentiated architecture, key correctness checks, a small real
molecular comparison and mechanism ablations; then deepen validation where the
claimed result needs it. Do not replace these with endless local diagnostics.
Data isolation, original electronic states, fair costs and honest limitations
remain mandatory. The full ICLR goal is unchanged. Current candidate and work:
`notes/edit_conditioned_bridge_v1.md`, `research/AI_FRAMEWORK_STATE_20260913.json`.

The user clarified that ICLR readiness means a convincing scoped contribution,
not a nearly perfect generator, universal physical superiority or an independent
new physical law. Retain the actual goal: correct theory/code, auditable original
electronic states and splits, meaningful molecular benefits against representative
strong baselines, and a reproducible paper. Do not conceal failures or claim a
finite-time Boltzmann law from correctness of MH alone.

The [ICLR2027 reviewer guide](https://www.iclr.cc/Conferences/2027/ReviewerGuidelines)
asks about the problem, motivation, evidence for claims and significance; it says
state-of-the-art results are not always necessary. AI-method novelty or useful
new knowledge, physically sound design and rigorous evidence are the intended
contribution. Neither a new physical law nor success on every element is a
separate mandatory deliverable. Acceptance is not guaranteed by beating a baseline.

## Two complementary comparisons

**Generation/correction quality:** compare the declared generator or generator
plus corrector against raw output and relevant learned alternatives. Available
artifacts include the FlowMol3-based positional FM64 initializer, the prior convex
adapter and published/compact EACF architectures. FM64 is not the native joint
FlowMol3 sampler and must not be labelled as such. Current EACF artifacts cover
condition0 only; this is a baseline implementation pilot, not broad validation.

**Sampling value of learning:** compare with strong physical MALA, force-angular,
site and fragment kernels on identical targets/conditions. Add learned sampling
proposals where interfaces and measures match. These controls isolate the value
of learning and do not replace generator comparisons.

`scripts/research/audit_generator_output_support.py` evaluates all512 common
development parents of the frozen generator baselines with a common structural
readout. It retains failures without relabelling original training targets or KL
quantities. Current MCMC output must be included only under a defined matching
output-count, source-attempt, correlation and cost contract.

## Relevant cost and quality regime

Report training/preparation once, inference separately, and measured total costs
at useful output volumes: C(N)=C_train+C_source(N)+C_inference(N). The fully charged
four-parent comparison is a stress test, not the only submission gate. Amortized
claims need fresh disjoint parent batches and measured reuse, not hypothetical
division of old costs by a selected sample count. Count neural generation time,
physical calls, unsupported source attempts and candidate checks consistently.

Keep all attempts in the generation readout. Measure structural validity,
validator errors, constitutional/conformational diversity and the declared
energy/quality target. Raw energies across different compositions/electronic
states are not comparable. Lowest energy is not equivalent to a300K distribution,
and correlated MCMC frames are not independent generated parents.

## Concrete comparable learned-sampler baseline

The EACF refiners inherit an unknown FM-source density. They cannot simply be
assigned an independent-proposal likelihood. Their audited forward/inverse maps
and joint volumes can instead support a direction-augmented MH proposal:

    draw a ~ rho(a | x), draw d uniformly from {forward,inverse}
    (y,b) = f_d(x,a)
    log R = -(U(y)-U(x))/kT + log rho(b | y) - log rho(a | x)
            + log |det D f_d(x,a)|.

The reverse direction is the opposite map and its auxiliary is b. This needs no
source likelihood. The auxiliary law/dimension and intrinsic volume must match
the actual upstream code: physical coordinates are COM constrained; auxiliary
coordinates may be unconstrained. Validate the known-target construction and
real forward/inverse interface, then freeze a bounded experiment. Keep both
published and compact capacities. This baseline is now implemented and audited;
`notes/eacf_directional_decision_v1.md` records the completed frozen-checkpoint
pilot. It is a repurposed refinement control, not native EACF/FAB performance or
a baseline trained for MH acceptance.

## Claim-specific gates

Mandatory: proof/code agreement, provenance, representative learned and physical
controls, reproducible gains for the stated task, appropriate cost accounting,
and honest limitations. Not mandatory: universal success across every element,
budget or molecular family; a new physical law; perfect mixing on all molecules.

For a finite-time equilibrium claim, obtain suitable reference and convergence
evidence. For first-passage or energy-aware generation claims, evaluate those
tasks directly without promoting them into stronger distribution guarantees.
Preserve the full original panel and freeze the final scope/evaluation before
reserved outcomes are opened. The Al validator failure remains recorded while
other compositions continue; it is not silently treated as a chemical rejection.

Deadlines remain September18/25 AoE, confirmed by the
[Dates page](https://iclr.cc/Conferences/2027/Dates) and
[Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines). A reviewer
FAQ inconsistently mentions September16. Use the dedicated submission pages,
which agree with the user's dates, and retain the discrepancy in the check record.
