# Claude continuation — context test negative; source geometry gain and assay mismatch

Checkout `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `research/NEXT.md`,
`research/TREE_TRANSFER_STATE_20260914.json` and
`notes/tree_results_and_scope_20260914.md` first.

The previously proposed tree-context network is now implemented and tested.
`cfm_mol/latent_tree_context.py` adds a zero-initialized invariant residual edge
adapter. The source tree follows the actual source permutation during FM pairing;
actual and independent-tree arms have identical adapter parameters, initialization
and budgets. Checkpoint restoration and missing-context/density guards work.
57 targeted tests pass. No shared FlowMol code or bond supervision was changed.

The matched generation experiment is COMPLETE and negative: graph passes/512 for
no context / actual / independent are118/102/119 and97/75/97. The actual-tree head
has no useful increment; second-seed intervals exclude zero negatively. All3,072
source/context trees and structural outputs replay, and full-sized checkpoints
restore. Do not scale this static adapter, vary its LR or gate to chase a win on
these outcomes. Retain its implementation and the negative result.

The source framework was then tested on32 prospectively selected additional
compositions without retraining. Gaussian / shell / harmonic graph passes/1,024
are180/200/183 and164/178/195. Shell geometry improves6.54 and5.86 percentage
points, with positive paired and descriptive composition intervals. Its graph-
gain intervals span zero. All6,144 outputs replay; no source is chosen per seed.

A reference assay check reveals a fundamental scope limitation: only6/32 original
reference graphs pass the same readout;20/32 pass geometry,12 references are
multifragment and12 cause validator exceptions. All16 all-arm-zero generated
conditions also have rejected reference graphs. This is not permission to change
primary counts or call rejected outputs chemically valid. Reference snapshots can
be reactive/distorted/multicomponent. Do not retroactively use the six passing
references as a new primary result. All32 references remain evaluation only.

Next is a prospective task/assay alignment for single-molecule generation, before
more neural architecture variants. Inspect original-reference eligibility and
metadata without output ranking, state any organic/monomer scope explicitly, and
keep the existing broad panel as a separate stress test. Resolve training overlap
before an unseen-composition claim. No replacement panel or successful architecture
has been invented here. The user's broad authorization permits bounded work
without another permission question; the ICLR goal remains active and unachieved.

Jobs46367714_0/1 and46368764_0/1 all completed0:0. Query live Slurm; no job from
these completed studies is intentionally left in the background. The two studies
add9,216 outputs, totaling19,456 tree-study outputs. Zero new physical calls;
the earlier energy readout remains5,120. Audits replay source draws and readouts,
not full training or final neural trajectories. `paper/tree_working.tex` is the
current internal manuscript with all results, proofs and limitations; it is not
submission ready. The older manuscripts remain historical.

Protect722 reserved outcomes, old12/18 cohorts,2,560 orbit outputs,19,456 tree
outputs and32 raw references from fitting. Keep bond-free OMol25, max_atoms200,
original charge/spin, separate environments and no home writes. The noisy T1
sampler has no qualified absolute density/ESS/Boltzmann law. Earlier failed
routing/collision/latent-mass sweeps remain closed. The user handles submission;
official abstract/paper dates remain September18/25,23:59 AoE.
