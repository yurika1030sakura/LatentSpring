# Geometry feedback challenge against GAGA

The published matched EGNN benchmark favors GAGA. This follow-up tests whether
feeding a provisional geometry back into the same network improves raw molecular
generation. It is an experiment, with no claim of improvement before evaluation.

Both challengers retain the original 2,381,566 parameters and their initial values.
The first prediction produces a provisional endpoint. Its detached geometry supplies
the second existing invariant edge channel for a second prediction. Both predictions
receive the flow-matching target. Distance feedback uses pairwise squared distances;
tree feedback uses negative log inclusion probabilities in a regularized auxiliary
weighted-tree distribution. These probabilities use all atoms through a Laplacian
inverse. They describe auxiliary connectivity, not chemical bond labels or the exact
posterior of the unchanged source distribution. The first invariant channel still
describes the current EGNN coordinates.

There are two initialization seeds and two challenger settings. Each trains for
15,000 updates with 32 examples and two supervised network passes per example.
This equals the original baselines' 960,000 backbone example passes per model,
but uses fewer optimizer updates and example presentations. Wall time is measured.
The batch stream is exactly the first 15,000 batches of the corresponding original
30,000-update schedule; noise seeds and harmonic source/coupling are retained.
Every sampler spends 128 actual backbone calls per output. Four tests verify
tree probabilities, original-channel compatibility, EMA-copy isolation, equivariance,
finite training gradients, and explicit counting of both sampler passes.

The internal validation panel contains 32 compositions: eight in each of four
atom-count bins from 17 to 64. It is selected from the already withheld validation
rows by fixed composition hashes. Each model generates 16 outputs per composition.
The original GAGA checkpoint also receives a validation-only choice among sampling
maxima 350, 500 and 650; its trained timestep range remains 0 through 650.
One challenger setting and one GAGA setting are selected across both seeds, using
pooled graph validity and fixed tie rules.

A separate 32-composition confirmation panel is frozen before training. It uses
the previously reference-qualified pool, excludes the old 64-composition benchmark,
and has zero exact-composition matches in both complete processed corpora. The
original pool also excludes earlier development, fresh-physics and reserved panels.
The 722 reserved outcomes remain untouched. Confirmation generates 32 outputs per
composition and reports both challengers, original harmonic FM, selected GAGA and
GAGA650 (deduplicating identical settings). The primary comparison is selected
challenger versus selected GAGA. It requires a positive gain in both model seeds
and a positive lower bound of the paired composition-bootstrap 95% interval.
No optimization or energy selection is applied. Physical readout can follow a
useful raw-generation result; the present campaign uses no new energy queries.

Protocol: `research/evidence/gaga_feedback_campaign_v1.json`.
Panel: `research/evidence/gaga_feedback_panel_v1.json`.
Runner: `scripts/research/run_gaga_feedback.py`.
The Slurm array runs the two seeds independently, with distance, tree and validation
baselines in sequence within each seed. All source and protocols are snapshotted
before submission. Publication conclusions stay unchanged until audited results.


## Execution

Training source: a19c95f. Seed0 runs as46634916_0. Seed1's first launch46634916_1
failed in a startup assertion before model initialization: its original baseline
seed is41401, whereas the assertion had assumed40402. The replacement validates
seeds against the actual frozen baseline protocol and preserves every seed/data
setting. Replacement46635650_1 and confirmation code use source5ab8586.

Direct confirmation submission was rejected by gpu_test's submission-count limit;
no confirmation job was created by that attempt. Dependency dispatch46636422 uses
source de00842 and submits the already frozen confirmation job once both training
jobs finish successfully and release their GPU submission slots. Submission
receipts and all failed attempts remain under runs/gaga_feedback_v1. The paper's
claims and PDF are unchanged by these ongoing experiments.
