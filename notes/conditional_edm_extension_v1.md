# Give the independent baseline an adequate optimization budget

The first conditional EDM adaptation is audited but very weak: raw graph passes
9/2 per768 at128 calls,13/0 at1001 calls. This is not evidence that LatentSpring
beats native EDM. The requested conditional task and pretraining differ, and the
3000-step EDM continuation consumed only42 seconds per seed, versus570–590 seconds
for the source-FM continuations. Equal update count did not equal training effort.

A single fixed extension resumes each EDM model and its exact AdamW state for
27000 additional updates (30000 total), repeating the same3000 isolated FIT rows.
Learning rate, clipping, noise schedule, conditions and sampler remain unchanged.
All original logs/checkpoints/results are retained. Evaluate both128 and1001 calls
again, with the same frozen draw seeds. This is an explicit post-initial-result
baseline strengthening, not an untouched model-selection experiment or a matched
total-pretraining comparison. No further LR, step or architecture sweep is planned.
The publication should report the stronger final adaptation with its real budget
and history, and should not promote a failed short adaptation into a strong-baseline
superiority claim. The matched Gaussian/harmonic source comparisons remain primary.
