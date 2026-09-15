# Independent generator and relaxation diagnostics

## Conditional EDM baseline

Use the official EGNN/EDM implementation and available GEOM EMA checkpoint, with
file hashes frozen in `conditional_edm_s{0,1}_v1.json`. Credit:
https://github.com/ehoogeboom/e3_diffusion_for_molecules and
https://proceedings.mlr.press/v162/hoogeboom22a.html (MIT license retained upstream).

Train positions with the official variance-preserving polynomial2 noise schedule
and epsilon MSE, holding clean normalized atom categories fixed as conditions at
both training and sampling. Original joint category predictions are unused. This
is a conditional adaptation trained for our task, not exact conditioning of the
native joint model. Only neutral singlets in the shared12-element organic domain
are evaluated. There is no force, bond-label or energy supervision.

Each continuation uses the same3000 OMol25 FIT rows as its LatentSpring source
continuation,3000 optimizer updates, published1e-4 learning rate and fixed clipping.
EDM has2,381,566 parameters and one denoiser pass per update; LatentSpring has
5,904,369 and two. Warm-start histories and pretraining data differ. GEOM pretraining
overlap with this panel is not certified. Therefore this compares task-adapted
pretrained architectures under a shared continuation dataset, not matched total
training resources or native published/SOTA performance.

Sampling uses127 ancestral transitions plus one final observation (128 calls), and
also all1000 native schedule transitions plus one observation (1001 calls). The
latter prevents the comparison from relying solely on an accelerated baseline.
Report both with elapsed time, raw graph/geometry support and diversity. Identical
call count is not identical FLOPs. Native observation noise is retained; no xTB
optimization or energy selection is added. Fresh24 remains evaluation-only but is
already observed by earlier studies. No LR, step, budget or checkpoint selection
on these outcomes is permitted.

A CPU smoke check strictly loaded all pretrained weights, checked rotation and
permutation covariance at nondegenerate coordinates, and instrumented all128 actual
EGNN calls in one four-atom numerical draw. The raw EGNN output is checked before
upstream's historical NaN-to-zero fallback. This is an implementation check, not
a molecular benchmark result or additional independent model.

## Separate capped relaxation

The raw generator outputs remain unchanged. A separate ASE BFGS diagnostic on
copies uses force tolerance0.05 eV/Angstrom and maximum displacement0.1 Angstrom
per step, with observations at0,5,20 steps. It takes first four draw indices for
all24 compositions, both original harmonic/force-update models and both existing
continuations, plus24 original-reference controls:408 trajectories. No energy or
validity filtering chooses the trajectories. Every failure is retained.

Initial singlepoint values are reused from the audited GFN2 campaign. Subsequent
calls use the same `--grad` parser and original charge/spin. The optimizer, not
xTB's CLI, moves diagnostic copies. Log every call and trajectory, report graph
retention, aligned coordinate RMS displacement, force convergence and actual cost.
A one-step integration smoke adds one separately counted GFN2 call. These optimized
coordinates never replace raw generation or enter training. Force convergence
alone does not certify a positive Hessian or a stable minimum, and this diagnostic
cannot establish canonical sampling. No claim of few optimization steps is made
before examining the measured result.

Both studies are bounded and retain scientific_submission_ready=false until the
remaining claims have adequate evidence. They do not create an extra novelty claim.
