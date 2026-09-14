# Geometry and unlabelled relation feedback for bond-free flow

Status: implementation and focused tests pass; a bounded four-control protocol is
frozen. No new generation result or ICLR-level novelty is claimed yet.

The earlier clamped research path disabled the native stochastic/history self-
conditioning branch for a memoryless field. This was a declared modeling choice,
not a hidden bug in its archived comparisons. Since output density is not the
current claim, generator quality now warrants a properly defined self-conditioning
baseline. Moreover, its native endpoint input must not be replaced by a raw
velocity-parameterized head output.

For the declared linear position schedule and any training coupling,
X1 = Xt + (1-t)*(X1-X0). The first displacement head predicts a velocity v0, so
use Xhat1=Xt+(1-t)*v0 as its conditional endpoint estimate. The first pass disables
native self-conditioning; the second receives this estimate. Atom categories
remain clamped, the legacy charge marker stays neutralized, and no chemical bond
labels are read. Both passes stay differentiable, with their mean velocity-FM
loss providing deep supervision. This defines a deterministic memoryless field;
there is no external previous-step history. Zero final native residual layers
preserve the initial one-pass field exactly.

The four controls are:
- plain: one-pass field;
- geometry: two-pass geometry-only feedback, with the edge interface clamped;
- latent: also feeds first-pass soft edge-output codes to the second pass;
- pooled: averages those codes over pairs within each molecule before feedback.

The latent and pooled arms train the existing edge-output head solely through
coordinate error flowing through the second pass. The bond classification loss
remains zero. These four channels have no assigned bond-order semantics; they are
learned relation features, not calibrated chemical bonds. The pooled control keeps
network capacity and gradients while removing pair-localized information. Geometry
self-conditioning alone is established prior art and cannot pass our novelty gate.
The candidate is useful differentiable relational feedback without bond supervision.
Neural relational inference, self-conditioning and recurrent pair representations
have prior art; comparative evidence is still essential.

All models use the same frozen clean TRAIN coordinate rows already selected for
the completed manifold study, but their stored geometric scaffolds are ignored.
All receive the same normalized fixed-shell source and typed rotation coupling,
3000 updates, base LR2e-5 and feedback LR3e-4. Feedback models have two differentiable
training passes, so training updates are matched but training compute is higher;
measured time and primitive forward counts must be reported. All generation arms
have128 primitive denoiser evaluations:64 midpoint steps for plain and32 for
feedback. All use the same64 fresh source draws per12 reused monomer development
compositions, with0.025-A terminal noise. No quantum queries are required.

Protocols: `research/evidence/geometry_feedback_s{0,1}_v1.json`.
The frozen primary gate requires localized latent codes to beat BOTH geometry-only
and pooled-code feedback in both seeds, positive pooled paired95 lower bounds over
both, and no >2pp drop in distinct connectivity per attempt versus geometry.
Plain is a baseline completeness check. A gain over plain alone does not establish
novelty. No post hoc width, LR, loss, sampler or sample-budget sweep on this panel.

The optional endpoint-tree adapter extension is implemented but is NOT part of
this four-arm experiment. It must not be mistaken for a tested success.
