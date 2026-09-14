# Separate internal geometry from velocity accumulation

Status: implemented, two focused tests pass, bounded comparison frozen; no
molecular improvement is claimed yet.

The coordinate-updating FlowMol backbone was designed as an endpoint predictor.
Our displacement adaptation interprets its final position increment as a velocity.
Its original internal coordinate states therefore need not be endpoint estimates.
Those states may still be useful arbitrary latent geometries; this is an inductive-
bias hypothesis, not proof that the old implementation was mathematically invalid.

For linear FM, the standard identity connects a velocity estimate V to endpoint
estimate Y=Xt+(1-t)*V. The new runtime patch keeps separate accumulators. Each native
position updater predicts a vector increment deltaV; V is incremented by deltaV,
and subsequent convolutions/edge-distance updates use Y=Xt+(1-t)*V. The returned
displacement head remains Xt+V, so the existing velocity FM target and sampler are
unchanged. Every learned module and parameter is reused. No new loss or parameters
are added. This exact relationship holds at every internal coordinate update.

The fixed-geometry control uses Y=Xt while accumulating the same velocity updates.
It distinguishes the proposed time-dependent geometry from simply avoiding moving
coordinates inside a velocity network. At t=0 the time-dependent path agrees with
the original coordinate-update network; at t=1 it agrees with the fixed-geometry
control. Tests check these limits, the internal identity, gradients, symmetry and
checkpoint restoration. Source-independent endpoint/velocity identities are
standard; their use here is not a new flow-matching theorem.

Protocols `research/evidence/dual_geometry_s{0,1}_v1.json` reuse exactly the same
warm checkpoint, clean TRAIN coordinates, fixed shell source, typed pairing,
optimizer,3000 updates and training seeds as the complete feedback study. The two
new models are compared with its unchanged one-pass and geometry-SC checkpoints,
using entirely new source streams. All have128 primitive denoiser evaluations.
The geometry-SC reference used more training compute, which stays reported.

The frozen gate requires time-dependent geometry to beat both original and fixed
geometry in both seeds, with positive pooled paired95 lower bounds and no >2pp
connectivity-diversity loss versus original. The stronger geometry-SC comparison
must also be reported, with no blanket superiority claim from this gate alone.
The panel is the reused12-composition monomer development set, not a fresh final
test. No envelope, step count, source, LR or data sweep is licensed by this pilot.

An audit of the original native endpoint checkpoint remains a distinct baseline
question. The30,000-step primary checkpoint restores successfully. Its native
sampling behavior must not be inferred from these adapted displacement models.
