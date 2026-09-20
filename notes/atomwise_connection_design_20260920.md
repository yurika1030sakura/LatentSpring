# Atomwise correction candidate

TRAIN-only attribution motivates this study. Among512 cached native outputs per
family, FM has more connected heavy skeletons than GAGA(346 versus314), but more
hydrogen-only fragmentations(60 versus18). Its128 hydrogens without a heavy
contact are mostly just beyond the geometric threshold:119 lie within1.5 summed
covalent radii. These counts come from training_attachment_diagnostic_v1.json,
not a new fit on test structures. Earlier static-tree conditioning, hard tree
manifold, and endpoint-connectivity penalty pilots did not establish utility and
are not being repeated.

## Network change

The original pair head sums bounded messages weighted by soft predicted contacts
and divides all atoms by the largest soft degree in the molecule. A low-degree
atom can therefore receive little correction even when the per-molecule bound
is loose. The candidate divides each atom's message sum by its own soft degree.
Its row average has norm at most one. Subtracting the molecular average restores
zero total coordinate displacement; a factor of one half restores the original
per-atom bound,2*p^2 Angstrom per unit normalized progress.

Scalar features, embeddings, two vector channels, initial weights and parameter
count remain identical(7106 in the16-element vocabulary). The operations are
established graph normalization and equivariant centering. This is a targeted
architecture hypothesis, not a claim that row normalization or equivariance is
new, or that connectivity/valence is guaranteed.

## Target change

Let F be the centered force at the parent's provisional endpoint and P the
coordinate-centering projector. Set M_i=1/sqrt(norm(F_i)^2+F_floor^2), with
F_floor=1 eV/Angstrom. The balanced direction is P*M*P*F. Scale it by a positive
scalar to have exactly the original global-force displacement norm for every
training state. Thus no candidate gets a larger target-displacement budget.

The teacher remains downhill for linearized energy because
F dot(a*P*M*P*F)=a*sum_i M_i*norm(F_i)^2 >=0. It is also a local Gaussian
energy-tilt mean shift with covariance kT*a*P*M*P on the centered coordinate
space. This interpretation does not assert improved nonlinear energy, a new
fluctuation theorem, or the distribution of the finite learned sampler. The
network still has to learn a useful correction from these labels.

## Experiment

A2x2 factorial separates global/atomwise network normalization from global/
balanced physical targets. Original pair/global heads and every eSEN query are
reused. Each new head gets the same20000 updates, initial weights, training-state
order,96 training compositions and32 internal teacher-validation compositions.
Both independently fitted FM and GAGA parents receive all four variants.

Sixteen distinct development compositions from the prepared model-validation
set select variant and strength, with8 draws per composition/fit. These are
separate from the eight compositions used for earlier strength selection. The
common strength grid is0,1,2,4; zero strength is generated once per family.
There are6656 new validation outputs and fixed-coordinate GFN2 attempts,
240000 new head updates and no new backbone training or eSEN queries.

A32-composition fresh panel is fixed by metadata before this validation. It is
generated only if a new FM variant improves joint yield by at least2pp over its
validation-selected pair/global control, improves both fits, and does not lower
pooled graph validity. Test-time choices are then frozen. All four component
variants are evaluated at the chosen strength, together with each independently
selected original control if its strength differs. GAGA receives the same
selection choices. No raw-structure optimization or energy selection is used.

Tests verify equal displacement norms, positive linearized-energy descent,
symmetry, bounds, weak-contact message capacity and exact zero-correction
behavior of the native FM and GAGA samplers. Regression error alone does not
qualify a candidate; actual generation and independent physical readouts decide.
