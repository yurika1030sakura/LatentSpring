# Next coverage test: lift terminal exchanges to pendant fragments

Status: IMPLEMENTED in `cfm_mol/fragment_exchange.py` and `fragment_sampler.py`.
Six targeted tests pass, including the full augmented Jacobian, known target,
force responses and physical-query accounting. The geometry screen and physical
pilot are recorded in `research/FRAGMENT_AND_BENCHMARK_STATE_20260912.json`.
The physical pilot includes a preserved Al validator failure; no fragment-specific
learner is trained yet. The design below records the implemented construction
and remaining learning/preparation work. It addresses a coverage gap: unlike
terminal H/halogen exchanges occur only in
development conditions0/4 and preserve the heavy-atom skeleton in practice.
Do not change the physical target to a fixed-skeleton target to hide that gap.

## Eligibility and physical prototype

Enumerate oriented single bonds whose removal disconnects a graph. The component
on the root side is a pendant fragment F_i, attached through root i to passive
anchor k. Pair two such fragments F_i,F_j only when they are disjoint, neither
contains either anchor, the anchors are distinct, and a passive core remains.
Canonicalize the unordered pair for bookkeeping only; selection must transform
under relabeling. Start with a prospectively bounded fragment-size screen,
without changing the general max_atoms200 architecture contract.

The desired graph replaces edges (i,k),(j,l) with (i,l),(j,k), retaining internal
fragment bonds. Require the endpoint's perceived bond matrix to match and require
the inverse action to recover exactly the same fragment atom sets. Same-element
roots are not automatically excluded: entire fragments and newly drawn geometries
can differ. Report graph-neutral moves and their physical shape displacement.

First implement a PHYSICAL control using the existing normalized radius/site
proposal for the two new root vectors and uniform torsion auxiliaries. Do not
blindly reuse the singleton guide's force labels for a multiatom fragment.

## Involutive coordinate lift

Let w_i=x_i-x_k and w_j=x_j-x_l be old root vectors. Draw new Cartesian root
vectors v_i,v_j (represented by log-normal radii and normalized sphere directions)
and two uniform circular auxiliaries delta_i,delta_j. For one fragment, set

    u = w / ||w||,  v = new_vector / ||new_vector||
    R = Rot(v,delta) A(u,v)
    y_a = new_anchor + new_vector + R (x_a-x_root),  a in F.

A(u,v) is the minimal proper rotation aligning the two unit vectors, and Rot is
a rotation about the new bond direction. Transform both disjoint fragments from
the ORIGINAL positions, then recenter COM. Passive relative coordinates remain
unchanged. Reverse auxiliaries are the OLD root vectors and -delta; reverse
anchors are exchanged. Because

    A(v,u) = A(u,v)^T
    Rot(u,-delta) A(v,u) = [Rot(v,delta) A(u,v)]^T,

the reverse map recovers every internal fragment coordinate. Near-antipodal
alignment requires an explicit symmetric rejection rule or a qualified chart;
do not choose an arbitrary lab-axis fallback that breaks equivariance.

In a linear chart with passive coordinates, root vectors and root-relative
internal fragment offsets, the augmented map swaps old/new root vectors, rotates
each internal offset with determinant1 and negates the two torsion auxiliaries.
Its intrinsic augmented absolute Jacobian is1. Dependence of R on old/new root
vectors occurs in off-diagonal derivative blocks. The COM chart factors cancel.
THIS DERIVATION MUST PASS A FULL AUGMENTED JACOBIAN AND INVERSE TEST BEFORE USE.

The MH correction uses the actual reverse/forward root-vector densities, uniform
torsion probabilities and action probabilities, plus the physical target ratio.
Each log-radius root law still has the Cartesian -3 log r term. Do not add the
old deterministic covalent-radius-swap Jacobian or treat a rigid-body proposal as
a full-dimensional endpoint likelihood for the generator.

Under an improper orthogonal transformation O, the root vectors transform with O
and the torsion auxiliaries change sign by det(O). Uniform torsions satisfy this
distributional symmetry. Tests must cover reflections and atom permutations as
well as proper rotations.

## Required tests and bounded molecular screen

1. Vectorized batched map and exact inverse for unequal fragment sizes and the
   singleton limit; source/passive coordinates and internal distances preserved.
2. Full Jacobian on (COM coordinates, six Cartesian auxiliaries, two torsions),
   not only determinant checks of the individual rotation matrices.
3. Known Gaussian target on COM space with a paired graph/action label, initialized
   from its exact joint distribution. Check moments and include a missing proposal
   density-ratio negative control. Do not initialize just one action label and
   mistake it for the joint stationary distribution.
4. Chemical graph inverse, atom inventories, charge/spin, symmetry and all random
   records. No molecular energy calls before these pass.
5. First census support on the existing independent-composition development
   inputs. Preserve validator errors and every failure; eligibility is chosen
   without energy ranking. Then freeze a small physical pilot with matched local,
   site and singleton controls before considering fragment-specific learning.

## Learning boundary

For an infinitesimal aligned-direction update at the source, the physical tangent
response involves both translation and torque:

    g_u = P_u [r sum_{a in F} F_H,a + tau_F cross u] / kT,
    tau_F = sum_{a in F} (x_a-x_root) cross F_H,a.

The torsional derivative is (tau_F dot u)/kT. These candidate labels must be
verified against independent differentiation of the full coordinate map.
The singleton case reduces to the existing angular label. A guide that sees the
old direction can have an unidentifiable concentration under one-point force
fitting; do not assume singleton masking automatically solves fragment conditioning.
Keep normalized proposal densities and full reverse-context evaluation regardless
of the eventual representation. No learning result is claimed by this design.

## Separate preparation-cost repair

`scripts/research/prepare_species_breadth_source.py` already saves coordinate
chunks BEFORE calling the physical oracle. A new, explicitly geometry-only source
entrypoint can preserve the FM64 midpoint/displacement/.025-A COM-noise law while
scoring only support-qualified teacher states afterward. It needs fresh disjoint
seeds, strict model/electronic provenance and replay before use. Keep the archived
source's4096 training-energy calls in its historical ledger; do not retrospectively
erase them. Measure new preparation costs, neural generation time, failures and
all generation attempts under a new protocol. This and a bounded shorter teacher
trajectory can test data efficiency without pretending that earlier costs were
never incurred.
