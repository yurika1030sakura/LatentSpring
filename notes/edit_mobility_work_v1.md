# Rapid revision: learn correlated mobility from physical work

Current result: `research/EDIT_MOBILITY_STATE_20260913.json`. Repeated internal
one-step gains over scalar adaptation are now measured; the matched128-query
chain differences remain inconclusive. Frozen models are moving to new parents.

Subsequent strong controls supersede that next step: bare edits and small root
noise match the learned transfer performance. No useful learned-mobility
superiority is established. Stop scale-up and read
`research/CHEMICAL_WORK_POLICY_STATE_20260913.json` and
`notes/chemical_work_policy_candidate_v1.md` for the current edit-choice hypothesis.

The first actual proposal pilot rejects the uniformly mobile candidate as a
useful current method. Across384 attempts per arm, both collective models accept
only1, versus37 for the physical arc control. Root-only models accept31/40 but
do not establish a potential-decrease-per-query gain over the physical control.
All3,456 attempts,4,980 raw calls and2,478 scored MH ratios are retained and
audited. Tiny differences from zero-field bridges are not useful AI evidence.

The revised hypothesis concerns WHICH collective motions to enable, rather than
adding unrestricted Cartesian motion. The neural model learns graph-conditioned
co-motion vectors. It sees the chemical edit and can correlate movements of
related atoms while protecting the remaining geometry. The existing force field
is trained jointly from actual endpoint physical work. This is a candidate
architecture, not an established new general sampling principle or useful result.

For each graph pair/electronic state/time, let R project onto the edited-root COM
subspace and P onto the full COM subspace. Define

    A0 = R + sqrt(epsilon) (P-R),  epsilon=0.01,
    A_theta = A0 + sum_k c_k v_k v_k^T,
    M_theta = s_theta A_theta A_theta^T.

The vectors v_k are centered graph-network outputs with norms bounded by1;
|c_k|<=0.25/rank and s is in[0.5,2]. Initial coefficients and log scale are zero,
so all variants have exactly the same initial mobility. The fixed variant uses
A0; the scalar variant learns only s; the collective variant additionally learns
the co-motion vectors and coefficients. All share the same force initialization.
Scalar adaptation controls whether any benefit is merely step-size tuning.

M depends only on declared graphs, electronic state and mirrored bridge time,
not on active coordinates or momentum. Applied to both momentum kicks and
coordinate drifts, it preserves their triangular unit-volume construction.
M is PSD but need not be invertible: its determinant is NOT a proposal Jacobian.
The existing augmented correction still includes the chemical map's volume,
auxiliary momentum density and forward/reverse action probabilities. The model
is equivariant to atom permutations and rigid rotations. Do not make M depend
on the active coordinates without rederiving the actual map density.

The training proxy is mean softplus of dimensionless augmented work, plus a
smooth desired-graph distance penalty. Actual eSEN energies and conservative
inversion-averaged forces supply the endpoint value and parameter derivative.
The physical target stays unchanged at evaluation. Every training candidate,
including unsupported chemistry, is physically queried; no validity-conditioned
subset is silently fitted. This smooth proxy is not claimed to be an unbiased
gradient of hard-support acceptance or a certified hard-target KL.

`research/evidence/edit_mobility_work_protocol_v1.json` freezes three variants,
two seeds,100 updates and batch4. Every variant starts from the same root-trained
force checkpoint within its seed; initial mobility matrices match. Condition
updates cycle through the four FIT compositions. Source/direction/momentum draws
match across variants. Each arm has16 raw source checks plus800 training calls:
816 per model,4,896 total. No evaluated parents or722 reserved outcomes enter
fitting. A single oracle worker receives the full original electronic condition
on every sequential request; condition changes and all raw query outputs are saved.

Next test actual corrected proposals promptly with the fixed/scalar/collective
ablations, the physical arc and original root bridge. Any claimed gain must
survive the scalar and fixed controls. Keep training/preparation costs explicit
and proceed to short matched-budget chains only if the pilot supplies a useful
signal. Do not resume the paused19-arm optimizer cleanup or require all chemical
cases to work perfectly. The full ICLR goal remains active and unachieved.
