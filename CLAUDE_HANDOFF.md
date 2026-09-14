# Claude continuation — main FM pairing experiment

Checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `research/NEXT.md`,
`research/ORBIT_PAIRING_STATE_20260913.json`,
`notes/orbit_pairing_fm_v1.md` and the frozen protocol first.

The main generator is now being trained with independent, standard rotation and
collision-aware symmetry pairings. Shared Haar symmetrization preserves the
Gaussian source; the endpoint data shape and chirality remain intact. Alignment
and physically motivated paths have close prior art. This is a concrete useful-
mechanism test, not an established ICLR contribution.

Current jobs: training46323226 (`runs/orbit_pairing_train_v3`), generation46323424
(`runs/orbit_pairing_eval_v2`), audit46324210 (`runs/orbit_pairing_audit_v1`). Query
live Slurm. Finish these and their summary before choosing another design.
All2048 generated outputs and all8 conditions stay in the readout. Only if steric
beats both continuation controls in pooled graph-supported count is the frozen
4096-call all-output physical energy check released. See NEXT for exact inputs.
Do not tune on evaluation outcomes or claim a thermal distribution from geometry.

45 unique targeted tests pass. The correlated pairing does not support the old
independent-Gaussian velocity-score proxy; training here is FM-only. Existing
BGFM hooks and separate environments are unchanged. No evaluated generated
parent, reserved outcome or fresh molecular oracle label enters fitting.

The previous coupled mass-estimator prototype is complete and audited. It has
no established distinct neural/molecular advantage and cannot fix marginal
importance-weight collapse. Previous routing/committee/scorer and toy-architecture
sweeps stay closed. Retain their full negative evidence. The ICLR goal is active
and unachieved; the old manuscript remains diagnostic.
