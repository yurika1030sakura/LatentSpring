# Boltzmann-Guided Flow Matching (BGFM)

Bond-free FlowMol3 on OMol25, with the loss family
`L_FM + lambda_1 L_force + lambda_2 L_energy (+ lambda_3 L_anchor)`.

**Research status, 2026-09-08: not submission-ready.** The archived results
show local ordering under a legacy scalar readout, not validated Boltzmann
sampling. The implementation/evidence audit found an endpoint-as-velocity
error in the energy routine, omitted trajectory/prior gradients, strong
solver sensitivity, and no demonstrated generation-strain improvement.

Start with [the audit](audit/20260908/REVIEW.md),
[recomputed evidence](audit/20260908/evidence.md), and
[the current project guide](CLAUDE.md). The previous broad consistency and
83-element sampling claims are not supported by these experiments.

## Reproduce the evidence

```bash
python scripts/audit_iclr_evidence.py --out /tmp/bgfm-evidence
```

No GPU, model download or ML library is needed. The script joins re-scores by
record identity, checks their archived scalar values, compares the same
checkpoints at 12/24/48 steps, and reports seed-level uncertainty and failures.
The group membership mask is an archived input, not a fresh identity audit.

## Corrected implementation

`cfm_mol/clamped_density.py` converts the actual endpoint head to velocity,
centers the differentiated field, uses deterministic midpoint stages, and
retains trajectory and prior gradients. It defines a clamped positional
`q_T`, default `T=0.95`; it does not score the joint CTMC/retracted sampler.
No reported molecular result was obtained with this correction.

The original routine is preserved for reproducibility. New training must
explicitly select `energy_density_options.mode: clamped_cnf` and use a new
run identifier. The smoke config under `configs/audit/` is experimental.

## Tests and manuscript

Use the flowmol environment, isolated from user-site packages:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/bgfm-mpl \
  envs/flowmol/bin/python -m pytest tests/ -q -o cache_dir=/tmp/bgfm-pytest
bash paper/build.sh /tmp/bgfm-paper-build
```

The paper is a working revision incorporating the audit, not a declaration
of ICLR readiness. The build enforces the nine-page main-text limit.
Keep fairchem in the separate omol25 environment. Keep generated data,
checkpoints and caches out of the home directory.
