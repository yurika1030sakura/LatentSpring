# Archived evidence audit

Generated from checked-in records; uncertainty is SEM across completed training seeds.
The legacy scalar readout is not a validated generator likelihood.

## Matched checkpoint resolution sweep

Same record keys and archived likelihoods verified before joining energies. Reference removed; archived 93-parent filter applied.

| Steps | FM r | Value r | Scrambled r | Value - FM | Value - scrambled |
|---|---|---|---|---|---|
| 12 | 0.2221 ± 0.0091 (5) | 0.3972 ± 0.0161 (3) | 0.2216 ± 0.0151 (3) | +0.1751 | +0.1757 |
| 24 | 0.1973 ± 0.0156 (5) | 0.2745 ± 0.0023 (3) | 0.2032 ± 0.0173 (3) | +0.0772 | +0.0713 |
| 48 | 0.1989 ± 0.0111 (5) | 0.2413 ± 0.0034 (3) | 0.1907 ± 0.0087 (3) | +0.0424 | +0.0506 |

## Generation strain (lower is better)

| Arm | Seeds | Mean strain, kcal/mol/atom | Failure rate | Not converged (all samples) |
|---|---|---|---|---|
| fm | 5 | 10.1590 ± 1.4186 | 0.130 | 0.433 |
| value | 3 | 14.7899 ± 4.3337 | 0.111 | 0.426 |
| scrambled | 3 | 14.0829 ± 3.2194 | 0.129 | 0.476 |

Equal failure rates do not establish absence of selection bias. Failed and unconverged cases remain distinct outcomes.
These observations do not establish improved generation quality or calibrated Boltzmann sampling.

Full per-seed results and input hashes are in evidence.json.
