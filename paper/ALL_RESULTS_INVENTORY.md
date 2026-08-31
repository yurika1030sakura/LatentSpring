# Complete inventory of every completed evaluation

Generated 2026-08-19 by recomputing all 131 completed evaluations from their own per-record
files under `/n/holylabs/woo_lab/Lab/yulili/bgfm/runs`.  Two conventions are given because the
older rounds only have one:

* **dropref** — per-group Pearson r of `log q_theta` against `-E/kT`, computed on the displaced
  geometries only (the unperturbed data geometry, `pert_id == 0`, is a high-leverage point and is
  dropped).  This is the paper's convention.
* **withref** — the same statistic with the data geometry included, as stored in each run's
  `boltz_independent.json`.  Reported only where no per-record file exists.

PIPELINE VALIDATION: the recomputation reproduces the paper's published per-seed values exactly
for `clean_p0_A_fmonly` (+0.2340, +0.1995, +0.2317, +0.2207, +0.1973 -> +0.217 +/- 0.008) and
`clean_p0_E_force` (+0.1053, +0.1297, +0.0435, +0.0857, +0.0584 -> +0.085 +/- 0.016).

---

## A. Currently reported, no change needed

| family | seeds | parents | dropref r | where in the paper |
|---|---|---|---|---|
| `clean_p0_A_fmonly` | 5 | 120 | +0.2167 +/- 0.0078 | matched grid, cell A |
| `clean_p0_B_flat` | 4 | 120 | +0.1871 +/- 0.0298 | matched grid, cell B (5 launched, 1 diverged) |
| `clean_p0_C_scram` | 5 | 120 | +0.1847 +/- 0.0196 | matched grid, cell C |
| `clean_p0_D_energy` | 7 total | 120 | matched 3 = +0.421; all 7 = +0.4160 +/- 0.0166 | matched block (s2,s3,s5) in the grid; s6-s9 as the separate later block |
| `clean_p0_E_force` | 5 | 120 | +0.0845 +/- 0.0156 | matched grid, cell E |
| `clean_p0_F_both` | 3 total | 120 | matched 2 = +0.194; all 3 = +0.1967 +/- 0.0094 | matched block (s3,s5); s10 as the later block |
| `wide_a1_fm_only` | 5 | 120 (93 used) | 120: +0.2154; 93: +0.200 +/- 0.018 | primary endpoint, control arm |
| `wide_a3_energy_only` | 4 | 120 (93 used) | 120: +0.3968; 93: +0.369 +/- 0.023 | primary endpoint, value arm |
| `wide_a6_energy_only_shuffled(_stab)` | 3 | 120 | +0.2331 (2), +0.2957 (1) | scrambled control |
| `abl_a1..a6` | 9 runs | 30 | withref +0.109 to +0.422 | earlier round, App. provenance |

**Per-seed, cell D, all seven:** +0.4524, +0.3984, +0.4120 (matched) | +0.3533, +0.3737, +0.4515,
+0.4707 (later block).  **Cell F, all three:** +0.1786 (s3) and +0.2100 (s5) are the MATCHED pair, mean +0.194 as printed in the paper; +0.2014 (s10) is the later block.  (An earlier draft of this file assigned these the wrong way round.)

---

## B. NOW REPORTED (this section was written before they were added; both landed 2026-08-19)

STATUS: E2 is in the main text (Sec. 5.1) and App. B.9; the n240 family is in App. B.11 with its
contamination caveat.  Their disposition is "reported", not "to be added".

### B1. `clean_p0_E2_force` — a second force weight, lambda_1 = 0.01

**5 of 5 seeds completed, none diverged.**

| arm | lambda_1 | dropref r | per seed |
|---|---|---|---|
| A, no physics | — | +0.2167 +/- 0.0078 | .2340 .1995 .2317 .2207 .1973 |
| E, force | 0.1 | +0.0845 +/- 0.0156 | .1053 .1297 .0435 .0857 .0584 |
| **E2, force** | **0.01** | **+0.0799 +/- 0.0127** | **.1177 .0764 .0890 .0780 .0384** |

Two force weights an order of magnitude apart give the same answer, both about 0.13 below the
no-physics control.  This answers the standard objection that the negative force result reflects
one badly chosen weight.  It changes no conclusion; it removes a hyperparameter escape route.

### B2. `n240_*` — the contrast on a 240-parent population

| arm | seeds | dropref r |
|---|---|---|
| `n240_a1_fm_only` | 4 | +0.2220 +/- 0.0082 (.2403 .2139 .2302 .2036) |
| `n240_a3_energy_only` | 3 | +0.4172 +/- 0.0088 (.4181 .4320 .4014) |
| `n240_a6_energy_only_shuffled` | 2 | +0.2284 +/- 0.0102 (.2386 .2181) |
| `n240_a6_..._shuffled_stab` | 2 | +0.2446 +/- 0.0500 (.1946 .2945) |

Contrast **+0.1952, Welch t = 16.18**, and the scrambled control sits at +0.228, essentially on
the flow-matching control.  Both the ordering result and the mechanism result reproduce.

**MANDATORY CAVEAT, which is why this cannot become the headline.**  The primary endpoint is
restricted to the 93 parents outside the value term's perturbation-shard pool precisely because
one of the two shards read by every value-supervised arm was built from the split the evaluation
parents are drawn from.  A 240-parent population contains MORE of that overlap, not less, so this
family carries exactly the arm-specific contamination the primary endpoint was designed to
remove.  It is a supporting, larger-population replication with a known confound in the direction
that favours the value arm — it must be reported as such, never as a better-powered headline.

---

## C. Completed, deliberately excluded — record the exclusion, do not use for claims

| family | seeds | what it is | why excluded |
|---|---|---|---|
| `eval_res/p0_A_fmonly`, `eval_res/p0_D_energy` | 10 + 12 | resolution curve on the matched-grid cells at n = 4/12/48 | author decision: this finer-grid family was reviewed and set aside; the manuscript reports the two-row resolution table only |
| `eval_res/a1_fm_only`, `a3_energy_only` (+ the `__ode*` singles) | 12 + 9 + 6 | resolution curve on the earlier arms | same decision; the n = 4 and n = 12 rows in the paper come from this family, the n = 48 cell is not powered and carries no value |
| `OOD_tmqm_*` | 3 | transition-metal out-of-distribution probes, 30 parents | spin multiplicity is not passed to the teacher; the paper states explicitly that no transition-metal result is reported |
| `v2_forceonly`, `v3_energy`, `v3b_energy_hi(_epoch0)`, `v4`, `v4_stable(_tmqm)`, `onpolicy_4m_epoch0` | 8 | earlier configuration generations, 30 parents | superseded by the matched grid; different training configurations, not comparable |
| `eval_sweep/*` | 5 | the first capped-step sweep, 30 parents, all r < 0 | run before the step-based lambda schedule was fixed, so every arm was effectively flow-matching only |

Their numbers, for the record: tmQM energy +0.365 / +0.329 and force-only −0.158 (withref, 30
parents); v3 +0.413, v3b +0.400 / +0.322, v4 +0.091, v4-stable +0.172, on-policy epoch 0 −0.082;
sweep arms −0.396, −0.326, −0.221, −0.051, −0.104.

---

## D. What this inventory is for

Every one of the 131 completed evaluations now appears in exactly one of sections A, B or C.  No
completed evaluation is silently absent from the manuscript: the ones in C are named, numbered
and given a reason.

---

## E. Count reconciliation

Verified against the filesystem on 2026-08-19:
`find runs -maxdepth 3 -name boltz_independent.json | wc -l` = **131**, distributed
`eval_ours` 77, `eval_res` 49, `eval_sweep` 5.

Section totals: **A = 50** (29 matched-grid + 12 primary-endpoint arms + 9 earlier 30-parent
round), **B = 16** (5 for E2 + 11 for the n240 family), **C = 65** (49 resolution + 3
transition-metal + 8 preceding configuration generations + 5 capped-step sweep).
50 + 16 + 65 = **131**.

The earlier version of this file printed the `abl_*` round as 10 runs; it is 9
(a1 x2, a2 x1, a3 x2, a4 x1, a5 x1, a6 x2), which is where the off-by-one came from.
