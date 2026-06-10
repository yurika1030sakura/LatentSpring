# BGFM: Boltzmann-Regularized Flow Matching for Universal-Neural-Potential Guided 3D Molecular Generation

> Working code accompanying the BGFM paper (under ICLR 2026 review).
> **Status:** code is patched against bugs flagged by external expert audit
> (2026-06-29); full v8 evaluations in progress.

---

## What BGFM is — and is *not*

BGFM is a **training-time physical regularization framework** for de novo
3D molecular flow matching. We take a FlowMol3-style flow-matching generator
as the backbone and add OMol25 neural-potential supervision so the model's
*coordinate density, conditional on molecular identity*, locally satisfies
the Boltzmann relation under the OMol25 potential.

**The defensible target distribution is:**
```
p_B(r | c, T) = Z_c(T)^{-1} · exp(-E_NP(r, c) / kT)
```
where `c` is the discrete molecular identity (atom counts, atom types,
total charge) and `r` are the 3D coordinates. The model learns a
position-only conditional density `p_θ(r | c)` and is supervised so that
it agrees with `p_B(r | c, T)` *up to a per-molecule additive constant*.

**BGFM does NOT claim:**
- exact natural Boltzmann sampling
- a full joint probability over arbitrary atom counts, compositions, charges,
  and coordinates
- supervised partition-function estimation
- a downstream-deployment guarantee in production

**BGFM DOES claim:**
- local conditional coordinate Boltzmann alignment under the OMol25 NP
- train-time OMol25 supervision rather than post-hoc relaxation
- a FlowMol3-style backbone plus a BGFM objective and geometric hooks
- force-score consistency at late flow-matching path times
- a FFJORD-style density-energy variance loss over precomputed perturbation
  clouds, removing the need to know `Z_c`
- a composition-conditioned offset anchor that stabilizes the unidentified
  per-molecule density offset

---

## What is novel here

1. **Per-molecule within-perturbation FFJORD variance loss**
   (`cfm_mol/bgfm_density.py:energy_consistency_loss_per_mol`).
   Because we take the variance across perturbations of the *same* parent
   molecule, the loss is invariant to any `c`-only additive constant —
   in particular to the unknown `log Z_c`. This is, to our knowledge,
   the first energy-consistency formulation that survives inside a single
   FM training loop on a universal neural potential.

2. **Composition-conditioned offset anchor**
   (`cfm_mol/log_z_predictor.py`). A small invariant aux network that only
   sees the atom-type histogram and total charge. It stabilizes the
   absolute density offset without claiming to estimate the true
   thermodynamic partition function.

3. **Score-from-velocity force consistency at late path times**
   (`cfm_mol/bgfm_loss.py:score_from_fm_velocity`,
   `cfm_mol/bgfm_loss.py:force_loss`). The implied marginal score is
   computed in closed form from the FM velocity at late but finite path
   times `t ∈ {0.70, 0.80, 0.90}`. This is matched against precomputed
   endpoint forces `F(x_1)/kT` — a late-time approximation
   `F(x_t) ≈ F(x_1)` that avoids querying OMol25 inside each training step.

4. **Optional temperature conditioning** (`cfm_mol/kt_conditioning.py`).
   A zero-init projection of `log kT` into the scalar feature stream,
   sampled log-uniform during training, so a single model serves
   `kT ∈ [0.025, 1.0]` eV at inference.

The neural backbone itself is **not** novel: it is the FlowMol3
GVP-Transformer (vendored in `baselines/flowmol3/`).

---

## Repository layout

```
bgfm/
├── cfm_mol/                   # BGFM extensions (our code)
│   ├── bgfm_density.py        # FFJORD log p, per-mol variance, anchor
│   ├── bgfm_loss.py           # force loss, score-from-velocity, total
│   ├── bgfm_train_hook.py     # patches FlowMol3 with BGFM losses
│   ├── kt_conditioning.py     # temperature conditioning (v8b)
│   ├── log_z_predictor.py     # invariant log-Z aux network
│   ├── perturbation_loader.py # loads precomputed perturbation shards
│   ├── physics.py             # OMol25 wrapper utilities
│   └── flow_model.py          # FlowMol3 monkey-patches (path/sampler hooks)
│
├── baselines/flowmol3/        # vendored FlowMol3 backbone (not in git)
│
├── configs/                   # training configs (yaml)
│   ├── omol25_4m_bgfm.yaml                          # FM baseline
│   ├── omol25_4m_bgfm_energy_v7c_from_fm.yaml       # v7c: force-only finetune
│   ├── omol25_4m_bgfm_energy_v8a_room_T.yaml        # v8a: full BGFM @ 300 K
│   ├── omol25_4m_bgfm_energy_v8b_T_conditional.yaml # v8b: T-conditional
│   └── omol25_4m_bgfm_energy_v8c_energy_only.yaml   # v8c: energy + anchor only
│
├── scripts/                   # training and evaluation
│   ├── precompute_energy_perturbations.py  # offline OMol25 perturbation shards
│   ├── run_train.py                        # Lightning training entry
│   ├── launch_omol25_bgfm_energy_h200.sh   # H200 training launcher
│   ├── eval_boltzmann_stage1.py            # Level 1: FFJORD log p eval
│   ├── eval_boltzmann_stage2.py            # Level 1: R² aggregation
│   ├── level2_relax_comparison.py          # Level 2: BFGS savings
│   └── level3_*.py                         # Level 3: MD-equivalence
│
├── paper/                     # ICLR 2026 paper (LaTeX + compiled PDF)
│   ├── bgfm_paper.tex
│   └── bgfm_paper.pdf
│
├── notes/                     # method derivations
├── CODE_MAP.md                # paper section → source file mapping
├── PROJECT_SUMMARY.md         # status, results
├── AGENTS.md                  # AI-agent guidance (locked decisions, gotchas)
├── CLAUDE.md                  # authoritative project guide
└── README.md                  # this file
```

`envs/`, `runs/`, `data/`, and `baselines/` are excluded from the
repository (see `.gitignore`). Reconstruct them via the scripts in
`scripts/download_*` and the conda environments in `envs/`.

---

## Environments

Two conda environments because of a torch / fairchem version conflict:

```bash
# training and inference
conda env create -f baselines/flowmol3/environment.yml -p envs/flowmol
conda activate envs/flowmol

# OMol25 data preprocessing and energy evaluation
conda activate envs/omol25
```

---

## Reproduction recipe

```bash
# 1. Get OMol25 data
sbatch scripts/download_omol25_4m.slurm
sbatch scripts/preprocess_omol25.slurm

# 2. Precompute perturbations (the σ ∈ {0.03,0.06,0.10,0.20,0.40} Å shards)
sbatch scripts/run_precompute_energy.slurm val   10000
sbatch scripts/run_precompute_energy.slurm train 30000

# 3. Train one BGFM variant (v8a: full BGFM @ room T)
sbatch scripts/launch_omol25_bgfm_energy_h200.sh \
       configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
       "" 42

# 4. Boltzmann correlation eval (Level 1)
sbatch scripts/run_boltzmann_eval.slurm \
       runs/omol25_4m_bgfm_energy_v8a_room_T/.../last.ckpt \
       configs/omol25_4m_bgfm_energy_v8a_room_T.yaml

# 5. Downstream relaxation comparison (Level 2)
sbatch scripts/run_level2_relax.slurm <ckpt>

# 6. MD-equivalent sampling efficiency (Level 3)
sbatch scripts/run_level3_efficiency.slurm <ckpt>
```

Total compute on H200: ~45–50 h per training seed; ~2 h Level 1 eval;
~10 h Level 2; ~5 h Level 3.

---

## Loss terms (quick reference)

| Term | Code | What it does |
|---|---|---|
| `L_FM` | FlowMol3 baseline | Standard flow-matching reconstruction (positions + atom-types + charges) |
| `L_force` | `cfm_mol/bgfm_loss.py:force_loss` | Cosine between FM-implied score at `x_t` and OMol25 endpoint force `F(x_1)/kT` |
| `L_energy` | `cfm_mol/bgfm_density.py:energy_consistency_loss_per_mol` | `Var_k(log p_θ + E/kT)` over K offline-precomputed perturbations |
| `L_anchor` | `cfm_mol/bgfm_density.py:energy_anchor_loss` | `(log p_θ + E/kT + a_φ(c))²` to stabilize the per-molecule offset |

Total: `L = L_FM + λ₁·L_force + λ₂·L_energy + λ₃·L_anchor`
with warm-up + ramp schedule on `λ₁, λ₂, λ₃`.

---

## Current results (verified 2026-06-29)

| Variant | Step | R² (Level 1, all val) | Note |
|---|---|---|---|
| FM baseline | 50k | 0.091 | flow-matching only |
| BGFM v7c (force-only + small energy) | 50k | **0.278** | 3.0× over FM baseline |
| BGFM v8a (full BGFM @ 300 K) | step ≥ 95k (NaN at ~98k) | **[pending re-eval]** | bug-fixed eval running |
| BGFM v8b (T-conditional) | step 95k | **[pending re-eval]** | best variant pre-NaN |
| BGFM v8c (energy + anchor only) | step 55k | **[pending re-eval]** | ablation: no force loss |

**Note on R² numbers reported before 2026-06-29:** prior Stage-1 eval
hardcoded `charge = 0` and `spin = 1` for every record, which is wrong
for ~44% of OMol25 val molecules. The fix landed 2026-06-29
(`scripts/eval_boltzmann_stage1.py`); all reported numbers above will be
updated as the re-evaluation completes. See
`notes/EXPERT_AUDIT_2026-06-29.md` for the full audit.

---

## Open issues and follow-ups (tracked)

1. **v8 NaN at step 30k–98k.** `L_anchor` overflow at bf16 limit on outlier
   perturbations. Mitigation: soft-clamp anchor + NaN-skip on anchor
   gradient; `λ₃` step-down. v9 implementation pending.
2. **Spin handling in eval.** `scripts/eval_boltzmann_stage1.py` still
   hard-codes `spin = 1`. OMol25 corrects this internally most of the
   time, but rare odd-electron radicals remain a known follow-up.
3. **Force loss is a late-time approximation.** `F(x_1)` is used as a
   surrogate for the true `F(x_t)`; we have not yet ablated this against
   an online `F(x_t)` evaluation.
4. **No QM9 sanity check yet.** A standard-benchmark validity number is
   planned for camera-ready.

---

## Citing

The paper is under double-blind review.

```bibtex
@misc{anonymous2026bgfm,
  title  = {Boltzmann-Regularized Flow Matching for
            Universal-Neural-Potential Guided 3D Molecular Generation},
  author = {Anonymous},
  year   = {2026},
  note   = {Under double-blind review at ICLR 2026.
            Code: https://github.com/yurika1030sakura/bgfm}
}
```

---

## License

MIT for code in this repository.
Forked baselines (FlowMol3) retain their original licenses.

---

## Acknowledgement

We thank the external expert reviewer (2026-06-29) for a line-by-line
code-and-paper audit that surfaced three real bugs (eval charge hardcode,
T-conditional kT not reaching FFJORD, eval not supporting T-conditional
checkpoints) and corrected several over-claims in the paper text. The
expert's audit and rewrite are preserved in
`notes/EXPERT_AUDIT_2026-06-29.md`. This README and the paper are
written against that audit.
