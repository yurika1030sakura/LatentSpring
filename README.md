# BGFM: Boltzmann-Calibrated Flow Matching with Universal Neural Potentials for 3D Molecular Generation

BGFM is a single trained 3D molecular generator that combines an
amortized flow proposal, an external-neural-potential Boltzmann
regularization, a calibrated scalar energy head, and a short
learned-energy Langevin corrector at inference time. It targets the
de novo generation regime — atom counts, atom types, charges, and 3D
coordinates are all sampled — and uses OMol25 ($\sim 10^8$ DFT-quality
configurations covering 83 elements) as a training-time energy
teacher.

This repository accompanies the paper [`paper/bgfm_paper.pdf`](paper/bgfm_paper.pdf).

---

## Method overview

BGFM has four modules trained jointly (Modules 1–3) or applied at
inference (Module 4):

1. **Flow proposal** — a FlowMol3-style GVP-Transformer
   parameterizing a velocity field over coordinates, atom types,
   and charges.
2. **OMol25 Boltzmann regularization** — two consistency losses
   that align the flow's implied score and its conditional density
   with the universal neural potential:
   - Force-score consistency at late flow path times.
   - Density-energy variance over precomputed per-parent perturbation
     clouds, with a composition-conditioned offset stabilizer.
3. **Calibrated scalar energy head $\hat E_\psi(r, c)$** — distilled
   from the same potential via L1 energy regression and gradient
   matching against OMol25 forces; provides an explicit calibrated
   energy landscape.
4. **Learned-energy Langevin corrector** — a short ($J = 20$–$100$
   steps) Langevin chain under $\hat E_\psi$ that refines flow
   proposals at inference time; reported with honest NFE and
   wall-clock accounting separate from the flow ODE budget.

A joint discrete-continuous density estimator
$\log p_\theta(x) = \log p_\theta(r \mid c) + \log p_\theta(c)$
combines a coordinate FFJORD integral with the categorical-path
log-probability of the discrete components, so the Boltzmann
regularization is well-defined across heterogeneous compositions.

The total training objective is

$$
\mathcal{L} = \mathcal{L}_{\rm FM} + \lambda_1 \mathcal{L}_{\rm force} + \lambda_2 \mathcal{L}_{\rm dens} + \lambda_3 \mathcal{L}_{\rm anchor} + \lambda_4 \mathcal{L}_{\rm head}.
$$

---

## Repository layout

```
bgfm/
├── paper/
│   ├── bgfm_paper.tex                    # LaTeX source
│   ├── bgfm_paper.pdf                    # compiled
│   └── iclr2026_conference.{bib,sty,bst}
│
├── cfm_mol/                              # BGFM extensions
│   ├── bgfm_density.py                   # FFJORD coordinate log-density
│   ├── bgfm_loss.py                      # force, density-energy, anchor, head loss
│   ├── bgfm_train_hook.py                # joint training step
│   ├── joint_density.py                  # discrete-continuous log p(x)
│   ├── energy_head.py                    # scalar energy head E_psi(r, c)
│   ├── refinement.py                     # Langevin corrector with NFE accounting
│   ├── log_z_predictor.py                # composition-conditioned offset stabilizer
│   ├── kt_conditioning.py                # optional temperature conditioning
│   ├── perturbation_loader.py            # offline perturbation shard loader
│   ├── physics.py                        # OMol25 wrapper
│   └── flow_model.py                     # backbone path/sampler hooks
│
├── baselines/flowmol3/                   # vendored FlowMol3 backbone (not in git)
│
├── configs/
│   ├── omol25_4m_bgfm.yaml               # flow-matching baseline
│   ├── omol25_4m_bgfm_v10_full.yaml      # full BGFM (all 4 modules)
│   └── ablations/...                     # per-module ablation configs
│
├── scripts/
│   ├── precompute_energy_perturbations.py   # offline OMol25 perturbation shards
│   ├── run_train.py                         # Lightning training entry
│   ├── launch_omol25_bgfm_h200.sh           # H200 launcher
│   ├── sample_bgfm.py                       # flow + corrector sampler
│   ├── eval_qm9_ebmol_protocol.py
│   ├── eval_geomdrugs_ebmol_protocol.py
│   ├── eval_xtb_relaxation.py
│   ├── eval_cross_model_ranking.py
│   ├── eval_negative_controls.py
│   └── eval_boltzmann_stage{1,2}.py         # mechanism diagnostic
│
├── CODE_MAP.md                           # paper section → source file map
└── README.md                             # this file
```

---

## Reproduction recipe

```bash
# 1. OMol25 data
sbatch scripts/download_omol25_4m.slurm
sbatch scripts/preprocess_omol25.slurm

# 2. Offline OMol25 perturbation shards (energy + force labels)
sbatch scripts/run_precompute_energy.slurm val   10000
sbatch scripts/run_precompute_energy.slurm train 30000

# 3. Train full BGFM
sbatch scripts/launch_omol25_bgfm_h200.sh \
       configs/omol25_4m_bgfm_v10_full.yaml "" 42

# 4. Sample with flow + Langevin corrector
python scripts/sample_bgfm.py \
       --checkpoint runs/omol25_4m_bgfm_v10_full/.../last.ckpt \
       --config configs/omol25_4m_bgfm_v10_full.yaml \
       --n_samples 10000 --flow_nfe 100 --corrector_steps 50 \
       --out runs/eval/geomdrugs/samples.json

# 5. Independent physical evaluation (GFN2-xTB)
python scripts/eval_xtb_relaxation.py \
       --samples runs/eval/geomdrugs/samples.json \
       --out_csv runs/eval/geomdrugs/xtb_relax.csv

# 6. EBMol-protocol QM9 / GEOM-Drugs benchmark
python scripts/eval_qm9_ebmol_protocol.py        --samples ... --out ...
python scripts/eval_geomdrugs_ebmol_protocol.py  --samples ... --out ...

# 7. Cross-model energy ranking
python scripts/eval_cross_model_ranking.py \
       --samples_dir runs/eval/cross_model/ \
       --out_csv runs/eval/cross_model/ranking.csv

# 8. Negative controls (shuffled-label retraining)
python scripts/eval_negative_controls.py \
       --in_shard  /path/perturbation_train_n30000_s0.pt \
       --out_shard /path/perturbation_train_shuffled.pt \
       --shuffle_energy_within_parent --seed 0
sbatch scripts/launch_omol25_bgfm_h200.sh \
       configs/ablations/bgfm_shuffled_energy.yaml "" 42
```

---

## Loss terms (quick reference)

| Term | Code | Purpose |
|---|---|---|
| $\mathcal L_{\rm FM}$ | FlowMol3 backbone | Standard flow-matching reconstruction over coordinates, atom types, charges. |
| $\mathcal L_{\rm force}$ | `cfm_mol/bgfm_loss.py:force_loss` | Cosine between FM-implied score and $F_{\rm OMol25}(r_1)/kT$ at late path times. |
| $\mathcal L_{\rm dens}$ | `cfm_mol/bgfm_density.py:energy_consistency_loss_per_mol` | $\mathrm{Var}_k(\log p_\theta + E/kT)$ over $K$ precomputed perturbations. |
| $\mathcal L_{\rm anchor}$ | `cfm_mol/bgfm_density.py:energy_anchor_loss` | $(\log p_\theta + E/kT + a_\phi(c))^2$ stabilizing the per-molecule offset. |
| $\mathcal L_{\rm head}$ | `cfm_mol/bgfm_loss.py:energy_head_calibration_loss` | L1 energy regression + force-gradient matching of $\hat E_\psi$ to OMol25. |

---

## Evaluation framework

| Experiment | Question | Evaluator |
|---|---|---|
| 1. Standard generation | Does BGFM preserve standard validity / diversity? | QM9, GEOM-Drugs (EBMol protocol + Vendi diversity) |
| 2. Independent physical quality | Does BGFM improve under evaluators NOT in training? | **GFN2-xTB + MMFF + DFT subset** |
| 3. Quality–diversity–compute frontier | Does BGFM dominate the Pareto vs EBMol? | $\mathrm{VULS}_\tau$ throughput, NFE, wall-clock |
| 4. Mechanism diagnostic | Did BGFM implement its target objective? | OMol25 log-density vs. $-E/kT$ on held-out perturbation families |
| 5. Ablations | What does each module contribute? | per-module removal |
| 6. Negative controls | Is the lift causal? | shuffled-energy / shuffled-force / wrong-$kT$ retraining |
| 7. Cross-model ranking | Does the BGFM scorer beat EBMol's r=0.65? | pooled EDM / FlowMol3 / EBMol / BGFM scored vs xTB $\Delta E$ |

---

## Citing

The paper is under double-blind review.

```bibtex
@misc{anonymous2026bgfm,
  title  = {BGFM: Boltzmann-Calibrated Flow Matching with
            Universal Neural Potentials for 3D Molecular Generation},
  author = {Anonymous},
  year   = {2026},
  note   = {Under double-blind review. Code: https://github.com/yurika1030sakura/bgfm}
}
```

---

## License

MIT for code in this repository.
Forked baselines (FlowMol3) retain their original licenses.
