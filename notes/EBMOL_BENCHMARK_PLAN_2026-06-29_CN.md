# BGFM vs EBMol — 公平 benchmark / 反质疑实验 / 击败路线图

> Source: external expert review, 2026-06-29.
> Saved into repo for reviewer transparency.

## 0. 核心判断

EBMol 是强竞争者，但不是直接 kill BGFM。EBMol 的核心是训练一个 time-unconditional EBM,
并用 Mirror-Langevin + parallel tempering 从 learned energy landscape 采样。
BGFM 应该把差异说为：保留 FlowMol3-style de novo generator，用外部 OMol25 neural
potential 的能量和力，在训练时把 generator 的 conditional coordinate density 推向
local Boltzmann relative probabilities。

要赢，不能只靠 "OMol25 logp-energy correlation"。这项指标和训练 objective 太近，
会被说成自证循环。应该把它降级为机制验证，把主胜利点转到：

1. EBMol 自己的 QM9 / GEOM-Drugs validity/stability/diversity metrics；
2. 独立物理 oracle 的 relaxation-energy / geometry metrics（GFN2-xTB, MMFF, DFT subset）；
3. 质量-多样性 Pareto frontier；
4. compute-normalized sampling efficiency；
5. external-oracle ranking / filtering / steering / linker design 能力。

---

## 1. EBMol 的主要 evaluation metrics

### 1.1 QM9 standard generation metrics
10k samples: atom stability, molecule stability, validity, uniqueness, valid&unique,
novelty, NFE-matched comparison. EBMol reported settings: 930, 1370, 1810 NFEs.

### 1.2 GEOM-Drugs revised validity metrics
10k samples + OpenBabel bond inference + RDKit sanitization + revised valency tables.
EBMol reported settings: 1080, 1960, 3720, 7240 NFEs.

### 1.3 GEOM-Drugs structural / physical metrics
On valid and connected molecules only:
- bond length / angle / torsion distribution differences
- mean / median GFN2-xTB relaxation energy

This is the most important table to beat physically.

### 1.4 Energy meaningfulness
EBMol correlates its learned energy with GFN2-xTB relaxation energy on EDM-generated
molecules. They report Pearson r = 0.65, p < 0.0001. BGFM should beat this with a
cross-oracle scoring function.

### 1.5 Controllable generation
EBMol claims shape steering and zero-shot linker design through energy composition.

---

## 2. Why the current BGFM evaluation can be criticized

The current BGFM Level-1 metric (log p_θ vs OMol25 energy on perturbation clouds) is
scientifically meaningful but too close to the training objective. A reviewer can say:

> You trained with OMol25 force/energy; then you evaluate with OMol25 force/energy.
> Of course the metric improves.

Correct response:
- Keep Level 1 as mechanism verification, NOT the main result.
- Add held-out molecules, held-out perturbation seeds, held-out perturbation types,
  and independent physical oracles.
- Report negative controls: shuffled energy, shuffled force, wrong kT, wrong parent
  grouping.

---

## 3. Fair evaluation design

### 3.1 Direct EBMol benchmark track

Train and evaluate BGFM on the exact same benchmarks as EBMol.

#### QM9
- 10k generated samples
- Same train/test split, atom-count distribution, NFE budget
- Same OpenBabel/RDKit postprocessing
- No post-relax unless every method receives it

Variants: FM baseline, BGFM-force, BGFM-energy, BGFM-full, BGFM-full+guidance, BGFM++.

#### GEOM-Drugs
- 10k generated samples
- EBMol/GEOM-Drugs Revisited metrics
- Compute budgets: 1080, 1960, 3720, 7240 NFEs
- Wall-clock and GPU-normalized sampling time

### 3.2 Independent physical-oracle track

Train with OMol25; evaluate with:
- GFN2-xTB relaxation energy
- MMFF relaxation where valid
- DFT single-points / short relaxation on 200–500 samples if affordable
- Optional MACE / OrbNet as secondary oracle

Metrics:
- median ΔE after relaxation
- mean ΔE after relaxation
- max force before relaxation
- RMSD after relaxation
- number of optimizer steps to converge
- relaxation failure rate
- bond length / angle / torsion distribution shift

This is the track that defeats the "you trained on this metric" criticism.

### 3.3 Distributional chemistry track

EBMol trades quality for diversity (Diversity 1096 → 645 from EBMol1080 to EBMol7240
vs data 901). BGFM should report Pareto curves:
- molecule stability vs uniqueness
- median ΔE vs Vendi diversity
- validity vs novelty
- valid-novel-low-energy rate
- energy quantile vs scaffold diversity

Composite metrics:
- VNL: valid + novel + low-energy fraction
- VUL: valid + unique + low-energy fraction
- Pareto AUC over guidance strength / temperature

### 3.4 Mechanism-validity track

Negative controls (essential):
- true OMol25 labels vs shuffled energy labels (within batch / across parents)
- true forces vs shuffled forces (within batch / across atoms)
- true parent grouping vs random parent grouping
- correct kT vs wrong kT
- held-out perturbation distribution: Gaussian, torsion rotation, bond stretch,
  angle bend, RDKit conformer noise, xTB-relaxed conformer noise
- held-out chemistry slice: charged, heteroatom-rich, metal-containing, larger N

The "real" story:
- True BGFM improves on independent xTB / DFT.
- Shuffled-label BGFM may have OMol25 Level-1 lift, but no xTB / DFT lift.
- This separates real Boltzmann alignment from training-metric memorization.

---

## 4. How to beat EBMol in practice

### Route A: Beat on diversity–quality Pareto
EBMol's high-NFE sampler improves stability and relaxation energy but reduces
uniqueness/diversity. BGFM exploits flow's diversity.

Target claim:
> At matched stability or matched relaxation energy, BGFM preserves higher uniqueness,
> novelty, and Vendi diversity than EBMol.

### Route B: Beat on independent relaxation energy
EBMol's strongest number is median GFN2-xTB relaxation energy.
- BGFM-full should target EBMol1960 first.
- BGFM++ should target EBMol3720 / 7240.

### Route C: Beat on sampling efficiency
EBMol needs many NFEs and PT chains. BGFM can win if similar quality at fewer steps.

Metrics:
- valid-connected-low-energy samples per second
- median ΔE at 500 / 1000 / 2000 NFEs
- sample throughput on same GPU
- oracle calls per accepted molecule

### Route D: Beat on calibrated external energy scoring
EBMol claims learned energy is useful for ranking. BGFM provides:

  S(x) = log p_θ^pos(r|c) − β·E_OMol25(r,c) − b_φ(c)

or a learned fast surrogate distilled from OMol25.

Evaluate Pearson r with GFN2-xTB relaxation energy on samples from EDM, FlowMol,
EBMol, AND BGFM. Beat EBMol's r = 0.65.

### Route E: Beat on controllability
- shape-guided flow sampling: add gradient of differentiable shape potential late
  in the trajectory
- fragment inpainting: freeze fragment atoms, sample missing atoms with mask,
  optionally OMol25-guided refinement
- scaffold-constrained molecule completion

Evaluate same task definitions as EBMol where possible.

---

## 5. Architecture changes if you want BGFM++

### 5.1 Flow proposal + energy refinement
Keep FlowMol3/BGFM as fast amortized proposal, then apply learned-energy or
OMol25-surrogate Langevin/MALA refinement for 20–100 steps.

  noise → BGFM flow (100–500 NFE) → short MALA refinement (20–100 steps)
        → optional xTB/OMol25 light filter

Honest NFE accounting: include both flow steps and energy-head steps in NFE budget.

### 5.2 Add persistent scalar energy head
Add E_ψ(x) to BGFM's vector field backbone:
- train E_ψ with RFM-style local-minima loss
- calibrate E_ψ with OMol25 / xTB energy labels
- use E_ψ for ranking, filtering, steering, and fast Langevin refinement

This directly neutralizes EBMol's strongest claim (explicit energy landscape).

### 5.3 Multi-oracle teacher training
- OMol25 force-score loss
- xTB relaxation-energy ranking loss
- DFT single-point calibration on small subset
- shuffled-control losses for causality

### 5.4 Full joint density extension
Current BGFM density is position-only. For a stronger theoretical paper:
- coordinate CNF log-density
- discrete atom-type / charge log-probability from CTMC path
- atom-count probability
- optional chemical-potential terms for cross-composition comparison

---

## 6. Minimal experiments required for a strong paper

Main tables (in priority order):

1. **QM9 standard metrics vs EBMol** (Section 4.1).
2. **GEOM-Drugs revised validity metrics vs EBMol** (Section 4.2).
3. **GEOM-Drugs physical metrics vs EBMol using GFN2-xTB** (Section 4.3).
4. **Pareto AUC: stability/energy vs diversity** (Section 4.4).
5. **Energy-ranking correlation against GFN2-xTB / DFT** (Section 4.5).
6. **Ablations**: FM, force-only, energy-only, full, full+guidance.
7. **Negative controls**: shuffled energy, shuffled force, wrong kT.
8. **Held-out perturbation types and chemistry slices**.

Do NOT use OMol25 logp-energy correlation as the main victory table. Demote it to
mechanism-verification status.

---

## 7. Clean paper claim

Best:
> BGFM is not merely an energy-scored generator. It is an external-oracle-calibrated
> flow generator whose samples match EBMol-level physical quality while preserving the
> diversity and amortized sampling speed of flow matching.

If results support it:
> Compared with EBMol, BGFM achieves a better quality-diversity-compute Pareto
> frontier: lower independent relaxation energy at matched diversity, or higher
> diversity at matched physical quality, while using fewer sampling steps.
