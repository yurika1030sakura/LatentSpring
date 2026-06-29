# BGFM 实验系统重构方案（可直接转成论文 §4）

## 核心原则

BGFM 的主实验不要把 OMol25-Boltzmann correlation 当成唯一主表。这个指标和训练目标太近，容易被审稿人质疑“把 evaluation 写进 loss”。主结果应该由三类互补证据组成：

1. **标准 3D 分子生成 benchmark**：QM9 和 GEOM-Drugs，用 EBMol/EDM/GeoLDM/SLDM/FlowMol3 等相同协议比较。
2. **独立物理 oracle**：GFN2-xTB、MMFF、DFT subset，不只用 OMol25，证明 BGFM 真的生成更接近低能、少 relaxation 的结构。
3. **机制与反事实实验**：OMol25 Level-1 Boltzmann alignment 只作为 mechanism diagnostic，再配 shuffled energy/force、wrong-kT、held-out perturbation family 等 controls。

---

## 推荐文章中的实验问题（paper §4 opening）

The experiments are designed to answer five questions.

**Q1. Standard generation quality.** Does adding BGFM physical regularization preserve or improve the standard validity, stability, uniqueness, novelty, and diversity metrics used for unconditional 3D molecular generation?

**Q2. Independent physical quality.** Does BGFM generate structures that require less downstream relaxation under physical evaluators not used as the direct training target, such as GFN2-xTB, MMFF, and a small DFT single-point/relaxation subset?

**Q3. Mechanistic Boltzmann alignment.** Does BGFM actually align the model’s conditional coordinate density with local Boltzmann relative probabilities, rather than merely improving local valence or geometry heuristics?

**Q4. Validity of the evaluation.** Are improvements caused by correct OMol25 energy/force information rather than by adding arbitrary regularization or by evaluating on the same signal used for training?

**Q5. Comparison to EBMol.** Compared with an energy-based sampler such as EBMol, does BGFM provide a better quality-diversity-compute frontier by retaining an amortized flow generator while injecting external neural-potential supervision?

---

## Benchmark 模型选择

### Main baselines

| 模型 | 为什么必须放 | 数据集 |
|---|---|---|
| Data / training reference | 作为 upper/lower sanity reference | QM9, GEOM-Drugs, OMol25 slices |
| EDM | 经典 E(3)-equivariant diffusion baseline | QM9, GEOM-Drugs |
| GeoLDM | latent 3D molecule diffusion baseline | QM9, GEOM-Drugs |
| GeoBFN / BFN model | 高质量 3D 生成 baseline，尤其 QM9 | QM9 |
| SLDM / SemlaFlow if available | EBMol paper 中强 baseline/revised protocol baseline | QM9, GEOM-Drugs |
| FlowMol3 or FlowMol-CTMC | BGFM 的最直接 backbone baseline | QM9, GEOM-Drugs, OMol25 |
| EBMol | 直接竞争者，energy-based 3D molecule generation | QM9, GEOM-Drugs |
| FM baseline in our repo | 同架构无 BGFM，证明收益来自 BGFM loss | OMol25, and optionally QM9/GEOM retrain |
| BGFM-force | force-score only ablation | all BGFM-supported |
| BGFM-energy | density-energy only ablation | all BGFM-supported |
| BGFM-full | main model | all BGFM-supported |
| BGFM-T | temperature-conditional variant | OMol25/Boltzmann diagnostics |

### Optional stronger model if you decide to rebuild architecture

**BGFM++ = BGFM flow proposal + calibrated energy head + short energy refinement.**

This version is designed to beat EBMol more directly. EBMol replaces the generator with a learned EBM and then uses Langevin/parallel tempering. BGFM++ keeps fast amortized flow sampling, adds a calibrated scalar energy head distilled from OMol25/xTB/DFT, and uses only short Langevin/MALA refinement.

---

## Main Experiment 1: Standard 3D molecular generation

### Goal
Show BGFM is not sacrificing ordinary generative quality.

### Dataset/protocol

- QM9: 10k generated samples, standard EDM/EBMol protocol.
- GEOM-Drugs: 10k generated samples, EBMol revised protocol with OpenBabel bond inference, formal charge/aromaticity, RDKit sanitization, and Vendi diversity.
- For every model, report sample count, NFE, wall-clock, parameters, post-processing, and failed generation count.

### Metrics

QM9:

- Atom stability ↑
- Molecule stability ↑
- Validity ↑
- Uniqueness ↑
- Valid & unique ↑
- Novelty ↑

GEOM-Drugs:

- Atom stability ↑
- Molecule stability ↑
- Validity ↑
- Uniqueness ↑
- Novelty ↑
- Vendi diversity ↑

### Paper table template

| Model | NFE | Params | Atom stab ↑ | Mol stab ↑ | Valid ↑ | Unique ↑ | Novelty ↑ | Diversity/Vendi ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EDM | 1000 |  |  |  |  |  |  |  |
| GeoLDM | 1000 |  |  |  |  |  |  |  |
| SLDM | 1000 |  |  |  |  |  |  |  |
| FlowMol3 | matched |  |  |  |  |  |  |  |
| EBMol-low | 930/1080 |  |  |  |  |  |  |  |
| EBMol-high | 1810/7240 |  |  |  |  |  |  |  |
| BGFM-force | matched |  |  |  |  |  |  |  |
| BGFM-energy | matched |  |  |  |  |  |  |  |
| BGFM-full | matched |  |  |  |  |  |  |  |
| BGFM++ | matched |  |  |  |  |  |  |  |

### What to claim if results work

BGFM should not claim only absolute top validity. The stronger claim is Pareto:

> BGFM maintains competitive validity and stability while preserving higher uniqueness/diversity than high-compute EBMol, which becomes increasingly mode-concentrated as inference compute increases.

---

## Main Experiment 2: Independent physical quality

### Goal
Directly answer the criticism: “You trained on OMol25 energy/force, so OMol25 evaluation is biased.”

### Protocol

Generate 10k samples per model. Keep both:

1. **Valid-connected-only protocol**, matching EBMol Table 3.
2. **All-samples protocol**, where invalid/disconnected/failed samples are counted as failures or assigned worst-bin statistics.

Run independent physical evaluators:

- GFN2-xTB relaxation: primary.
- MMFF94 relaxation: cheap secondary for organic subsets.
- DFT single-point or short relaxation on 200–500 molecules: expensive but strongest reviewer-proof subset.

### Metrics

- Median relaxation energy ΔE ↓
- Mean relaxation energy ΔE ↓
- 90th percentile ΔE ↓
- Initial force norm / max force ↓
- Relaxation steps to convergence ↓
- Relaxation failure rate ↓
- RMSD after relaxation ↓
- Bond length difference after xTB relaxation ↓
- Bond angle difference ↓
- Torsion difference ↓

### Paper table template

| Model | Valid-connected ↑ | xTB median ΔE ↓ | xTB mean ΔE ↓ | p90 ΔE ↓ | max force ↓ | relax steps ↓ | failure ↓ | Vendi ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EBMol1960 |  |  |  |  |  |  |  |  |
| EBMol7240 |  |  |  |  |  |  |  |  |
| FlowMol3 |  |  |  |  |  |  |  |  |
| BGFM-force |  |  |  |  |  |  |  |  |
| BGFM-energy |  |  |  |  |  |  |  |  |
| BGFM-full |  |  |  |  |  |  |  |  |
| BGFM++ |  |  |  |  |  |  |  |  |

### Main expected claim

> BGFM improves physical quality under xTB/MMFF/DFT evaluators that are not identical to the OMol25 labels used during training, showing that BGFM learns transferable physical structure rather than merely optimizing a self-referential OMol25 metric.

---

## Main Experiment 3: Quality-diversity-compute Pareto vs EBMol

### Goal
Win against EBMol where it is weak: high-quality samples require many NFEs and diversity drops as compute increases.

### Protocol

For each model, sample at multiple budgets.

EBMol budgets:

- QM9: 930, 1370, 1810 NFEs.
- GEOM-Drugs: 1080, 1960, 3720, 7240 NFEs.

BGFM budgets:

- Same NFE ODE steps if possible.
- Same wall-clock budget.
- Same number of generated molecules.
- Optional BGFM++ refinement budget: 0, 20, 50, 100 energy-head Langevin/MALA steps.

### Plots

- x-axis: wall-clock seconds / 10k samples; y-axis: median xTB ΔE.
- x-axis: Vendi diversity; y-axis: molecule stability.
- x-axis: Vendi diversity; y-axis: median xTB ΔE.
- x-axis: NFE; y-axis: valid-connected-low-energy unique molecules per GPU-hour.

### New throughput metric

Define:

\[
\mathrm{VLU}_{\tau}
= \frac{\#\{\text{valid, connected, unique samples with } \Delta E_{\rm xTB}<\tau\}}{\text{GPU-hours}}.
\]

Use thresholds \(\tau \in \{2, 5, 10\}\) kcal/mol.

### Main expected claim

> EBMol improves quality by spending more MCMC compute, but this concentrates samples and reduces diversity. BGFM provides a more favorable quality-diversity-compute frontier because the flow model amortizes generation while BGFM regularization moves samples toward physically realistic coordinate basins during training.

---

## Main Experiment 4: Boltzmann consistency as mechanism, not main victory metric

### Goal
Show that the method is doing what it mathematically claims.

### Protocol

For held-out parent molecules, create perturbation clouds using perturbations not identical to training shards:

- Gaussian perturbation at unseen sigma values.
- Torsion rotation perturbation.
- Bond stretch perturbation.
- Angle bend perturbation.
- Short xTB or OMol25 Langevin perturbation.
- RDKit conformer perturbation.

For each parent molecule and perturbation family, compute:

\[
y_{m,k} = \log p_\theta^{\rm pos}(r_m^{(k)} \mid c_m)
\]

and compare with:

\[
- E_{\rm oracle}(r_m^{(k)},c_m)/kT.
\]

Use three oracle tiers:

- OMol25: direct mechanism check.
- GFN2-xTB: independent physical oracle.
- DFT subset: small but strongest validation.

### Metrics

- Per-parent centered Pearson r ↑
- Per-parent Spearman ρ ↑
- Slope of logp vs \(-E/kT\), target ≈ 1
- Calibration error of slope ↓
- Within-parent residual variance ↓
- Per-chemistry-slice result

### Important wording

This experiment should be described as:

> a mechanism diagnostic for conditional coordinate Boltzmann alignment

not as:

> the primary evidence that BGFM beats EBMol.

---

## Main Experiment 5: Negative controls and ablations

### Goal
Prove the result is not simply due to adding arbitrary force/energy loss or evaluating on a trained objective.

### Required controls

| Control | What it tests |
|---|---|
| FM baseline | same backbone, no BGFM physics |
| BGFM-force | force-score contribution |
| BGFM-energy | density-energy contribution |
| BGFM-full | synergy |
| shuffled force across atoms | force direction matters |
| shuffled force across molecules | molecule-specific force matters |
| force norm only | direction is more important than magnitude prior |
| shuffled energy within perturbation groups | correct relative energies matter |
| shuffled parent grouping | within-parent Boltzmann grouping matters |
| wrong kT during eval | slope/temperature consistency matters |
| random scalar regularizer | not just regularization benefit |

### Paper table template

| Model/control | OMol25 Boltzmann r ↑ | xTB median ΔE ↓ | DFT subset ΔE ↓ | Valid ↑ | Diversity ↑ |
|---|---:|---:|---:|---:|---:|
| FM baseline |  |  |  |  |  |
| BGFM true labels |  |  |  |  |  |
| shuffled force |  |  |  |  |  |
| shuffled energy |  |  |  |  |  |
| wrong parent grouping |  |  |  |  |  |
| wrong kT |  |  |  |  |  |

### Main expected claim

> The correct BGFM labels improve both OMol25 mechanism metrics and independent xTB/DFT physical quality, whereas shuffled-label and wrong-temperature controls fail to transfer. This rules out the trivial explanation that any force/energy-shaped regularizer would improve the proposed evaluation.

---

## Main Experiment 6: Energy/scorer ranking vs EBMol

### Goal
Beat EBMol on one of its own claimed advantages: physically meaningful scoring.

### Protocol

Collect a mixed pool of samples:

- EDM samples
- GeoLDM samples
- FlowMol3 samples
- EBMol samples
- BGFM samples
- high-strain perturbed samples
- relaxed samples

Score each molecule using:

- EBMol learned energy
- OMol25 energy
- BGFM score \(S(x)=\log p_\theta^{pos}(r\mid c)-\beta E_{\rm OMol25}(r,c)-b_\phi(c)\)
- BGFM++ calibrated energy head, if implemented

Ground truth:

- GFN2-xTB relaxation ΔE
- xTB max force
- DFT subset ΔE or force norm

Metrics:

- Pearson r ↑
- Spearman ρ ↑
- AUROC for detecting top-10% high-strain samples ↑
- Top-k filtering: median ΔE after keeping best 10%, 20%, 50% ↓

### Paper table template

| Scorer | Test sample pool | Pearson r ↑ | Spearman ρ ↑ | AUROC high-strain ↑ | Top-10% kept ΔE ↓ |
|---|---|---:|---:|---:|---:|
| EBMol energy | cross-model pool |  |  |  |  |
| OMol25 energy | cross-model pool |  |  |  |  |
| BGFM score | cross-model pool |  |  |  |  |
| BGFM++ energy head | cross-model pool |  |  |  |  |

### Main expected claim

> Because BGFM is trained with an external physical oracle, its score/ranking signal transfers better to independent xTB/DFT strain labels than EBMol’s unsupervised learned energy, while retaining a fast amortized sampler.

---

## Main Experiment 7: Broad-chemistry OMol25 slices

### Goal
Show BGFM’s unique value beyond QM9/GEOM, where EBMol is currently reported.

### Slices

- neutral vs charged
- small vs medium vs large atom count
- organic-only vs heteroatom-rich
- metal-containing / organometallic if supported
- electrolyte-like species
- high conformational flexibility
- rare elements / underrepresented elements

### Baselines

- FM baseline same architecture
- BGFM-force
- BGFM-energy
- BGFM-full
- BGFM-T
- Optional: EBMol retrained on OMol25 subset if feasible, clearly labeled as our reimplementation

### Metrics

- valid connected fraction
- xTB/OMol25/DFT subset relaxation ΔE
- max force
- Boltzmann consistency within held-out perturbations
- failure rate by slice
- diversity by slice

### Main expected claim

> BGFM’s advantage grows in broad chemistry where a universal neural potential provides physical information unavailable in standard QM9/GEOM-only generation benchmarks.

---

## 推荐论文 §4 结构

### 4.1 Experimental questions and design
State the five questions Q1–Q5.

### 4.2 Standard unconditional generation benchmarks
QM9 and GEOM-Drugs; compare with EBMol, FlowMol3, EDM, GeoLDM, SLDM, GeoBFN where applicable.

### 4.3 Independent physical quality
xTB/MMFF/DFT relaxation metrics; valid-connected and all-samples protocols.

### 4.4 Quality-diversity-compute frontier
Pareto curves vs EBMol NFE budgets.

### 4.5 Mechanistic Boltzmann consistency
Held-out perturbation families; OMol25/xTB/DFT alignment; slopes and residual variance.

### 4.6 Ablations and negative controls
force-only, energy-only, full, shuffled labels, wrong kT.

### 4.7 Broad-chemistry generalization
OMol25 slices and per-slice analysis.

---

## 最推荐主文 claim

Use this exact style:

> BGFM is not evaluated only on the OMol25 objective it is trained against. We first compare against EBMol and standard 3D molecular generators under the same QM9 and GEOM-Drugs protocols. We then evaluate physical quality using independent xTB/MMFF/DFT relaxation metrics. Finally, we use held-out Boltzmann perturbation tests and shuffled-label controls to verify the mechanism and rule out self-fulfilling evaluation.

---

## 最重要的不要写

Do not write:

- “We beat EBMol because our OMol25 Boltzmann R² is higher.”
- “BGFM samples the exact natural Boltzmann distribution.”
- “Diffusion/flow cannot learn probabilities.”
- “Log-Z predictor estimates the true partition function.”

Instead write:

- “BGFM aligns conditional coordinate densities with local Boltzmann relative probabilities induced by an external neural potential.”
- “OMol25-based metrics are mechanism diagnostics; the main physical evaluation uses independent xTB/MMFF/DFT oracles.”
- “The key comparison to EBMol is quality-diversity-compute frontier, not only a single stability number.”
