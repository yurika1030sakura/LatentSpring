# BGFM ICLR/Oral Readiness Audit（代码 + 论文 + 方法 + 实验）

> 结论先行：这个课题有明确的 ICLR 潜力，最强定位应该是 **external neural-potential calibrated amortized 3D molecular flow generator**。但当前 zip 还不能直接按 paper 的强 claim 投：最大问题不是 idea，而是 **paper claim > code wiring > experiment implementation** 三者不一致。要冲 ICLR，尤其 oral，必须先把下面的 P0/P1 问题修掉，并把主结果从 OMol25 自证指标切换到 independent physical metrics + quality-diversity-compute Pareto。

---

## 0. 最终推荐定位

论文里不要把方法拆成 BGFM 和 BGFM++ 两个工作。第一篇文章应只叫 **BGFM**，但它内部就是你想要的 BGFM++：

1. FlowMol3-style amortized flow proposal；
2. OMol25 force-score + density-energy Boltzmann regularization；
3. calibrated scalar energy head；
4. short learned-energy Langevin corrector。

旧版“只加 force/density loss 的 BGFM”应放在 ablation：`BGFM w/o head`、`BGFM w/o corrector`、`BGFM-force`、`BGFM-energy`。

推荐标题：

> **BGFM: Boltzmann-Calibrated Flow Matching with Universal Neural Potentials for 3D Molecular Generation**

推荐一句话 novelty：

> BGFM keeps a fast de novo 3D flow generator, but calibrates its coordinate density and a learned energy corrector against an external universal neural potential, yielding low-strain diverse molecules at a better quality-diversity-compute frontier than energy-based samplers.

---

## 1. Novelty 审核

### 1.1 能 claim 什么

可以 claim：

- 不是发明新 diffusion/flow backbone，而是把 external universal neural potential 的能量/力变成 de novo 3D flow generator 的训练和 refinement 信号。
- 和 EBMol 区别清楚：EBMol 自己学 scalar energy，再用 Langevin/parallel tempering 慢采样；BGFM 保留 amortized flow proposal，用 OMol25 校准 energy head，再做短步 corrector。
- 和 FlexiFlow 区别清楚：FlexiFlow 强在 joint molecule + conformer ensemble；BGFM 强在 external NP calibrated physical quality 和 low-strain generation。
- 和 iEFM 区别清楚：iEFM 是 fixed continuous unnormalized density sampling；BGFM 是 mixed discrete-continuous de novo molecules + external NP teacher。

### 1.2 不能 claim 什么

不要 claim：

- “first physically consistent 3D molecule generator”；EBMol 已经直接占这个大表述。
- “generated distribution equals natural Boltzmann distribution”；代码和物理定义都不支持。
- “full molecular Boltzmann distribution over all atom counts/types/charges”；跨 composition 需要 chemical potentials / reservoirs，不是简单 `exp(-E/kT)`。
- “joint density regularizer is fully used in training”；当前代码里 joint density 模块存在，但没有被主 training hook 使用。
- “corrector is exact Boltzmann sampler”；当前 corrector 是 short learned-energy Langevin-like refinement，默认没有 Metropolis 校正。

---

## 2. Paper 审核

### P0：Abstract 过度 claim

当前 abstract 说：

- joint density estimator makes objective well-defined over full molecular configuration；
- on QM9 and GEOM-Drugs, BGFM already matches/improves baselines；
- independent xTB/MMFF 已经 lower-strain；
- improves Pareto frontier。

问题：paper tables 全是 placeholders，相关 eval scripts 也还是 NotImplemented。不能这样写。

修改：我已给出重写版 `bgfm_iclr_oral_rewrite.tex`，把这些改成实验 protocol 和待填结果，不编造数字。

### P0：Joint density 表述过强

当前 paper §3.7 把 joint density 写成 full molecular likelihood，并说 lifting from coordinate-conditional to full molecular alignment。代码上：

- `cfm_mol/joint_density.py` 存在；
- `cfm_mol/bgfm_density.py` 支持 `discrete_log_p=None`；
- 但 `bgfm_train_hook.py` 没有构造或传入 `discrete_log_p`；
- `joint_density_enabled: true` 在 config 里目前没有实质作用。

修改：主方法应写成 **conditional coordinate Boltzmann alignment**。Joint density 放成 optional extension / diagnostic，除非你真把它 wired 到 training loss。

### P0：Energy head / corrector 是核心，但代码目前还没完全闭合

Paper 把 calibrated energy head 和 corrector 放成主贡献，这是正确方向，但代码里有关键 bug：

- energy head target shape 可能不对，导致 `L_head_runtime_skip` 静默跳过；
- sampler 没有正确 attach `_bgfm_energy_head`；
- sampler 调用 FlowMol3 API 不兼容；
- corrector 的 `energy_and_force` 调用参数错误；
- `metropolis=True` 多图情况未实现。

我已给出 critical patch 修复核心路径。

### P1：Related Work 要更精准

必须加入：

- EBMol：direct competitor；
- FlexiFlow：standard generation / conformer ensemble competitor；
- iEFM：theoretical related work, not benchmark；
- FlowMol3：direct backbone baseline；
- OMol25：external universal NP teacher。

不要再用 `anonymous2025ebmol`、`flexiflow2025` 等占位 bib 作者；至少在 internal draft 里把作者和 arXiv ID 改准确。

---

## 3. 公式审核

### 3.1 Conditional Boltzmann target：正确

应写：

\[
 p_B(r\mid c,T)=Z_c(T)^{-1}\exp[-E_{\rm NP}(r,c)/kT].
\]

其中 `c` 固定 atom types / charges / molecular identity，`r` 是 coordinates。

### 3.2 Force-score loss：可用，但要叫 late-time approximation

理想条件：

\[
 \nabla_r \log p_B(r\mid c,T)=F_{\rm NP}(r,c)/kT.
\]

代码当前实际做的是 late-time `s_theta(x_t,t)` vs endpoint `F(x_1)/kT`。这不是 exact score matching at arbitrary `x_t`，应该写成：

> late-time force-score consistency regularizer.

### 3.3 Density-energy variance：这是最强 originality

正确公式：

\[
\mathcal L_{\rm dens}=\mathbb E_m\operatorname{Var}_{k}
\left[\log p^{\rm pos}_\theta(r_m^{(k)}\mid c_m)+\beta E_{\rm NP}(r_m^{(k)},c_m)\right].
\]

这个是最像 ICLR 的核心方法点。重点是 within-parent relative probability，不是跨分子的 global likelihood。

### 3.4 Anchor / log-Z：不要叫 true partition function

当前 `LogZPredictor` 更准确叫 composition-conditioned offset head。不能说估计真实 thermodynamic partition function。

### 3.5 Energy head：是 calibrated strain scorer，不是 free energy

Energy head 可以很高级，但必须诚实：它蒸馏的是 OMol25 single-point energy/force，不是严格 free energy surface。可以用于 ranking/refinement，但不是 thermodynamic proof。

### 3.6 Corrector：不是 exact sampler

当前 corrector 是 short unadjusted Langevin-like coordinate refinement；如果不开 Metropolis，不应叫 exact Boltzmann correction。应该报 energy-head NFEs、wall-clock、accept/reject（如果用 Metropolis）。

---

## 4. 代码审核：P0 / P1 / P2

### P0-1：Energy head loss 可能被静默跳过

文件：`cfm_mol/bgfm_train_hook.py`

问题：`E_pred` 是 per-graph `(B,)`，但 `E_target = g.ndata.get('energy_1_true')` 可能是 per-node `(N,)`。shape mismatch 会进入 `except RuntimeError`，只记录 `train_L_head_runtime_skip`，导致主方法的 energy head 不训练。

已修：patch 将 per-node energy 聚合为 per-graph scalar，并给 `lambda_4` 加 warmup/ramp。

### P0-2：Sampler / corrector 当前不可作为主实验入口

文件：`scripts/sample_bgfm.py`

问题：

- `model.sample(n_samples=..., n_steps=...)` 不符合 FlowMol3 API；
- checkpoint load 前没有 instantiate energy head；
- training attach 的是 `_bgfm_energy_head`，script 检查的是 `model.energy_head`；
- `energy_and_force` 调用参数错误；
- `model.scalar_features_at` 不存在；
- SampledMolecule 不是 dict，不能 `sample["positions"]`。

已修：patch 重写 sampler，使用 `sample_random_sizes`、正确 attach head、正确调用 `energy_and_force`、逐样本 corrector、JSON 输出和 accounting。

### P0-3：Independent eval scripts 还是模板

这些脚本仍然 `NotImplementedError`：

- `scripts/eval_qm9_ebmol_protocol.py`
- `scripts/eval_geomdrugs_ebmol_protocol.py`
- `scripts/eval_xtb_relaxation.py`
- `scripts/eval_cross_model_ranking.py`

这意味着 paper 的主实验目前不能复现。ICLR submission 前必须完成。

### P0-4：Joint density 未 wired

文件：`cfm_mol/joint_density.py` 模块存在，但主 training loss 没有使用。要么：

- 真正把 `discrete_log_p` 传入 `energy_consistency_loss_per_mol`；
- 要么 paper 改成 optional extension / diagnostic。

我建议先 paper 降 claim；之后再把 joint density 做成 ablation。

### P1-1：Config 注释不一致

文件：`configs/omol25_4m_bgfm_v10_full.yaml`

注释说 `kT=1 eV` / `lambda_2 disabled`，实际是 `kT: 0.025` 且 `lambda_2>0`。已在 patch 中改注释。

### P1-2：Energy loader device 硬编码

`PerturbationLoader(... device="cuda")` 不适合 CPU smoke / DDP / 非 cuda:0。已修成 `next(model.parameters()).device`。

### P1-3：Corrector 没有 recenter

short Langevin step 会引入整体平移噪声。patch 加了 per-sample centroid recenter。

### P1-4：Metropolis 多图分支未实现

patch 改成显式 `NotImplementedError`，避免 silent pass。

### P2：EnergyHead O(N²) 成本

`energy_head.py` 当前 pairwise MLP 是 O(N²)。主文可以先限制 corrector 到 `N<=200` 或 GEOM typical sizes，但最终最好加 radius graph/chunked pair evaluation，否则 broad chemistry 大分子会出问题。

---

## 5. 实验设计：主文必须回答 6 个问题

### Q1. Standard generation quality

证明 BGFM 没破坏 validity/stability/diversity。

Models：EDM、GeoLDM/GeoBFN/SemlaFlow、FlowMol3/FM backbone、FlexiFlow、EBMol、BGFM ablations、BGFM full。

Metrics：atom stability、molecule stability、validity、uniqueness、novelty、Vendi diversity、NFE、wall-clock。

### Q2. Independent physical quality

主胜利表，不用 OMol25 自证。

Metrics：GFN2-xTB relaxation energy、MMFF relaxation、DFT subset、max force、RMSD after relaxation、relax steps、failure rate。

报告两个 protocol：

- valid-connected-only（和 EBMol 对齐）；
- all-samples（invalid/disconnected/relax failed 计入 failure，更严格）。

### Q3. Quality-diversity-compute Pareto

这是打 EBMol 的核心。

Plot：x-axis = NFE / wall-clock / GPU-hour；y-axis = median xTB ΔE；color/size = Vendi diversity / valid-low-strain count。

定义：

\[
\mathrm{VULS}_\tau=\frac{\#\{\text{valid, unique, low-strain samples with }\Delta E_{xTB}<\tau\}}{\text{GPU-hour}}.
\]

### Q4. Mechanism diagnostic

OMol25 Boltzmann correlation 只作为 mechanism sanity check，不作为主胜利指标。

Perturbation families：Gaussian、torsion、bond stretch、angle bend、RDKit conformer、short Langevin perturbation。

Metrics：within-parent Pearson/Spearman、slope、calibration error、residual variance。

### Q5. Negative controls

必须包含：shuffled force、shuffled energy、wrong parent grouping、wrong kT、force norm only、random scalar regularizer。

目标：true BGFM 在 independent xTB/DFT 上提升；shuffled controls 不提升。

### Q6. Cross-model ranking / filtering

正面打 EBMol 的 learned-energy scorer claim。

Sample pool：EDM/FlowMol3/EBMol/BGFM/strained perturbations/relaxed structures。

Scorers：EBMol energy、OMol25 energy、BGFM score、BGFM energy head。

Targets：xTB ΔE、DFT subset ΔE、force norm。

Goal：BGFM scorer 在 independent target 上超过或匹配 EBMol 的 reported ranking ability。

---

## 6. Benchmark 选择

### 必做 benchmark

- EBMol：direct competitor。
- FlexiFlow：standard generation / conformer ensemble competitor。
- FlowMol3/FM backbone：same-backbone control。
- EDM / GeoLDM / GeoBFN / SemlaFlow：standard 3D baselines。

### 不做 main benchmark，但必须引用

- iEFM：fixed unnormalized-density CNF sampler；不是 de novo molecule benchmark。
- Boltzmann Generators / Equivariant FM：fixed-system/conformer sampling。

---

## 7. Submission roadmap

### Week 1：修 P0 code

1. 应用 `bgfm_iclr_oral_critical_fixes.patch`。
2. 跑 one-batch training，确认 `train_L_head` 不再一直 runtime skip。
3. 跑 `sample_bgfm.py --corrector_steps 0/20`，确认能导出 samples JSON。
4. 实现 xTB eval script。

### Week 2：EBMol protocol 对齐

1. 跑 EBMol 官方 generated samples / checkpoint。
2. 用同一 eval 脚本评估 BGFM 与 EBMol。
3. 生成 QM9/GEOM standard table。

### Week 3：Independent physics + Pareto

1. xTB/MMFF/DFT subset。
2. flow-only vs corrector steps sweep：0/20/50/100。
3. Pareto plots。

### Week 4：negative controls + ablations

1. force/density/head/corrector ablation。
2. shuffled controls。
3. wrong kT。
4. cross-model ranking。

### Week 5：paper hardening

1. 填真实数字。
2. 删所有 placeholder。
3. 加 failure analysis 和 limitations。
4. CodeMap 更新 commit hash。
5. 复现脚本一键运行。

---

## 8. 口头报告/Oral 级别的“高级感”怎么来

最强故事不是“我们加了能量 loss”。而是：

> Energy-based molecule generators have explicit energy but expensive, diversity-reducing MCMC. Flow generators are fast and diverse but energy-agnostic. BGFM is the first external-oracle calibrated amortized generator: it learns a fast flow proposal, calibrates its probability geometry with universal neural-potential forces and densities, and uses a short learned-energy corrector to obtain low-strain molecules with better physical-quality/diversity/compute Pareto.

要冲 oral，必须有三张漂亮图：

1. EBMol vs BGFM Pareto frontier；
2. independent xTB/DFT relaxation distribution，BGFM 全曲线左移；
3. negative controls 图：true physical labels 有效，shuffled labels 无效。

---

## 9. 我改了什么文件

生成的 patch 主要修：

- `cfm_mol/bgfm_loss.py`：降 exact Boltzmann claim，标注 `compute_bgfm_step` 是 deprecated contract。
- `cfm_mol/bgfm_train_hook.py`：修 energy head per-node/per-graph target、lambda4 ramp、device、过强注释。
- `cfm_mol/refinement.py`：加 recenter；多图 Metropolis 不再 silent pass。
- `scripts/sample_bgfm.py`：重写成可用的 FlowMol3 API + energy head corrector sampler。
- `configs/omol25_4m_bgfm_v10_full.yaml`：修过期注释。

---

## 10. Bottom line

现在这个课题值得继续冲，而且比之前更高级了。但要诚实：当前 zip 是 **strong idea + partial implementation + paper overclaim**。如果按我给的方向修，最后可以变成：

> **BGFM: a universal-neural-potential calibrated amortized 3D molecular generator that beats energy-based samplers on the quality-diversity-compute Pareto frontier under independent physical evaluators.**

这才是 ICLR 主会甚至 oral 可以讨论的故事。
