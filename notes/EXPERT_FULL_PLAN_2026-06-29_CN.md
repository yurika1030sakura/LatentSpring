# BGFM：框架与代码完整修改方案

> 目标：把项目从“容易被审稿人认为 claim 过强的 Boltzmann generator”改成“与当前代码一致、可复现、可审稿的 OMol25-guided conditional-coordinate Boltzmann-regularized de novo 3D flow matching”。

---

## 0. 一句话总纲

不要把 BGFM 写成“生成分子的概率等于自然 Boltzmann 分布”。要改成：

> BGFM keeps a FlowMol3-style de novo 3D molecular flow backbone, and adds training-time OMol25 energy/force regularizers so that the model’s coordinate density, conditional on molecular identity/composition/charge, is locally aligned with OMol25-induced Boltzmann relative probabilities.

中文：

> BGFM 保留 FlowMol3 风格生成主干，在训练时加入 OMol25 能量/力，让固定分子组成/元素/电荷后的坐标分布，在局部扰动上更符合 OMol25 能量定义的 Boltzmann 相对概率。

---

## 1. 框架怎么改

### 1.1 旧框架的问题

当前 paper / code comments 里有几类危险表述：

1. **全局 Boltzmann claim 过强**  
   不能写 `p(x) ∝ exp(-E(x)/kT)` 且让 `x` 包含 atom count、atom type、charge、coordinates。代码只计算 position-only FFJORD density，所以目标只能是 conditional coordinate density。

2. **“exact Boltzmann sampler”过强**  
   当前实现是有限 perturbation、finite ODE steps、Hutchinson trace、late-time score approximation，不是精确采样器。

3. **“没有改 flow 底层”不完全准确**  
   主干没换，但 `cfm_mol/flow_model.py` 确实对 conditional path / sampler step / retraction / tangent projection 做了 wrappers。

4. **log-Z predictor 不是 thermodynamic partition function estimator**  
   它没有 true log-Z label，只是 composition-conditioned offset stabilizer。

5. **v7c force-only label 和 config 不一致**  
   如果 config 中 `lambda_2 != 0`，就不能叫 force-only，除非 runtime override 过。

### 1.2 新框架：四层结构

建议把方法写成四层：

#### Layer A: FlowMol3-style de novo backbone

负责生成 atom type / charge / coordinates。论文里承认这是 FlowMol3-style backbone，不把它作为核心 novelty。

#### Layer B: Geometry-preserving path/sampler hooks

在 `cfm_mol/flow_model.py` 中实现：

- `sample_conditional_path` 后 retract 到 steric fibre；
- sampler `step` 前后做 tangent projection / retraction；
- final discrete projection 保证 valence/connectivity；
- optional BGFM score guidance。

论文中写：

> We keep the neural backbone and categorical flow formulation, but wrap the conditional path and sampler with geometry-preserving hooks.

#### Layer C: OMol25 force-score regularizer

核心公式：

\[
\mathcal L_{\rm force}
=
\mathbb E_{t\in\mathcal T}
\left[1-\cos\left(s_\theta(x_t,t), F_{\rm NP}(x_1)/kT\right)\right].
\]

注意当前代码是：

- `s_theta` 在 conditional path 的 `x_t` 上算；
- force target 是 endpoint 的 `F(x_1)`；
- 所以必须写成 **late-time force-score regularizer**，不要写成 exact force matching at `x_t`。

#### Layer D: OMol25 energy-density variance regularizer

核心公式：

\[
\mathcal L_{\rm energy}
=
\mathbb E_m\,\mathrm{Var}_{k}\left[
\log p^{\rm pos}_\theta(r_m^{(k)}\mid c_m)
+ E_{\rm NP}(r_m^{(k)},c_m)/kT
\right].
\]

这才是最像 ICLR 的 novelty：同一 parent molecule 的多个 perturbations 内，低能应该对应高概率，概率差和能量差符合 Boltzmann ratio。

#### Layer E: optional log-Z offset anchor

写成：

\[
\mathcal L_{\rm anchor}
=
\mathbb E_{m,k}\left[
\rho\left(
\log p_\theta^{\rm pos}(r_m^{(k)}\mid c_m)
+ E_{\rm NP}(r_m^{(k)},c_m)/kT
+ b_\phi(c_m)
\right)\right]
\]

其中 `b_phi(c_m)` 是 composition-conditioned offset，不要叫 true partition function。

---

## 2. 代码总改法：优先级排序

### P0：必须修，否则 paper/code 不一致

1. `scripts/eval_boltzmann_stage1.py`：不要把 charge 固定写成 0。  
2. `scripts/eval_boltzmann_stage1.py`：T-conditional checkpoint 评估前必须 patch kT conditioning。  
3. `cfm_mol/bgfm_train_hook.py`：T-conditional energy loss 必须把 `kT_tensor` 传给 FFJORD density。  
4. `scripts/eval_boltzmann_stage2.py`：输出不要说 “learned density IS Boltzmann”，改成 “locally Boltzmann-aligned”。  
5. `cfm_mol/bgfm_loss.py`：force_loss docstring 改成 `s(x_t)` vs `F(x_1)`，避免 reviewer 说你代码和论文不一致。

### P1：强烈建议修，防止审稿人抓 reproducibility

6. `configs/*v8*.yaml`：训练用 perturbation shards 不要包含 val shard；val shard 只用于 eval。  
7. `configs/*v7c*.yaml`：要么把 `lambda_2` 改成 0 并叫 force-only，要么保留 `lambda_2=0.001` 但把 label 改成 mixed force+energy。  
8. `scripts/run_train.py`：加入 `lambda_2/lambda_3/kT/energy_every_k_steps` override，方便 ablation。  
9. `cfm_mol/bgfm_train_hook.py`：`kT_conditioning` 不应该依赖 `energy_enabled`，即使只跑 force T-cond 也应该 patch。  
10. `cfm_mol/log_z_predictor.py`：把 docstring 从 “predicts log-Z” 改成 “predicts composition offset”。

### P2：增强 ICLR 说服力

11. 加 negative controls：`shuffle_atoms` force target、shuffled energy within parent、wrong kT。  
12. 加 unit tests：score formula、FFJORD sign、within-parent variance、kT propagation、charge export。  
13. 加 result scripts：自动输出 `R2 / Pearson / Spearman / slope / BFGS steps / MD overlap`。

---

## 3. 文件级修改细节

## 3.1 `cfm_mol/bgfm_loss.py`

### 要改的概念

这个文件应该只负责：

- 从 FM velocity 得到 implied score；
- 用 OMol25 force 做 late-time score alignment；
- 组合 loss diagnostics。

不要在 docstring 里说模型已经精确满足 global Boltzmann。

### 修改 1：模块 docstring

把开头从类似：

```python
p_theta(x) proportional to exp(-E_OMol25(x) / kT) on the data support
```

改成：

```python
BGFM regularizes the coordinate density p_theta(r | c) toward local
OMol25-induced Boltzmann relative probabilities.  The density term is
position-only and conditional on fixed molecular identity/composition.
```

### 修改 2：`score_from_fm_velocity` docstring

保留公式：

```python
score = (t * v_theta - x_t) / ((1 - t) * prior_std**2)
```

但删掉 “DOUBLE-CHECK” 和不确定注释。写清楚：

```python
At t=1 this expression is singular. BGFM therefore evaluates it at late
but finite t_eval values, e.g. 0.70, 0.80, 0.90.
```

### 修改 3：`force_loss` docstring

当前代码真实做的是：

```text
s_theta(x_t, t_eval)  vs  F(x_1) / kT
```

所以 docstring 应改成：

```python
"""Late-time force-score regularizer.

In the current training hook, s_theta is evaluated at a conditional-path
point x_t, while forces_data is the precomputed OMol25 endpoint force F(x_1).
This is a late-time approximation F(x_t)≈F(x_1), used to avoid online
OMol25 calls inside every training step.
"""
```

---

## 3.2 `cfm_mol/bgfm_density.py`

### 要改的概念

这个文件是最核心 contribution：FFJORD density + within-parent energy variance。

### 修改 1：开头 docstring 降 claim

把：

```python
which is what makes the "Boltzmann" in BGFM real
log p_theta(x) = -E(x)/kT + const, i.e., the Boltzmann distribution
```

改成：

```python
This module implements position-only FFJORD density for coordinates r,
conditioned on fixed molecular identity c.  The loss enforces local
relative Boltzmann consistency within perturbations of the same parent
molecule, not an exact global molecular Boltzmann sampler.
```

### 修改 2：保留 position-only API

当前函数已经比较好：

```python
def make_position_velocity_fn(..., kT: torch.Tensor | None = None)
def log_density_via_flow(..., kT: torch.Tensor | None = None)
def energy_consistency_loss_per_mol(..., kT_tensor: torch.Tensor | None = None)
def energy_consistency_loss_per_mol_with_anchor(..., kT_tensor: torch.Tensor | None = None)
```

这说明代码已经支持 T-conditional density，只是训练 hook 和 eval 没完全传进去。

### 修改 3：variance loss 保持 within-parent

保留：

```python
within_group_variance_loss(residual, parent_id)
```

不要回到 cross-batch variance。cross-batch variance 会混入不同 molecule 的 `log Z_m` 和 size/composition confound。

### 修改 4：anchor 名字改弱

`energy_anchor_loss` 不要说 “true log-Z”。建议注释：

```python
log_Z_pred is a learned composition-dependent offset b_phi(c), not a
supervised thermodynamic partition-function estimator.
```

---

## 3.3 `cfm_mol/bgfm_train_hook.py`

这个文件是最需要改的地方，因为它决定 paper 里的方法到底有没有真的跑。

### 修改 1：`lambda_2` docstring

当前 docstring 旧注释说：

```python
lambda_2: float (must be 0.0 in current hook; energy term not active)
```

这是错的。要改成：

```python
lambda_2: float (energy-consistency term; active when > 0)
```

### 修改 2：anchor comment 降 claim

把：

```python
Prevents the trivial-constant failure mode...
```

改成：

```python
Stabilizes the absolute density offset.  The variance term constrains
relative probabilities but is insensitive to per-parent additive constants.
```

### 修改 3：T-conditional energy loss 必须传 `kT_tensor`

在 energy block 里，拿到 perturbation batch 后加：

```python
kT_pert = None
if getattr(self, '_bgfm_kT_conditioning', False):
    kT_pert = torch.full((g_pert.batch_size,), float(kT_step),
                         device=g_pert.device, dtype=torch.float32)
```

然后传给两个 energy loss：

```python
energy_consistency_loss_per_mol_with_anchor(...,
    kT=kT_step,
    kT_tensor=kT_pert,
    ...)
```

以及：

```python
energy_consistency_loss_per_mol(...,
    kT=kT_step,
    kT_tensor=kT_pert,
    ...)
```

否则 v8b 的 residual 用了 sampled kT，但 FFJORD vector field 没有用这个 kT 条件，T-conditional claim 不成立。

### 修改 4：`kT_conditioning` 不应该放在 `if energy_enabled:` 里面

现在代码里 `patch_kT_conditioning` 在 energy loader 初始化 block 里面。建议移到外面：

```python
if kT_conditioning:
    from cfm_mol.kt_conditioning import patch_kT_conditioning
    n_hidden_scalars = int(getattr(model.vector_field, 'n_hidden_scalars', 256))
    patch_kT_conditioning(model.vector_field, n_hidden_scalars)
    model._bgfm_kT_conditioning = True
    model._bgfm_kT_min = kT_min
    model._bgfm_kT_max = kT_max
```

这样即使 `lambda_2=0`，force-only T-conditional ablation 也能跑。

### 修改 5：PerturbationLoader device 不要完全 hard-code

当前：

```python
loader = PerturbationLoader(..., device="cuda", ...)
log_z_pred = LogZPredictor(...).to("cuda")
```

推荐：

```python
model_device = next(model.parameters()).device
loader_device = bgfm_config.get(
    'energy_loader_device',
    'cuda' if torch.cuda.is_available() else str(model_device)
)
loader = PerturbationLoader(..., device=loader_device, ...)
log_z_pred = LogZPredictor(...).to(model_device)
```

注意：如果用 Lightning DDP，真正最干净的方法是 first training step lazy init loader on `self.device`，但 log-Z predictor 必须在 optimizer 创建前注册，所以 predictor 仍应在 patch 阶段挂到 model 上。

### 修改 6：force-corrected target 的注释改弱

当前注释里有 “provably closer to Boltzmann”。这个 reviewer 会抓。改成：

```python
This optional target shift is a heuristic energy-descent augmentation.
It should be treated as an ablation, not as the main BGFM guarantee.
```

---

## 3.4 `cfm_mol/kt_conditioning.py`

当前设计合理：zero-init residual projection，`log(kT)` 输入，hook 到 scalar embedding。

需要补两个点：

1. 在 eval 里必须先 patch 再 load checkpoint；否则 `_kT_projection` 的权重会成为 unexpected keys。
2. `sample_kT` 是 per-step single kT，不是 per-molecule kT。论文要写清楚：当前 v8b samples one scalar kT per training step and broadcasts to all molecules in that batch。

如要更强，可以改成 per-graph kT：

```python
kT_step = sample_kT(g.batch_size, device, kT_min, kT_max)
```

然后 force loss 需要 per-node kT：

```python
kT_per_atom = kT_step[node_batch_idx]
```

不过这会影响 `force_loss` API，目前不建议在主线里改，除非要主打 multi-T。

---

## 3.5 `cfm_mol/log_z_predictor.py`

### 修改重点

把它从 “log-Z predictor” 改成 “composition offset head”。类名可以暂时不改，因为改名会影响 checkpoint；但文中和注释要改。

### 推荐 docstring

```python
"""Composition-conditioned offset head for BGFM anchor loss.

This module predicts a scalar b_phi(c) from invariant composition features.
It is not supervised by true thermodynamic log Z and should not be interpreted
as a guaranteed partition-function estimator.  It stabilizes the additive
constant left unconstrained by the within-parent variance loss.
"""
```

---

## 3.6 `scripts/eval_boltzmann_stage1.py`

这个文件必须修。

### 修改 1：保存真实 charge

当前导出：

```python
"charge": 0,
```

要改成：

```python
ns, ne = int(val.node_idx_array[di, 0]), int(val.node_idx_array[di, 1])
mol_charge = int(val.atom_charges[ns:ne].sum().item())
```

导出：

```python
"charge": mol_charge,
```

否则 charged slice / OMol25 energy evaluation 会错。

### 修改 2：增加 `--kT`

argparse 加：

```python
ap.add_argument("--kT", type=float, default=None,
                help="Optional kT in eV for T-conditional FFJORD eval.")
```

### 修改 3：pop bgfm 前保存 bgfm cfg

当前：

```python
cfg = read_config_file(args.config)
cfg.get("mol_fm", {}).pop("bgfm", None)
```

改成：

```python
cfg = read_config_file(args.config)
bgfm_eval_cfg = dict(cfg.get("mol_fm", {}).get("bgfm", {}) or {})
cfg.get("mol_fm", {}).pop("bgfm", None)
```

### 修改 4：T-conditional checkpoint 要先 patch 再 load state_dict

在 `model_from_config(cfg)` 后、`load_state_dict` 前加：

```python
if bool(bgfm_eval_cfg.get("kT_conditioning", False)):
    from cfm_mol.kt_conditioning import patch_kT_conditioning
    n_hidden_scalars = int(getattr(model.vector_field, 'n_hidden_scalars', 256))
    patch_kT_conditioning(model.vector_field, n_hidden_scalars)
    if args.kT is None:
        args.kT = float(bgfm_eval_cfg.get("kT", 0.025))
```

### 修改 5：FFJORD eval 时传 kT

```python
kT_tensor = None
if args.kT is not None:
    kT_tensor = torch.full((gbatch.batch_size,), float(args.kT),
                           device=device, dtype=torch.float32)

logp = log_density_via_flow(..., kT=kT_tensor)
```

---

## 3.7 `scripts/eval_boltzmann_stage2.py`

### 修改 1：解释文本改弱

当前：

```python
learned density IS Boltzmann
```

改成：

```python
local coordinate density is Boltzmann-aligned on these perturbations
```

### 修改 2：建议输出更多统计

现在有 Pearson、slope、pooled R²。建议加 Spearman：

```python
from scipy.stats import spearmanr
rho = float(spearmanr(logp, negE_kT).correlation)
```

如果不想加 scipy 依赖，可以用 rankdata 自己实现。

---

## 3.8 `scripts/run_train.py`

### 修改 1：增加 ablation override

现在只能 override `lambda_1`、force loss type、target mode、probe mode、t_eval。建议加：

```python
p.add_argument('--override_bgfm_lambda_2', type=float, default=None)
p.add_argument('--override_bgfm_lambda_3', type=float, default=None)
p.add_argument('--override_bgfm_kT', type=float, default=None)
p.add_argument('--override_bgfm_energy_every_k_steps', type=int, default=None)
```

然后：

```python
if args.override_bgfm_lambda_2 is not None:
    bgfm_cfg['lambda_2'] = float(args.override_bgfm_lambda_2)
if args.override_bgfm_lambda_3 is not None:
    bgfm_cfg['lambda_3'] = float(args.override_bgfm_lambda_3)
if args.override_bgfm_kT is not None:
    bgfm_cfg['kT'] = float(args.override_bgfm_kT)
if args.override_bgfm_energy_every_k_steps is not None:
    bgfm_cfg['energy_every_k_steps'] = int(args.override_bgfm_energy_every_k_steps)
```

这样你可以轻松跑：

```bash
# force-only
python scripts/run_train.py --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --override_bgfm_lambda_2 0 --override_bgfm_lambda_3 0

# energy-only
python scripts/run_train.py --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --override_bgfm_lambda_1 0

# shuffled force negative control
python scripts/run_train.py --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --override_bgfm_force_target_mode shuffle_atoms
```

---

## 3.9 `configs/*.yaml`

### 训练 config 不要混入 val perturbation shards

当前 v8 configs 里 training energy shards 包含：

```yaml
- perturbation_val_n10000_s0.pt
- perturbation_train_n30000_s0.pt
```

建议训练只用 train：

```yaml
energy_perturbation_shards:
  - /n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed/perturbation_train_n30000_s0.pt
```

val shard 保留给 eval，不用于 training loss。

### v7c label 修正

二选一：

#### 方案 A：真的做 force-only

```yaml
lambda_1: 0.05
lambda_2: 0.0
lambda_3: 0.0
```

然后重新训练/评估，paper 叫 force-only。

#### 方案 B：保留现有 checkpoint

如果现有 v7c 确实用了：

```yaml
lambda_2: 0.001
```

那 paper / code map / table 里必须叫：

```text
BGFM-v7c mixed force+energy, kT=1 eV
```

不能叫 force-only。

### 推荐保留的 ablation configs

1. `FM baseline`: `bgfm.enabled=false` or no bgfm block。  
2. `BGFM-force`: `lambda_1>0, lambda_2=0, lambda_3=0`。  
3. `BGFM-energy`: `lambda_1=0, lambda_2>0, lambda_3>0`。  
4. `BGFM-full`: `lambda_1>0, lambda_2>0, lambda_3>0`。  
5. `BGFM-T`: full + `kT_conditioning=true`。  
6. Negative controls: shuffled force / shuffled energy / wrong kT。

---

## 4. 训练与评估流程怎么改

### Step 1：预训练 / baseline

```bash
python scripts/run_train.py \
  --config configs/omol25_4m_bgfm.yaml \
  --override_output_dir runs/fm_baseline
```

### Step 2：预计算 perturbation shards

训练 shard：

```bash
python scripts/precompute_energy_perturbations.py \
  --split train \
  --n_molecules 30000 \
  --sigmas 0.03,0.06,0.10,0.20,0.40 \
  --out /path/perturbation_train_n30000_s0.pt
```

验证 shard：

```bash
python scripts/precompute_energy_perturbations.py \
  --split val \
  --n_molecules 10000 \
  --sigmas 0.03,0.06,0.10,0.20,0.40 \
  --out /path/perturbation_val_n10000_s0.pt
```

### Step 3：训练 full BGFM

```bash
python scripts/run_train.py \
  --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --resume_from runs/fm_baseline/.../last.ckpt
```

### Step 4：训练 ablations

```bash
# force-only
python scripts/run_train.py \
  --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --override_bgfm_lambda_2 0 \
  --override_bgfm_lambda_3 0 \
  --override_output_dir runs/bgfm_force_only

# energy-only
python scripts/run_train.py \
  --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --override_bgfm_lambda_1 0 \
  --override_output_dir runs/bgfm_energy_only

# shuffled force negative control
python scripts/run_train.py \
  --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --override_bgfm_force_target_mode shuffle_atoms \
  --override_output_dir runs/bgfm_shuffle_force
```

### Step 5：Level-1 Boltzmann correlation eval

Stage 1: FlowMol env, compute log p:

```bash
python scripts/eval_boltzmann_stage1.py \
  --checkpoint runs/bgfm_full/checkpoints/last.ckpt \
  --config configs/omol25_4m_bgfm_energy_v8a_room_T.yaml \
  --eval_data /path/omol25_4m_processed \
  --n_molecules 60 \
  --n_perturb 16 \
  --sigma 0.15 \
  --kT 0.025 \
  --max_atoms_filter 60 \
  --out_dir runs/eval/bgfm_full_level1
```

Stage 2: OMol25 env, compute energies + correlation:

```bash
python scripts/eval_boltzmann_stage2.py \
  --samples_json runs/eval/bgfm_full_level1/boltzmann_samples.json \
  --out_csv runs/eval/bgfm_full_level1/boltzmann_correlation.csv \
  --kT 0.025
```

For T-conditional checkpoints, run multiple kT:

```bash
for kt in 0.025 0.05 0.1 0.25 1.0; do
  python scripts/eval_boltzmann_stage1.py \
    --checkpoint runs/bgfm_T/checkpoints/last.ckpt \
    --config configs/omol25_4m_bgfm_energy_v8b_T_conditional.yaml \
    --eval_data /path/omol25_4m_processed \
    --n_molecules 60 --n_perturb 16 --sigma 0.15 \
    --kT $kt \
    --out_dir runs/eval/bgfm_T_kT_${kt}
  python scripts/eval_boltzmann_stage2.py \
    --samples_json runs/eval/bgfm_T_kT_${kt}/boltzmann_samples.json \
    --out_csv runs/eval/bgfm_T_kT_${kt}/boltzmann_correlation.csv \
    --kT $kt
done
```

---

## 5. 推荐单元测试

### Test 1：score formula toy

目标：验证

```python
s = (t*v - x) / ((1-t)*sigma^2)
```

在 1D Gaussian conditional path 上符号正确。

### Test 2：FFJORD sign

构造简单 linear flow `x_t = scale(t) x_0`，解析 log-density 可算，检查 `log_density_via_flow` 的 sign。

### Test 3：within-parent variance

构造两组 parent：

```python
residual = [-1, 0, 1, 100, 101, 102]
parent_id = [0, 0, 0, 1, 1, 1]
```

within variance 应忽略 parent offset 100。

### Test 4：kT propagation

mock vector_field，传不同 kT 返回不同 velocity，检查 `energy_consistency_loss_per_mol(kT_tensor=...)` 确实改变 logp。

### Test 5：charge export

构造 charged molecule，跑 Stage1，检查 JSON 里的 `charge != 0`。

---

## 6. 论文对应怎么改

### Title

推荐：

```text
Boltzmann-Regularized Flow Matching for Universal-Neural-Potential Guided 3D Molecular Generation
```

### Abstract 核心句

```text
BGFM keeps a FlowMol3-style de novo 3D molecular flow backbone, but uses
OMol25 neural-potential energies and forces to regularize the model's
coordinate density, conditional on molecular identity, toward local
Boltzmann relative probabilities.
```

### Contributions

写三点：

1. late-time force-score consistency from OMol25 forces；
2. FFJORD position-density + within-parent energy variance；
3. universal-NP-guided evaluation protocol: Boltzmann correlation, relaxation savings, ensemble overlap。

### Limitations 必须写

1. position-only density，不是 full molecular joint density；
2. fixed-composition conditional Boltzmann，不是 grand-canonical distribution；
3. force loss currently uses endpoint force `F(x1)` as late-time approximation for `x_t`；
4. log-Z head is offset stabilizer, not true partition-function estimator；
5. OMol25 coverage/spin/charge limitations。

---

## 7. 最终应该改成的 project story

大白话：

> FlowMol3 会生成像数据的 3D 分子。BGFM 不换这套主干，而是在训练时加一个 OMol25 物理老师。这个老师给两个信号：力告诉模型“概率应该往哪边升”，能量告诉模型“低能扰动应该比高能扰动更可能”。所以 BGFM 不是精确 Boltzmann sampler，而是把 de novo 生成器的坐标分布往 conditional Boltzmann relative probabilities 拉近。

这个 story 和代码最一致，也最适合 ICLR。
