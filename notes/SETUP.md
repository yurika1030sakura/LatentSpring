# Running HBFM/BGFM on a new machine (e.g. H200)

What **git carries** vs. what you **set up locally** (large/external → not in git):

| In git (pull the repo) | Set up locally |
|---|---|
| `cfm_mol/` (method), `scripts/` (train/eval/preprocess + SLURM templates + **flowmol3 patch** + env recipes), `configs/`, `notes/` (PROJECT.md, appendix_hbc.tex, benchmark_plan.md) | conda envs, OMol25 data, eSEN oracle ckpt, training checkpoints, `baselines/` (FlowMol3 etc.) |

> ⚠️ **The single most important step is #2** (apply the FlowMol3 force patch). Without it,
> the BGFM physics loss silently trains plain FM (this exact bug cost us a full run).

## 0. Clone
```
git clone git@github.com:yurika1030sakura/bgfm.git && cd bgfm && git checkout norman
```

## 1. Build the two conda envs
Two envs are required (torch version conflict — do not merge):
- **`envs/flowmol`** — training/inference: torch 2.2 (+CUDA for your H200, e.g. cu124) + DGL 2.0.0 + PyTorch-Lightning 2.1.3 + FlowMol3 (`pip install -e baselines/flowmol3`) + this repo on `PYTHONPATH`; also `pip install pytest posebusters datamol loguru`.
- **`envs/omol25`** — preprocessing + teacher potential: torch 2.8 + fairchem-core 2.21.0 + ase 3.29.
Recipes (adapt CUDA/module-load for the H200 box): `scripts/setup/build_flowmol_env.sh`, `scripts/setup/build_omol25_env.sh`. Also install `xtb` (conda-forge) into `envs/flowmol` — needed for the independent-potential eval.

## 2. ⚠️ Apply the FlowMol3 force patch (CRITICAL)
Stock FlowMol3's `MoleculeDataset` does **not** wire DFT forces into the graph, so the physics
loss is a silent no-op. Fix:
```
bash scripts/flowmol3_patch/apply.sh   # drops patched dataset.py into the installed flowmol3
```
(or apply `scripts/flowmol3_patch/dataset_forces.patch`.) **Verify** after training starts:
the log must show `train_L_force` / `train_score_force_cos`; if you ever see
`train_bgfm_no_force_skip`, the patch didn't take.

## 3. Data + oracle
- **OMol25 4M** (no auth): `wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz` then `tar xzf` → 160 `.aselmdb` shards.
- **eSEN oracle** (gated — request HF access to `facebook/OMol25` + `facebook/UMA`): download
  `esen_sm_conserving_all.pt`; set `HF_TOKEN=...` and `HF_HUB_DISABLE_XET=1`.

## 4. Preprocess — MUST store forces (omol25 env)
```
envs/omol25/bin/python scripts/preprocess_omol25.py \
  --src <train_4M_dir> --out_dir <proc>/omol25_4m --max_atoms 200 --bond_tolerance 1.2
```
Faster first run: add `--max_n_mols 1000000` (1M subset). The output `.pt` must contain a
`forces` key (the patch in #2 reads it into `force_1_true`).

## 5. Config — add the on-policy BGFM block
Copy `configs/omol25_4m_bgfm.yaml` → `configs/omol25_4m_bgfm_onpolicy.yaml`; set
`dataset.processed_data_dir` and `training.output_dir` to your paths; then under `mol_fm.bgfm`:
```yaml
    lambda_1: 0.1
    lambda_onpolicy: 0.1
    onpolicy_K_steps: 10
    onpolicy_every_k_steps: 8
    onpolicy_t: 0.9
    onpolicy_rpc_url: http://127.0.0.1:5900
    onpolicy_clip_force: 50.0
    kT: 1.0
    t_eval_values: [0.85, 0.92, 0.97]   # optionally [0.9] for ~2x faster
```
**Batch size:** on-policy peaks ~27 GB at `batch_size=2` on a 46 GB L40S. An H200 (141 GB) can
run `batch_size` ~8–12 — set `training.batch_size` accordingly (big speedup vs the L40S runs).

## 6. Train (teacher worker must be co-located)
```
# 1) start the teacher (omol25 env), wait for "serving on":
envs/omol25/bin/python scripts/omol25_worker.py --port 5900 --device cuda --ckpt <esen_ckpt>
# 2) train (flowmol env):
PYTHONPATH=. envs/flowmol/bin/python scripts/run_train.py --config configs/omol25_4m_bgfm_onpolicy.yaml
```
SLURM template that orchestrates both: `scripts/unity/train_bgfm_onpolicy.slurm` (adapt
partition/account/paths — it's Unity-specific). Sanity: `train_score_force_cos` should climb
positive (our 50k run went −0.10 → +0.36); `train_L_force` should drop.

## 7. Eval — fills the `ours` rows in `notes/benchmark_plan.md`
All in `envs/flowmol` with `xtb` on PATH:
- **Table 2** (de-novo): `scripts/sample_to_json.py` → `benchmarks/omol25/validity_from_json.py`
  (validity + PoseBusters config="mol" all-checks + uniqueness) + `scripts/eval_xtb_relaxation.py`.
- **Table 3** (Boltzmann fidelity): `scripts/eval_boltzmann_stage1.py` (FFJORD log p) →
  `scripts/eval_boltzmann_independent.py` (**independent GFN2-xTB Boltzmann-R² + ESS**).
- **Table 1** (GEOM COV/MAT, zero-shot): `scripts/eval_geom_covmat.py` (+ datamol; download the
  torsional-diffusion `test_mols.pkl` for DRUGS/QM9).
- One-shot template (adapt paths): `scripts/unity/eval_onpolicy.slurm <ckpt> <config> <tag> [proc_dir]`.

## Gotchas / notes
- **Step 2 patch** or physics silently no-ops. **`xtb`** on PATH for the independent Boltzmann
  eval + xTB ΔE. **Sign conventions are verified correct** (`notes/appendix_hbc.tex`).
- Full 4M is slow on one GPU (~3 days/epoch on L40S; faster on H200 but still large) — a 1M
  subset (~1 epoch overnight on H200) or multi-GPU DDP is the practical headline path.
- Everything else about the project (background, method, theory, experiment design) is in
  `notes/PROJECT.md`; verified baseline numbers + the `ours` rows are in `notes/benchmark_plan.md`.
