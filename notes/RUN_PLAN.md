# First full run: capped-step sweep → scale the winner (FASRC / H200)

Strategy: run a cheap capped-step sweep (`max_steps=30000`, ~overnight) across the physics
recipes + baseline + negative control, eval all, then scale the winner **plus the FM-only
twin** to full 4M for the headline. Fills the empty **"Ours — HBFM (4M, full)"** row in
`notes/benchmark_plan.md` (Tables 2 & 3) and the `FM-only (ablation)` row.

## The sweep (5 configs, `configs/sweep/*.yaml`) — ALL smoke-tested on an H200

| Config | What it isolates | Physics signal at 10-batch smoke |
|---|---|---|
| `fm_only` | baseline, physics OFF (fair twin) | no force loss (correct) |
| `bgfm_offpolicy` | λ₁ force loss on precomputed DFT labels | score_force_cos **+0.046** |
| `hbfm_onpolicy` | λ₁ + live eSEN teacher distillation (50k winner) | L_onpolicy=495, cos +0.046 |
| `hbfm_onpolicy_energy` | full three-term (+ λ₂ energy) | L_energy=402, L_onpolicy=476, cos +0.046 |
| `shuffle_control` | negative control (shuffled force targets) | score_force_cos **−0.013** (signal destroyed ✓) |

All five reach `max_epochs`/complete cleanly, no OOM, no NaN/no_force skip. The
off-policy-vs-shuffle contrast (+0.046 vs −0.013) already shows the physics signal is real.

### Submit + eval
```bash
cd /n/home04/yulili/bgfm
bash scripts/fasrc/submit_sweep.sh                 # 5 jobs, gpu_h200, -A woo_lab
#   on a 4-GPU node up to 4 run in parallel; on-policy jobs self-start their eSEN
#   teacher on ports 28901/28902 (moved off the node's VNC 59xx range).
# when a run finishes, eval it (fills the ours-rows):
sbatch scripts/fasrc/eval_h200.slurm \
   $STORE/runs/sweep/<name>/lightning_logs/version_<jobid>/checkpoints/last.ckpt \
   configs/sweep/<name>.yaml <name>
```

### Scale the winner to full 4M
Winner (by Boltzmann-R² + validity) + `fm_only` → full 4M:
```bash
sbatch scripts/fasrc/train_bgfm_onpolicy_h200.slurm   # or train_bgfm_4m_h200.slurm (off-policy)
# FM-only twin at full 4M:
sbatch scripts/fasrc/train_bgfm_4m_h200.slurm configs/omol25_4m_fm_only.yaml
```

## Gotchas learned while smoke-testing (already baked into the configs/scripts)
- **Teacher ports moved to 289xx.** This node runs VNC on 5900/5901 → `Address already in use`.
  On-policy configs use 28901/28902; the standalone launcher uses 28900. `sweep.slurm` reads the
  port from the config so worker+trainer always agree.
- **Energy term (`hbfm_onpolicy_energy`) is memory-heavy.** The FFJORD divergence is a
  double-backward; at `batch_size=16` it OOM'd (135 GB). Baked-in fix: `batch_size=2`,
  `energy_b_parents=1`, `energy_max_atoms_per_parent=50`, `energy_n_hutchinson=2` → 16.5 GiB peak.
  Treat this arm as experimental; if it misbehaves at scale, the headline stands on the other 3.
- **`env.sh` sets node-local `TMPDIR`** (avoids NFS teardown tracebacks) and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Auto-resume is job-scoped** (`version_<SLURM_JOB_ID>`), so smoke/leftover checkpoints in an
  output_dir never poison a fresh real run.

## Ablations to add later (subset scale — for paper rigor, not the headline)
- kT: 1.0 eV vs 0.025 (room-T); λ₁ sweep 0.05/0.1/0.2. Generate like the sweep configs.
- The `shuffle_control` is already the key negative control.
