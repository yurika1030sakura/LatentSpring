"""Patched FlowMol3 training entry point.

Launches FlowMol3's pl.Trainer-based training loop, but with our constrained
patches applied to the model.

Mirrors baselines/flowmol3/train.py but inserts patch_flowmol() after model
instantiation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from flowmol.model_utils.load import (
    data_module_from_config,
    model_from_config,
    read_config_file,
)
from flowmol.utils.ema import ExponentialMovingAverage

import torch
from cfm_mol.domain import default_d_min_table
from cfm_mol.flow_model import patch_flowmol
from cfm_mol.bgfm_train_hook import patch_flowmol_bgfm




def _loss_summary(weights: dict, bgfm_cfg: dict | None) -> str:
    parts = []
    for feat in ("x", "a", "c", "e"):
        val = float(weights.get(feat, 1.0))
        suffix = " SKIP" if val == 0.0 else ""
        parts.append(f"{feat}({val:g}{suffix})")
    if bgfm_cfg is not None and bgfm_cfg.get("enabled", False):
        t_vals = bgfm_cfg.get("t_eval_values", bgfm_cfg.get("t_eval", 0.95))
        force_loss_type = bgfm_cfg.get("force_loss_type", "mse")
        force_target_mode = bgfm_cfg.get("force_target_mode", "true")
        probe_mode = bgfm_cfg.get("probe_mode", "path")
        parts.append(
            f"bgfm.force(lambda={float(bgfm_cfg.get('lambda_1', 0.0)):g}, "
            f"type={force_loss_type}, target={force_target_mode}, "
            f"probe={probe_mode}, t={t_vals})"
        )
        if float(bgfm_cfg.get("lambda_2", 0.0)) > 0.0:
            parts.append(f"bgfm.energy(lambda={float(bgfm_cfg.get('lambda_2')):g})")
    return " ".join(parts)


def _print_run_header(args, cfg: dict, bgfm_cfg: dict | None, bond_free: bool) -> None:
    prefix = "[smoke]" if args.fast else "[train]"
    weights = cfg["mol_fm"].get("total_loss_weights", {})
    bgfm_enabled = bool(bgfm_cfg is not None and bgfm_cfg.get("enabled", False))
    print(f"{prefix} config: {args.config}", flush=True)
    print(
        f"{prefix} dataset: {cfg['dataset'].get('dataset_name')} "
        f"processed={cfg['dataset'].get('processed_data_dir')}",
        flush=True,
    )
    print(
        f"{prefix} mode: bond_free={bond_free} "
        f"bgfm_enabled={bgfm_enabled} "
        f"parameterization={cfg['mol_fm'].get('parameterization')}",
        flush=True,
    )
    print(f"{prefix} active losses: {_loss_summary(weights, bgfm_cfg)}", flush=True)
    print(
        f"{prefix} trainer: batch_size={cfg['training'].get('batch_size')} "
        f"max_num_edges={cfg['training'].get('max_num_edges')} "
        f"accumulate_grad_batches="
        f"{cfg['training'].get('trainer_args', {}).get('accumulate_grad_batches')}",
        flush=True,
    )


def _latest_checkpoint_in(path: Path) -> Path | None:
    if not path.exists():
        return None
    last = path / "last.ckpt"
    if last.exists():
        return last
    candidates = [p for p in path.glob("*.ckpt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _slurm_resume_checkpoint(output_dir: str | Path) -> Path | None:
    """Resume only from the same SLURM job version, avoiding stale smoke ckpts."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    root = Path(output_dir)
    search_dirs = [
        root / "lightning_logs" / f"version_{job_id}" / "checkpoints",
        root / "checkpoints",
    ]
    for ckpt_dir in search_dirs:
        ckpt = _latest_checkpoint_in(ckpt_dir)
        if ckpt is not None:
            return ckpt
    return None


def _checkpoint_callbacks(cfg: dict, fast: bool) -> list[pl.Callback]:
    ckpt_cfg = cfg.get("checkpointing", {})
    callbacks: list[pl.Callback] = []

    monitor = ckpt_cfg.get("monitor", "val_total_loss")
    save_top_k = int(ckpt_cfg.get("save_top_k", 3))
    every_n_epochs = int(ckpt_cfg.get("every_n_epochs", 1))
    callbacks.append(ModelCheckpoint(
        monitor=monitor,
        save_top_k=save_top_k,
        save_last=bool(ckpt_cfg.get("save_last", True)),
        every_n_epochs=every_n_epochs,
        filename="{epoch}-{step}",
    ))

    every_n_train_steps = int(ckpt_cfg.get("every_n_train_steps", 0) or 0)
    if not fast and every_n_train_steps > 0:
        callbacks.append(ModelCheckpoint(
            save_top_k=-1,
            save_last=False,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=0,
            filename="midstep-{step}",
        ))
    return callbacks


class FiniteWeightGuard(pl.Callback):
    """Halt training the moment the weights go non-finite.

    Why this exists: the BGFM energy term's within-parent variance estimator is
    high-variance at small energy_b_parents. A finite-but-huge outlier produces a
    gradient that blows the weights to NaN. The in-loop `isfinite(L_energy)` guard
    then fires forever and silently ZEROES the loss -- so training happily
    continues for days, writing corrupted checkpoints, and the run *looks* healthy
    (loss printed as nan, but nothing stops). We lost a run to exactly this.

    Fail loudly and immediately instead: the last good checkpoint stays usable.
    """

    def __init__(self, every_n_steps: int = 200):
        super().__init__()
        self.every_n_steps = int(every_n_steps)

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = int(trainer.global_step)
        if step == 0 or step % self.every_n_steps != 0:
            return
        bad = []
        for name, p in pl_module.named_parameters():
            if p is not None and p.is_floating_point() and not torch.isfinite(p).all():
                bad.append(name)
                if len(bad) >= 5:
                    break
        if bad:
            msg = (f"[FiniteWeightGuard] NON-FINITE WEIGHTS at global_step={step}. "
                   f"First offenders: {bad}. Halting so the last good checkpoint "
                   f"stays usable. Likely cause: an L_energy / L_force outlier "
                   f"blew up the gradient (see bgfm.energy_loss_cap, "
                   f"bgfm.energy_b_parents).")
            print(msg, flush=True)
            raise RuntimeError(msg)

class BatchStatsCallback(pl.Callback):
    """Print first-batches shape/memory stats so smoke logs are self-auditing."""

    def __init__(self, prefix: str, max_batches: int = 5):
        super().__init__()
        self.prefix = prefix
        self.max_batches = max_batches
        self.max_atoms: list[int] = []
        self.total_edges: list[int] = []
        self._printed_summary = False

    @staticmethod
    def _to_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        return list(x)

    @staticmethod
    def _mem_line() -> str:
        if not torch.cuda.is_available():
            return "cuda_mem=unavailable"
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
        return (
            f"cuda_mem={alloc:.2f}GiB peak={peak:.2f}GiB "
            f"reserved={reserved:.2f}GiB total={total:.1f}GiB"
        )

    def on_fit_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _print_batch_stats(self, stage: str, batch, batch_idx: int, collect: bool) -> None:
        if batch_idx >= self.max_batches and collect:
            return
        if not hasattr(batch, "batch_num_nodes"):
            return
        nodes = self._to_list(batch.batch_num_nodes())
        edges = self._to_list(batch.batch_num_edges())
        max_atoms = int(max(nodes)) if nodes else 0
        total_edges = int(sum(edges)) if edges else int(batch.num_edges())
        max_edges = int(max(edges)) if edges else 0
        if collect:
            self.max_atoms.append(max_atoms)
            self.total_edges.append(total_edges)
        print(
            f"{self.prefix} {stage} batch {batch_idx}: graphs={len(nodes)} "
            f"max_atoms={max_atoms} mean_atoms={sum(nodes)/max(len(nodes), 1):.1f} "
            f"total_nodes={int(sum(nodes))} max_edges={max_edges} "
            f"total_edges={total_edges} {self._mem_line()}",
            flush=True,
        )
        if collect and len(self.max_atoms) == self.max_batches and not self._printed_summary:
            print(
                f"{self.prefix} first {self.max_batches} train batches: "
                f"max_atoms={self.max_atoms} total_edges={self.total_edges}",
                flush=True,
            )
            self._printed_summary = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._print_batch_stats("train", batch, batch_idx, collect=True)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx < min(self.max_batches, 2):
            stage = "sanity-val" if getattr(trainer, "sanity_checking", False) else "val"
            self._print_batch_stats(stage, batch, batch_idx, collect=False)

    def on_train_end(self, trainer, pl_module):
        print(f"{self.prefix} peak CUDA memory: {self._mem_line()}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    # E3 ablation flags (default: all hooks on).
    p.add_argument('--no-tangent', action='store_true',
                   help='Disable tangent-projection hook (E3 ablation).')
    p.add_argument('--no-retract', action='store_true',
                   help='Disable post-step retraction hook (E3 ablation).')
    p.add_argument('--no-gluing', action='store_true',
                   help='Disable interpolation retraction (E3 ablation).')
    p.add_argument('--no-patch', action='store_true',
                   help='Skip patching entirely: train vanilla FlowMol3 baseline.')
    p.add_argument('--fast', action='store_true',
                   help='Fast dev-run: 1 epoch, 10 train batches, 1 val batch. '
                        'For GPU pipeline smoke-test on gpu_test partition.')
    p.add_argument('--resume_from', type=Path, default=None,
                   help='Checkpoint path to resume training from (fine-tune).')
    p.add_argument('--resume_ckpt_path', type=Path, default=None,
                   help='Full Lightning trainer-state resume (continues epoch/'
                        'optimizer/scheduler). Use for walltime restart.')
    p.add_argument('--no_auto_resume', action='store_true',
                   help='Do not auto-resume from the latest checkpoint under '
                        'the same SLURM job version.')
    p.add_argument('--override_output_dir', type=Path, default=None,
                   help='Override cfg.training.output_dir for controlled '
                        'fine-tune/ablation runs.')
    p.add_argument('--override_base_lr', type=float, default=None,
                   help='Override cfg.lr_scheduler.base_lr.')
    p.add_argument('--override_max_epochs', type=int, default=None,
                   help='Override cfg.training.trainer_args.max_epochs.')
    p.add_argument('--override_bgfm_lambda_1', type=float, default=None,
                   help='Override bgfm.lambda_1.')
    p.add_argument('--override_bgfm_force_loss_type',
                   choices=['mse', 'cosine', 'norm_mse'], default=None,
                   help='Override bgfm.force_loss_type.')
    p.add_argument('--override_bgfm_force_target_mode',
                   choices=['true', 'shuffle_atoms'], default=None,
                   help='Override bgfm.force_target_mode. shuffle_atoms is a '
                        'negative-control ablation that preserves the marginal '
                        'force distribution but breaks geometry-force pairing.')
    p.add_argument('--override_bgfm_probe_mode',
                   choices=['path', 'endpoint'], default=None,
                   help='Override bgfm.probe_mode.')
    p.add_argument('--override_bgfm_t_eval_values', default=None,
                   help='Comma-separated override for bgfm.t_eval_values, '
                        'for example 0.92,0.97,0.985.')
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    cfg = read_config_file(args.config)

    # The 'bgfm' sub-block lives under mol_fm in our configs but FlowMol's
    # __init__ doesn't accept it. Extract it before model construction; we
    # apply it later via patch_flowmol_bgfm.
    bgfm_cfg = cfg.get('mol_fm', {}).pop('bgfm', None)
    if args.override_output_dir is not None:
        cfg['training']['output_dir'] = str(args.override_output_dir)
    if args.override_base_lr is not None:
        cfg.setdefault('lr_scheduler', {})['base_lr'] = float(args.override_base_lr)
    if args.override_max_epochs is not None:
        cfg['training'].setdefault('trainer_args', {})['max_epochs'] = int(args.override_max_epochs)
    if bgfm_cfg is not None:
        if args.override_bgfm_lambda_1 is not None:
            bgfm_cfg['lambda_1'] = float(args.override_bgfm_lambda_1)
        if args.override_bgfm_force_loss_type is not None:
            bgfm_cfg['force_loss_type'] = args.override_bgfm_force_loss_type
        if args.override_bgfm_force_target_mode is not None:
            bgfm_cfg['force_target_mode'] = args.override_bgfm_force_target_mode
        if args.override_bgfm_probe_mode is not None:
            bgfm_cfg['probe_mode'] = args.override_bgfm_probe_mode
        if args.override_bgfm_t_eval_values:
            bgfm_cfg['t_eval_values'] = [
                float(x.strip()) for x in args.override_bgfm_t_eval_values.split(',')
                if x.strip()
            ]
    e_weight = cfg['mol_fm'].get('total_loss_weights', {}).get('e', 2.0)
    bond_free = float(e_weight) == 0.0
    _print_run_header(args, cfg, bgfm_cfg, bond_free)

    # Data
    datamodule = data_module_from_config(cfg)

    # Model
    model = model_from_config(cfg)

    # --- cfm_mol integration ---------------------------------------------
    if args.no_patch:
        print("[cfm_mol] --no-patch: training vanilla FlowMol3 baseline.")
    else:
        atom_map = cfg['dataset']['atom_map']
        n_real = len(atom_map)
        has_fake = cfg['mol_fm'].get('fake_atom_p', 0.0) > 0
        has_mask = cfg['mol_fm'].get('parameterization', '') == 'ctmc'
        n_total = n_real + int(has_fake) + int(has_mask)
        d_min = torch.zeros(n_total, n_total)
        d_min[:n_real, :n_real] = default_d_min_table(
            n_atom_types=n_real, atom_map=atom_map,
        )
        # Bond-free detection: if bond-type loss weight is 0 we disable the
        # valence/connectivity discrete projections (they assume a bond
        # concept which doesn't exist in a bond-free regime). The geometric
        # pair-distance hooks (tangent / retract / gluing) stay active
        # because they operate on positions only.
        if bond_free:
            print("[cfm_mol] bond-free mode: disabling discrete_projection + "
                  "train_time_discrete (valence/connectivity hooks require bonds).")
        patch_flowmol(
            model, d_min,
            tangent=not args.no_tangent,
            retract=not args.no_retract,
            gluing=not args.no_gluing,
            discrete_projection=not bond_free,
            train_time_discrete=not bond_free,
            atom_map=atom_map,
        )
        print("[cfm_mol] constraints patched into FlowMol.")

        # BGFM hook (Boltzmann-Guided Flow Matching) — optional Level-2 upgrade.
        # Activated if bgfm.enabled is true. Adds force-consistency auxiliary
        # loss at data endpoint (t = t_eval near 1). bgfm_cfg was popped from
        # cfg['mol_fm'] above so it doesn't reach FlowMol.__init__.
        if bgfm_cfg is not None and bgfm_cfg.get('enabled', False):
            patch_flowmol_bgfm(model, bgfm_cfg)
    # ---------------------------------------------------------------------

    # --- guard against early-training posebusters crash ------------------
    # Early-training samples can be degenerate (zero atoms after sanitize),
    # and FlowMol3's SampleAnalyzer -> posebusters crashes on that.
    # Wrap analyze() so it swallows those errors and returns empty metrics,
    # letting training continue.
    if hasattr(model, 'sample_analyzer') and model.sample_analyzer is not None:
        _orig_analyze = model.sample_analyzer.analyze
        def _safe_analyze(*a, **kw):
            try:
                return _orig_analyze(*a, **kw)
            except Exception as e:
                print(f"[cfm_mol] sample_analyzer.analyze raised {type(e).__name__}: "
                      f"{e}. Skipping metrics for this interval.")
                return {}
        model.sample_analyzer.analyze = _safe_analyze
        print("[cfm_mol] sample_analyzer wrapped with error guard.")
    # ---------------------------------------------------------------------

    # Logger (optional)
    logger = None
    if cfg.get('wandb', {}).get('mode', 'disabled') != 'disabled':
        logger = WandbLogger(
            project=cfg['wandb']['project'],
            name=cfg['wandb'].get('name'),
            mode=cfg['wandb'].get('mode'),
        )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=50),
        BatchStatsCallback(prefix="[smoke]" if args.fast else "[train]"),
        FiniteWeightGuard(every_n_steps=200),
    ]
    callbacks.extend(_checkpoint_callbacks(cfg, fast=args.fast))

    trainer_kwargs = dict(cfg['training']['trainer_args'])
    if args.fast:
        print("[run_train] --fast: overriding max_epochs=1, "
              "limit_train_batches=10, limit_val_batches=1")
        trainer_kwargs.update(dict(
            max_epochs=1,
            limit_train_batches=10,
            limit_val_batches=1,
        ))

    trainer = pl.Trainer(
        default_root_dir=cfg['training']['output_dir'],
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    fit_kwargs = {"datamodule": datamodule}
    if args.resume_ckpt_path is not None:
        fit_kwargs["ckpt_path"] = str(args.resume_ckpt_path)
        print(f"[run_train] resuming full trainer state from {args.resume_ckpt_path}")
    elif not args.no_auto_resume and args.resume_from is None and not args.fast:
        auto_ckpt = _slurm_resume_checkpoint(cfg['training']['output_dir'])
        if auto_ckpt is not None:
            fit_kwargs["ckpt_path"] = str(auto_ckpt)
            print(f"[run_train] auto-resuming same SLURM job from {auto_ckpt}")
    if args.resume_from is not None:
        # Load model WEIGHTS ONLY (not trainer state). Start fresh epochs
        # with the (potentially new) hooks active — this is fine-tune mode,
        # not true resume.
        import torch as _torch
        ckpt = _torch.load(str(args.resume_from), map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[run_train] loaded weights from {args.resume_from}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    trainer.fit(model, **fit_kwargs)


if __name__ == '__main__':
    main()
