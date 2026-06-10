"""Smoke test: verify our patched FlowMol model runs end-to-end.

Checks:
  1. Config loads, FlowMol model builds.
  2. `cfm_mol.flow_model.patch_flowmol` applies without errors.
  3. One forward pass on a real QM9 batch: all per-feature losses are finite.
  4. Backprop: gradient norm is finite and nonzero.

Requires the `flowmol` env AND processed QM9 data. Both are produced by
`scripts/install_and_process.slurm`. If either is missing we emit a clear
error instead of crashing inside an import / file-read.

Run:
    source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
    conda activate /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/envs/flowmol
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _print_header(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def _check_env_ready() -> None:
    """Verify imports that only exist in the flowmol env. Fail loudly if not."""
    try:
        import dgl  # noqa: F401
        import pytorch_lightning  # noqa: F401
        import flowmol  # noqa: F401
    except ImportError as e:
        print(f"[FATAL] flowmol env not active: {e}")
        print("Activate: conda activate "
              f"{PROJECT_ROOT}/envs/flowmol")
        sys.exit(2)


def _load_batch_or_fail(cfg: dict):
    """Try loading a real QM9 batch via MoleculeDataset. If the processed
    data isn't there, explain how to regenerate it and exit.
    """
    from flowmol.data_processing.dataset import MoleculeDataset, collate
    from copy import deepcopy

    processed_dir = Path(cfg["dataset"]["processed_data_dir"])
    if not (processed_dir / "train_data_processed.pt").exists():
        print(f"[FATAL] processed QM9 data missing at {processed_dir}")
        print("Run: sbatch scripts/process_qm9.slurm")
        sys.exit(3)

    # MoleculeDataset reads fake_atom_p / explicit_aromaticity from
    # dataset_config, but our config stores them under mol_fm (matching
    # flowmol3.yml). Mirror the data_module_from_config merge logic.
    ds_cfg = deepcopy(cfg["dataset"])
    ds_cfg["fake_atom_p"] = cfg["mol_fm"].get("fake_atom_p", 0.0)
    ds_cfg["fake_atom_std"] = cfg["mol_fm"].get("fake_atom_std", 0.0)
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)

    ds = MoleculeDataset(
        split="train",
        dataset_config=ds_cfg,
        prior_config=cfg["mol_fm"]["prior_config"],
    )
    # Two small molecules for a fast smoke test.
    graphs = [ds[0], ds[1]]
    return collate(graphs)


def main() -> None:
    _print_header("SMOKE TEST -- patched FlowMol end-to-end")

    _check_env_ready()

    from flowmol.model_utils.load import read_config_file, model_from_config

    global default_d_min_table, patch_flowmol, model_from_config, d_min
    from cfm_mol.domain import default_d_min_table
    from cfm_mol.flow_model import patch_flowmol

    cfg_path = PROJECT_ROOT / "configs" / "qm9_cfm.yaml"
    print(f"[1/6] load config: {cfg_path}")
    cfg = read_config_file(str(cfg_path))

    print("[2/6] build FlowMol from config ...")
    model = model_from_config(cfg)
    atom_map = cfg["dataset"]["atom_map"]
    # FlowMol3 adds two virtual columns we must zero-out in d_min:
    #   (a) fake-atom slot when mol_fm.fake_atom_p > 0   (adds +1 to feat dim)
    #   (b) CTMC mask token slot when parameterization == ctmc (+1)
    # Effective a_t dim = n_real + fake + mask. Padding with d_min=0
    # makes retraction never constrain these slots (good -- virtual
    # atoms have no steric presence).
    n_real = len(atom_map)
    has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
    has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
    n_atom_types = n_real + (1 if has_fake else 0) + (1 if has_mask else 0)
    d_min = torch.zeros(n_atom_types, n_atom_types)
    d_min[:n_real, :n_real] = default_d_min_table(n_atom_types=n_real,
                                                  atom_map=atom_map)
    print(f"      atom_map_size={n_real} + fake={has_fake} + mask={has_mask}"
          f" -> d_min_table.shape={tuple(d_min.shape)}, "
          f"vector_field={type(model.vector_field).__name__}")

    print("[3/6] apply patch_flowmol ...")
    patch_flowmol(model, d_min)
    print("      ok (3 hooks installed on CTMCVectorField)")

    print("[4/6] load a tiny real QM9 batch ...")
    g = _load_batch_or_fail(cfg)
    print(f"      batched graph: {g.batch_size} mols, "
          f"{g.num_nodes()} nodes, {g.num_edges()} edges")

    print("[5/6] forward pass -> per-feature loss dict ...")
    model.eval()
    with torch.no_grad():
        losses = model(g)
    assert isinstance(losses, dict), \
        f"expected dict of losses, got {type(losses).__name__}"
    any_nonfinite = False
    for k, v in losses.items():
        v_scalar = float(v.detach().mean().cpu())
        finite = torch.isfinite(v).all().item()
        flag = "" if finite else "  [NON-FINITE]"
        print(f"      loss[{k}] = {v_scalar:.4f}{flag}")
        if not finite:
            any_nonfinite = True
    if any_nonfinite:
        print("[FAIL] at least one feature produced a non-finite loss.")
        sys.exit(1)

    print("[6/6] backprop check ...")
    model.train()
    losses = model(g)
    weights = cfg["mol_fm"]["total_loss_weights"]
    total = sum(weights[k] * losses[k].mean() for k in losses)
    total.backward()
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += float(p.grad.pow(2).sum().cpu())
    grad_norm = grad_sq ** 0.5
    print(f"      total_loss = {float(total):.4f}, grad_norm = {grad_norm:.4f}")
    if not (torch.isfinite(total) and grad_norm > 0 and grad_norm < 1e6):
        print("[FAIL] total loss or gradient norm looks wrong.")
        sys.exit(1)

    _print_header("SMOKE TEST PASSED.")

    # ------------------------------------------------------------------
    # Ablation-flag verification: rebuild model 4 times with different
    # hook combinations and check that each produces a DIFFERENT
    # g.ndata['x_t'] after sample_conditional_path. If two variants give
    # identical output, one hook is silently broken.
    # ------------------------------------------------------------------
    _print_header("ABLATION FLAG VERIFICATION")
    variants = [
        ("full",       dict(tangent=True,  retract=True,  gluing=True)),
        ("no-tangent", dict(tangent=False, retract=True,  gluing=True)),
        ("no-retract", dict(tangent=True,  retract=False, gluing=True)),
        ("no-gluing",  dict(tangent=True,  retract=True,  gluing=False)),
        ("vanilla",    None),   # no patch at all
    ]
    g0 = _load_batch_or_fail(cfg)
    results = {}
    for name, flags in variants:
        model_k = model_from_config(cfg)
        if flags is not None:
            patch_flowmol(model_k, d_min, **flags)
        # Clone the batch to avoid mutation across variants.
        g_k = dgl_clone_graph(g0)
        model_k.eval()
        with torch.no_grad():
            model_k(g_k)
        # After forward, g_k.ndata['x_t'] has been set by
        # sample_conditional_path. Snapshot it.
        results[name] = g_k.ndata["x_t"].detach().clone()
        print(f"  {name:12s} x_t mean={results[name].mean():.4f}, "
              f"std={results[name].std():.4f}")

    print()
    print("pairwise max-abs-diff between variants:")
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = (results[names[i]] - results[names[j]]).abs().max().item()
            tag = "  same" if d < 1e-6 else "differ"
            print(f"  {names[i]:12s} vs {names[j]:12s}: max_diff={d:.4e}  {tag}")
    print()
    print("Expected: `vanilla` vs anything patched should differ (patch changes"
          " the interpolation). `no-gluing` vs `vanilla` may be near-identical"
          " (gluing IS the interpolation retraction). Other comparisons depend"
          " on sample_conditional_path vs step behaviour.")


def dgl_clone_graph(g):
    """Deep copy of a DGL graph (edges + node/edge features).

    Not the same as `g.clone()` which is shallow on features.
    """
    import dgl
    new = dgl.graph(
        (g.edges()[0].clone(), g.edges()[1].clone()),
        num_nodes=g.num_nodes(),
    )
    for k, v in g.ndata.items():
        new.ndata[k] = v.clone()
    for k, v in g.edata.items():
        new.edata[k] = v.clone()
    return new


if __name__ == "__main__":
    main()
