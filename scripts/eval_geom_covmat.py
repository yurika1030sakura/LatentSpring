"""Zero-shot GEOM-DRUGS/QM9 conformer COV/MAT for our OMol25-trained flow model
(Table 1 "ours" row). We do NOT train on GEOM: we transfer zero-shot.

Pipeline per torsional-diffusion test molecule (from test_mols.pkl = {smiles:[Mol]}):
  1. take the reference ensemble pos_ref (T,N,3) in the molecule's native atom order;
  2. map each atom to the OMol25 index (index == Z-1) and per-atom formal charge;
  3. generate 2T conformers with our model (bond-free, identity fixed) -> pos_gen;
  4. build {smiles, pos_ref, pos_gen} exactly as ET-Flow does (dm.to_smiles with
     with_atom_indices=True; CovMatEvaluator rebuilds via dm.to_mol(ordered=True),
     which round-trips the atom order), then run CovMatEvaluator.

Run in envs/flowmol AFTER `pip install datamol loguru`. rdkit/torch/dgl already present.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import torch
import dgl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/
from level3_bgfm_sample import _build_molecule_template_graph, sample_positions_via_flow  # noqa: E402

DEFAULT_COVMAT = ("/home/renhaozhang_umass_edu/scratch_workspace/bgfm/baselines/"
                  "etflow/etflow/commons/covmat.py")


def _load_covmat(path):
    spec = importlib.util.spec_from_file_location("covmat_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_model(config, checkpoint, device):
    from flowmol.model_utils.load import model_from_config, read_config_file
    from cfm_mol.domain import default_d_min_table
    from cfm_mol.flow_model import patch_flowmol
    cfg = read_config_file(config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    atom_map = cfg["dataset"]["atom_map"]
    n_real = len(atom_map)
    has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
    has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
    n_total = n_real + int(has_fake) + int(has_mask)
    model = model_from_config(cfg)
    e_weight = cfg["mol_fm"].get("total_loss_weights", {}).get("e", 2.0)
    bond_free = float(e_weight) == 0.0
    d_min = torch.zeros(n_total, n_total)
    d_min[:n_real, :n_real] = default_d_min_table(n_atom_types=n_real, atom_map=atom_map)
    patch_flowmol(model, d_min, tangent=True, retract=True, gluing=True,
                  discrete_projection=not bond_free,
                  train_time_discrete=not bond_free, atom_map=atom_map)
    state = torch.load(str(checkpoint), map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[covmat] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    return model.to(device).eval(), n_total, atom_map


def _gen_confs(model, Z, charges, n_gen, n_total, device, K_steps=12, batch_per_pass=50):
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    atom_type_idx = np.array([z - 1 for z in Z], dtype=np.int64)          # OMol25 idx = Z-1
    charges_raw = np.clip(np.array(charges, dtype=np.int64), -2, 3)
    N = len(Z)
    out, remaining = [], n_gen
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_per_pass, remaining)
            graphs = [_build_molecule_template_graph(atom_type_idx, charges_raw, n_total,
                                                     device=device) for _ in range(bs)]
            g = dgl.batch(graphs)
            nbi, _ = get_batch_idxs(g)
            uem = get_upper_edge_mask(g)
            x = sample_positions_via_flow(model, g, nbi, uem, K_steps=K_steps, prior_std=1.0)
            out.append(x.cpu().numpy().reshape(bs, N, 3))
            remaining -= bs
    return np.concatenate(out, 0).astype(np.float32)                     # (n_gen, N, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--test_mols", required=True, help="torsional-diffusion test_mols.pkl")
    ap.add_argument("--covmat_py", default=DEFAULT_COVMAT)
    ap.add_argument("--partition", default="drugs", choices=["drugs", "qm9"])
    ap.add_argument("--threshold", type=float, default=None)             # 0.75 drugs / 0.5 qm9
    ap.add_argument("--max_mols", type=int, default=0, help="0 = all")
    ap.add_argument("--K_steps", type=int, default=12)
    ap.add_argument("--out_pkl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    thr = a.threshold if a.threshold is not None else (0.75 if a.partition == "drugs" else 0.5)

    import datamol as dm
    model, n_total, atom_map = _load_model(a.config, a.checkpoint, a.device)
    test = pickle.load(open(a.test_mols, "rb"))          # {smiles: [rdkit Mol, ...]}
    items = list(test.items())
    if a.max_mols:
        items = items[:a.max_mols]

    data_list, n_skip = [], 0
    for i, (_smi_key, mols) in enumerate(items):
        if not mols:
            n_skip += 1; continue
        ref = mols[0]
        Z = [at.GetAtomicNum() for at in ref.GetAtoms()]
        ch = [at.GetFormalCharge() for at in ref.GetAtoms()]
        if any(z < 1 or z > len(atom_map) for z in Z):   # element outside OMol25 map
            n_skip += 1; continue
        try:
            pos_ref = np.stack([m.GetConformer().GetPositions() for m in mols]).astype(np.float32)
        except Exception:
            n_skip += 1; continue
        T = pos_ref.shape[0]
        smiles = dm.to_smiles(ref, canonical=False, explicit_hs=True,
                              with_atom_indices=True, isomeric=True)
        pos_gen = _gen_confs(model, Z, ch, 2 * T, n_total, a.device, K_steps=a.K_steps)
        data_list.append({"smiles": smiles, "pos_ref": pos_ref, "pos_gen": pos_gen})
        if (i + 1) % 50 == 0:
            print(f"[covmat] generated {i + 1}/{len(items)} (skipped {n_skip})", flush=True)

    pickle.dump(data_list, open(a.out_pkl, "wb"))
    print(f"[covmat] {len(data_list)} molecules kept, {n_skip} skipped -> {a.out_pkl}", flush=True)

    cm = _load_covmat(a.covmat_py)
    # num_workers=1 -> serial map. The evaluator's Pool path (num_workers>1)
    # pickles worker_fn by module name, which fails because covmat.py is loaded
    # standalone via importlib (not an importable module). Serial avoids that.
    evaluator = cm.CovMatEvaluator(num_workers=1)
    out = evaluator(data_list, start_idx=0)
    results = out[0] if isinstance(out, tuple) else out
    cov_df, matching = cm.print_covmat_results(results)

    # Pull COV at the target threshold. cov_df has a "Threshold" column (not
    # index); COV values are fractions in [0,1] -> report as % to match the
    # literature (Table 1). AMR (MAT) is threshold-independent.
    thr_col = cov_df["Threshold"].to_numpy(dtype=float)
    idx = int(np.abs(thr_col - thr).argmin())
    row = cov_df.iloc[idx]
    summary = {
        "partition": a.partition, "threshold": float(thr_col[idx]),
        "n_molecules": len(data_list),
        "COV-R_mean_pct": 100.0 * float(row["COV-R_mean"]),
        "COV-R_median_pct": 100.0 * float(row["COV-R_median"]),
        "COV-P_mean_pct": 100.0 * float(row["COV-P_mean"]),
        "COV-P_median_pct": 100.0 * float(row["COV-P_median"]),
        "AMR-R_mean": float(matching.get("MAT-R_mean", float("nan"))),
        "AMR-R_median": float(matching.get("MAT-R_median", float("nan"))),
        "AMR-P_mean": float(matching.get("MAT-P_mean", float("nan"))),
        "AMR-P_median": float(matching.get("MAT-P_median", float("nan"))),
        "cov_curve": {  # COV% at the standard reporting thresholds
            f"{t:.2f}": {
                "COV-R_mean_pct": 100.0 * float(cov_df.iloc[j]["COV-R_mean"]),
                "COV-P_mean_pct": 100.0 * float(cov_df.iloc[j]["COV-P_mean"]),
            }
            for j, t in enumerate(thr_col) if abs((t * 20) - round(t * 20)) < 1e-6
        },
    }
    json.dump(summary, open(a.out_json, "w"), indent=2)
    print(f"[covmat] SUMMARY {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
