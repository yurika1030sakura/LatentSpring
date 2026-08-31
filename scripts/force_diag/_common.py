"""Shared plumbing for the force-loss diagnostic experiments (A: t-sweep,
B: gradient cosine).

NOTHING here trains anything or writes to the manuscript.  Everything is
forward/backward passes on an existing checkpoint.

Three responsibilities:
  1. load a FlowMol3 + BGFM checkpoint and PROVE its weights are finite
     before anything is scored (a NaN checkpoint has produced a wrong
     conclusion in this project before);
  2. build a FIXED, reproducible set of held-out minibatches whose prior
     sample x_0 is drawn ONCE, so a sweep over t is paired (same x_0, same
     x_1, only t moves);
  3. flat-gradient utilities (grad -> flat vector, cosine, per-block cosine).

Score read-out note (measured, not assumed -- see report):
  The configs used for Paper 1 set ``parameterization: ctmc``.  Under that
  parameterization FlowMol3's ``vector_field(...)['x']`` is the ENDPOINT
  prediction x_hat_1 (flowmol/models/flowmol.py builds the 'x' target as
  ``data_src['x_1_true']``), not the velocity v.  The training hook
  (cfm_mol/bgfm_train_hook.py) feeds that output to
  ``score_from_fm_velocity`` as if it were v.  Since
  v = (x_hat_1 - x_t)/(1-t), the implemented read-out equals
  (1-t) x the score implied by the endpoint prediction.  Both read-outs are
  computed here:
     'as_implemented'   s = (t * x_hat_1 - x_t) / ((1-t) * sigma^2)
     'endpoint_correct' s = (t * x_hat_1 - x_t) / ((1-t)^2 * sigma^2)
  'as_implemented' is the primary number because it is what the reported
  baseline actually optimised.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCORE_NORM_CAP = 1000.0


# ---------------------------------------------------------------------------
# checkpoint / model
# ---------------------------------------------------------------------------
def checkpoint_health(ckpt_path: str) -> dict:
    """Load the raw state dict and report finiteness. Raises on any non-finite."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    sd = state.get("state_dict", state)
    n_tensors = 0
    n_elems = 0
    n_nonfinite = 0
    bad = []
    absmax = 0.0
    for k, v in sd.items():
        if not torch.is_tensor(v) or not v.is_floating_point() or v.numel() == 0:
            continue
        n_tensors += 1
        n_elems += int(v.numel())
        nf = int((~torch.isfinite(v)).sum().item())
        if nf:
            n_nonfinite += nf
            bad.append({"key": k, "n_nonfinite": nf, "numel": int(v.numel())})
        else:
            absmax = max(absmax, float(v.abs().max().item()))
    out = {
        "checkpoint": str(ckpt_path),
        "global_step": int(state.get("global_step", -1)) if isinstance(state, dict) else -1,
        "epoch": int(state.get("epoch", -1)) if isinstance(state, dict) else -1,
        "n_float_tensors": n_tensors,
        "n_float_elements": n_elems,
        "n_nonfinite": n_nonfinite,
        "max_abs_weight": absmax,
        "nonfinite_tensors": bad[:20],
    }
    if n_nonfinite:
        raise RuntimeError(
            f"REFUSING TO SCORE: checkpoint {ckpt_path} has {n_nonfinite} "
            f"non-finite weights in {len(bad)} tensors.")
    return out


def load_model(config_path, checkpoint_path, device, patch: bool = True):
    """Return (model, cfg, bgfm_cfg, health_dict). Model is in eval() mode with
    requires_grad on so parameter gradients can be taken."""
    from flowmol.model_utils.load import model_from_config, read_config_file

    health = checkpoint_health(checkpoint_path)

    cfg = read_config_file(Path(config_path))
    bgfm_cfg = dict(cfg.get("mol_fm", {}).pop("bgfm", {}) or {})
    atom_map = cfg["dataset"]["atom_map"]

    model = model_from_config(cfg)
    if patch:
        from cfm_mol.domain import default_d_min_table
        from cfm_mol.flow_model import patch_flowmol
        n_real = len(atom_map)
        has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
        has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
        n_total = n_real + int(has_fake) + int(has_mask)
        d_min = torch.zeros(n_total, n_total)
        d_min[:n_real, :n_real] = default_d_min_table(n_atom_types=n_real,
                                                      atom_map=atom_map)
        e_weight = cfg["mol_fm"].get("total_loss_weights", {}).get("e", 2.0)
        bond_free = float(e_weight) == 0.0
        patch_flowmol(model, d_min, tangent=True, retract=True, gluing=True,
                      discrete_projection=not bond_free,
                      train_time_discrete=not bond_free, atom_map=atom_map)

    state = torch.load(str(checkpoint_path), map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    health["load_missing_keys"] = len(missing)
    health["load_unexpected_keys"] = len(unexpected)
    health["load_missing_sample"] = list(missing)[:10]
    health["load_unexpected_sample"] = list(unexpected)[:10]

    model = model.to(device)
    model.eval()
    # Determinism: the FM forward adds coordinate noise when distort_p > 0.
    health["distort_p_disabled_from"] = float(getattr(model, "distort_p", 0.0))
    model.distort_p = 0.0
    for p in model.parameters():
        p.requires_grad_(True)

    # post-load finiteness of the LIVE parameters (catches a partial load).
    live_nonfinite = 0
    for p in model.parameters():
        live_nonfinite += int((~torch.isfinite(p)).sum().item())
    health["live_param_nonfinite"] = live_nonfinite
    if live_nonfinite:
        raise RuntimeError("REFUSING TO SCORE: live model parameters non-finite "
                           "after load_state_dict.")
    health["n_trainable_params"] = sum(p.numel() for p in model.parameters()
                                       if p.requires_grad)
    return model, cfg, bgfm_cfg, health


# ---------------------------------------------------------------------------
# fixed held-out batches
# ---------------------------------------------------------------------------
def build_val_batches(cfg, n_minibatches: int, batch_size: int, max_atoms: int,
                      seed: int, device, split: str = "val"):
    """Build n_minibatches FIXED minibatches of held-out molecules.

    The prior sample x_0 is drawn inside MoleculeDataset.__getitem__, so it is
    baked into the graph here ONCE.  Every later use of these graphs re-reads
    the SAME x_0 -> a sweep over t is paired by construction.
    """
    import dgl
    from flowmol.data_processing.dataset import MoleculeDataset

    ds_cfg = dict(cfg["dataset"])
    ds_cfg["fake_atom_p"] = 0.0
    ds_cfg["fake_atom_std"] = 1.0
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)
    ds = MoleculeDataset(split, ds_cfg, prior_config=cfg["mol_fm"]["prior_config"])

    n_atoms_arr = ds.node_idx_array[:, 1] - ds.node_idx_array[:, 0]
    eligible = (n_atoms_arr <= max_atoms).nonzero(as_tuple=True)[0].tolist()
    if len(eligible) < n_minibatches * batch_size:
        raise RuntimeError(
            f"only {len(eligible)} {split} molecules with <= {max_atoms} atoms; "
            f"need {n_minibatches * batch_size}")
    g_cpu = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(eligible), generator=g_cpu).tolist()
    picked = [eligible[i] for i in perm[:n_minibatches * batch_size]]

    torch.manual_seed(seed)  # makes the prior draw in __getitem__ reproducible
    graphs = [ds[i] for i in picked]
    batches = []
    for b in range(n_minibatches):
        chunk = graphs[b * batch_size:(b + 1) * batch_size]
        batches.append(dgl.batch(chunk).to(device))
    meta = {
        "split": split,
        "n_minibatches": n_minibatches,
        "batch_size": batch_size,
        "max_atoms_filter": max_atoms,
        "n_eligible": len(eligible),
        "dataset_indices": picked,
        "seed": seed,
    }
    return batches, meta


# ---------------------------------------------------------------------------
# score read-out (both variants)
# ---------------------------------------------------------------------------
def score_readout(vf_x, x_t, t_per_atom, variant: str, prior_std: float = 1.0):
    """Implied score from the vector-field 'x' output.

    variant='as_implemented'   -> exactly cfm_mol.bgfm_loss.score_from_fm_velocity
    variant='endpoint_correct' -> the score implied when vf_x is read as the
                                  endpoint prediction x_hat_1 (extra 1/(1-t)).
    Both apply the same per-atom SCORE_NORM_CAP clamp the training path uses.
    """
    from cfm_mol.bgfm_loss import score_from_fm_velocity
    if variant == "as_implemented":
        return score_from_fm_velocity(vf_x, x_t, t_per_atom, prior_std=prior_std)
    if variant != "endpoint_correct":
        raise ValueError(variant)
    omt = (1.0 - t_per_atom).clamp(min=1e-3)
    t = t_per_atom
    if t.dim() < x_t.dim():
        t = t.view(-1, *[1] * (x_t.dim() - 1))
        omt = omt.view(-1, *[1] * (x_t.dim() - 1))
    score = (t * vf_x - x_t) / (omt ** 2 * (prior_std ** 2))
    norm = score.norm(dim=-1, keepdim=True).clamp(min=1.0)
    scale = (SCORE_NORM_CAP / norm).clamp(max=1.0)
    return score * scale


# ---------------------------------------------------------------------------
# gradient utilities
# ---------------------------------------------------------------------------
def named_trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def grad_flat(loss, named):
    """Flattened gradient of `loss` wrt the named parameters, plus per-name slices.
    Missing grads are treated as zeros. Does not touch .grad buffers."""
    params = [p for _, p in named]
    grads = torch.autograd.grad(loss, params, retain_graph=False,
                                create_graph=False, allow_unused=True)
    chunks = []
    spans = {}
    off = 0
    for (n, p), gr in zip(named, grads):
        gr = torch.zeros_like(p) if gr is None else gr
        flat = gr.reshape(-1).detach().float()
        chunks.append(flat)
        spans[n] = (off, off + flat.numel())
        off += flat.numel()
    return torch.cat(chunks), spans


def cosine(a, b, eps: float = 1e-12):
    na = a.norm()
    nb = b.norm()
    if float(na) < eps or float(nb) < eps:
        return float("nan")
    return float((a @ b) / (na * nb))


def block_name(pname: str) -> str:
    """Coarse parameter block label (cheap breakdown for the report)."""
    parts = pname.split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else pname


def per_block_cosine(a, b, spans):
    """cos per coarse parameter block, plus each block's share of ||g||^2."""
    agg = {}
    for name, (i, j) in spans.items():
        blk = block_name(name)
        d = agg.setdefault(blk, {"dot": 0.0, "aa": 0.0, "bb": 0.0, "numel": 0})
        av, bv = a[i:j], b[i:j]
        d["dot"] += float(av @ bv)
        d["aa"] += float(av @ av)
        d["bb"] += float(bv @ bv)
        d["numel"] += (j - i)
    out = {}
    for blk, d in agg.items():
        denom = math.sqrt(d["aa"]) * math.sqrt(d["bb"])
        out[blk] = {
            "cos": (d["dot"] / denom) if denom > 0 else float("nan"),
            "norm_a": math.sqrt(d["aa"]),
            "norm_b": math.sqrt(d["bb"]),
            "numel": d["numel"],
        }
    return out


def summarize(vals):
    """mean / sd / var / min / max of a list of floats, NaN-safe."""
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "var": None,
                "min": None, "max": None, "n_dropped": len(vals)}
    m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / (n - 1) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": math.sqrt(var), "var": var,
            "min": min(v), "max": max(v), "n_dropped": len(vals) - n}


def dump_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)
    print(f"[force_diag] wrote {p}", flush=True)
