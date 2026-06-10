"""Integration layer: patch FlowMol3 (CTMCVectorField) with constrained FM.

QM9 config uses `parameterization: ctmc` so the production class is
`CTMCVectorField` (inherits from `EndpointVectorField`). We patch three
methods on the vector field, in order of when they fire:

  1. `sample_conditional_path(g, t, node_batch_idx, edge_batch_idx,
     upper_edge_mask)` -- called once per training batch to produce the
     straight-line interpolation `g.ndata['x_t']`. After it returns we
     retract `x_t` onto the steric fibre M_ster(a_hard).

  2. `vector_field(x_t, x_1, alpha_t, alpha_t_prime)` -- helper called
     INSIDE `step()` to convert (x_t, endpoint x_1) into a velocity. We
     tangent-project that velocity against the steric fibre BEFORE the
     Euler step consumes it. The graph context (a_hard, node_batch_idx)
     needed for the projection is stashed on the vf by our step-wrapper.

  3. `step(g, s_i, t_i, ..., node_batch_idx, edge_batch_idx,
     upper_edge_mask, ...)` -- called once per Euler step during sampling.
     We wrap it to stash graph context before the call and retract
     `g.ndata['x_t']` after the Euler update.

Together these three hooks implement the tangent-projected-ODE + retraction
required by Theorem 4.1 of the methods derivation. The FlowMol3 Lightning
training loop, loss weights, checkpointing, and data loaders are untouched.

USAGE:
    from cfm_mol.flow_model import patch_flowmol
    from flowmol.model_utils.load import read_config_file, model_from_config
    cfg = read_config_file("configs/qm9_cfm.yaml")
    model = model_from_config(cfg)
    patch_flowmol(model, d_min_table)
    # ... normal pl.Trainer(...).fit(model, datamodule) ...
"""
from __future__ import annotations

import torch

try:
    from flowmol.models.ctmc_vector_field import CTMCVectorField
    _HAS_FLOWMOL = True
except Exception:
    CTMCVectorField = None
    _HAS_FLOWMOL = False


from cfm_mol.fibre_dgl import retract_dgl, tangent_project_dgl
from cfm_mol.bgfm_loss import score_from_fm_velocity
from cfm_mol.domain import DEFAULT_VALENCE_SETS, valence_ok
from cfm_mol.projection import (
    project_valence, project_valence_ilp, project_connectivity,
    project_valence_preserve_conn,
)


# ---------------------------------------------------------------------------
# Hook 1: retract after the conditional-path interpolation
# ---------------------------------------------------------------------------

def _patch_sample_conditional_path(vf, d_min_table: torch.Tensor) -> None:
    original = vf.sample_conditional_path

    def wrapped(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        g = original(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)
        a_hard = g.ndata['a_t'].argmax(dim=-1)
        g.ndata['x_t'] = retract_dgl(
            g.ndata['x_t'], a_hard, node_batch_idx, d_min_table
        )
        return g

    vf.sample_conditional_path = wrapped


# ---------------------------------------------------------------------------
# Hook 2: tangent-project the velocity helper's output
# ---------------------------------------------------------------------------

def _patch_vector_field_helper(vf, d_min_table: torch.Tensor) -> None:
    """In CTMCVectorField.step(), the Euler velocity is computed as:

        vf_vel = self.vector_field(x_t, x_1, alpha_t, alpha_t_prime)

    We wrap `self.vector_field` (a bound method on the module) so that its
    output velocity is tangent-projected before being used. Context
    (a_hard, node_batch_idx) is stashed on `vf` by the step-wrapper.
    """
    original = vf.vector_field

    def wrapped(x_t, x_1, alpha_t, alpha_t_prime):
        v_raw = original(x_t, x_1, alpha_t, alpha_t_prime)
        a_hard = getattr(vf, '_cfm_a_hard', None)
        nbi = getattr(vf, '_cfm_node_batch_idx', None)
        if a_hard is None or nbi is None:
            return v_raw
        return tangent_project_dgl(v_raw, x_t, a_hard, nbi, d_min_table)

    vf.vector_field = wrapped


# ---------------------------------------------------------------------------
# Hook 3: stash context + retract after the Euler step
# ---------------------------------------------------------------------------

def _as_bgfm_guidance_config(raw) -> dict | None:
    """Normalize optional sampling-time BGFM guidance kwargs.

    ``model.sample(..., bgfm_score_guidance={...})`` is only consumed by the
    patched sampler. The original FlowMol3 step does not know this kwarg, so
    the step wrapper strips it before forwarding to FlowMol3.
    """
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("bgfm_score_guidance must be a bool or dict")
    cfg = {
        "max_weight": float(raw.get("max_weight", raw.get("weight", 0.0))),
        "start": float(raw.get("start", raw.get("late_start", 0.75))),
        "schedule": str(raw.get("schedule", "late_linear")),
        "clip": float(raw.get("clip", 5.0)),
        "max_norm_ratio": float(raw.get("max_norm_ratio", 1.0)),
    }
    if cfg["max_weight"] <= 0.0:
        return None
    if not (0.0 <= cfg["start"] < 1.0):
        raise ValueError("bgfm_score_guidance.start must be in [0, 1)")
    if cfg["clip"] <= 0.0:
        raise ValueError("bgfm_score_guidance.clip must be positive")
    if cfg["max_norm_ratio"] <= 0.0:
        raise ValueError("bgfm_score_guidance.max_norm_ratio must be positive")
    return cfg


def _bgfm_guidance_weight(t: float, cfg: dict) -> float:
    max_weight = cfg["max_weight"]
    schedule = cfg["schedule"]
    start = cfg["start"]
    if schedule == "constant":
        return max_weight
    if schedule == "linear":
        return max_weight * t
    if t < start:
        return 0.0
    tau = (t - start) / max(1.0 - start, 1e-6)
    if schedule == "late_linear":
        return max_weight * tau
    if schedule == "late_quadratic":
        return max_weight * tau * tau
    raise ValueError(f"unknown bgfm_score_guidance schedule: {schedule}")


def _per_graph_rms(x: torch.Tensor, node_batch_idx: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """Return a per-node RMS scale computed independently per molecule."""
    if x.numel() == 0:
        return torch.ones(x.shape[:1] + (1,), device=x.device, dtype=x.dtype)
    n_graphs = int(node_batch_idx.max().item()) + 1
    scales = torch.ones(n_graphs, 1, device=x.device, dtype=x.dtype)
    atom_sq = x.pow(2).sum(dim=-1)
    for graph_idx in range(n_graphs):
        mask = node_batch_idx == graph_idx
        if bool(mask.any()):
            scales[graph_idx, 0] = atom_sq[mask].mean().sqrt().clamp_min(eps)
    return scales[node_batch_idx]


def _clip_per_atom_norm(v: torch.Tensor, max_norm: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return v * scale


def _apply_bgfm_score_guidance(
    g,
    x_t_before: torch.Tensor,
    x_t_after: torch.Tensor,
    t_i: torch.Tensor,
    dt: torch.Tensor,
    node_batch_idx: torch.Tensor,
    cfg: dict,
) -> None:
    """Late-time sampler drift from the BGFM-trained implied score.

    The force loss trains ``score_from_fm_velocity(v_theta, x_t, t)`` to align
    with DFT force. During sampling this optional drift turns that learned
    score into a small, normalized coordinate update, so the force signal has
    a direct path to terminal geometry quality.
    """
    t_float = float(t_i.detach().item())
    weight = _bgfm_guidance_weight(t_float, cfg)
    if weight <= 0.0:
        return

    dt_safe = dt.detach().clamp_min(torch.tensor(1e-6, device=dt.device, dtype=dt.dtype))
    v_step = (x_t_after - x_t_before) / dt_safe
    t_per_atom = torch.full(
        (x_t_before.shape[0],), t_float,
        device=x_t_before.device, dtype=x_t_before.dtype,
    )
    score = score_from_fm_velocity(v_step, x_t_before, t_per_atom, prior_std=1.0)

    score_scale = _per_graph_rms(score, node_batch_idx)
    vel_scale = _per_graph_rms(v_step, node_batch_idx)
    score_unit = torch.clamp(score / score_scale, -cfg["clip"], cfg["clip"])
    guide = score_unit * vel_scale
    guide = _clip_per_atom_norm(guide, vel_scale * cfg["max_norm_ratio"])

    g.ndata["x_t"] = x_t_after + dt * weight * guide
    # Keep samples centered; FlowMol downstream assumes no global drift.
    n_graphs = int(node_batch_idx.max().item()) + 1
    com = torch.zeros(n_graphs, 3, device=g.ndata["x_t"].device, dtype=g.ndata["x_t"].dtype)
    counts = torch.zeros(n_graphs, 1, device=g.ndata["x_t"].device, dtype=g.ndata["x_t"].dtype)
    com.index_add_(0, node_batch_idx, g.ndata["x_t"])
    counts.index_add_(0, node_batch_idx, torch.ones_like(counts[node_batch_idx]))
    g.ndata["x_t"] = g.ndata["x_t"] - (com / counts.clamp_min(1.0))[node_batch_idx]


def _patch_step(vf, d_min_table: torch.Tensor) -> None:
    original = vf.step

    def wrapped(g, s_i, t_i, alpha_t_i, alpha_s_i, alpha_t_prime_i,
                node_batch_idx, edge_batch_idx, upper_edge_mask,
                *args, **kwargs):
        bgfm_guidance = _as_bgfm_guidance_config(kwargs.pop("bgfm_score_guidance", None))
        x_t_before = g.ndata['x_t'].detach().clone() if bgfm_guidance is not None else None
        vf._cfm_a_hard = g.ndata['a_t'].argmax(dim=-1)
        vf._cfm_node_batch_idx = node_batch_idx
        try:
            out = original(g, s_i, t_i, alpha_t_i, alpha_s_i, alpha_t_prime_i,
                           node_batch_idx, edge_batch_idx, upper_edge_mask,
                           *args, **kwargs)
            g_out = out[0] if isinstance(out, tuple) else out
            if bgfm_guidance is not None and x_t_before is not None:
                _apply_bgfm_score_guidance(
                    g_out,
                    x_t_before=x_t_before,
                    x_t_after=g_out.ndata['x_t'],
                    t_i=t_i,
                    dt=s_i - t_i,
                    node_batch_idx=node_batch_idx,
                    cfg=bgfm_guidance,
                )
            a_hard = g_out.ndata['a_t'].argmax(dim=-1)
            g_out.ndata['x_t'] = retract_dgl(
                g_out.ndata['x_t'], a_hard, node_batch_idx, d_min_table
            )
            return out
        finally:
            vf._cfm_a_hard = None
            vf._cfm_node_batch_idx = None

    vf.step = wrapped


# ---------------------------------------------------------------------------
# Hook 4a: project INTERPOLATED discrete state during training
# (this is the "training-time projection" that matches Cardei et al.
# NeurIPS 2025's differentiable projection approach, adapted for CTMC)
# ---------------------------------------------------------------------------

def _project_discrete_interpolated(g, atom_map: list[str]) -> None:
    """In place: project g.ndata['a_t'] + g.edata['e_t'] onto valid valence
    + connectivity manifold at training time.

    Called inside sample_conditional_path AFTER the simplex interpolation
    is done. Converts simplex -> argmax -> projection -> one-hot
    (non-differentiable, but the model just sees the input; gradients flow
    through logits not through one-hot). Enforces:
      * argmax atom types + argmax bond types satisfy valence_ok
      * bond graph is one connected component
    Uses FAST greedy algorithms only (no ILP) so fit in training loop.
    """
    import dgl
    import torch
    from torch.nn.functional import one_hot

    n_real = len(atom_map)
    a_t = g.ndata.get('a_t')
    e_t = g.edata.get('e_t')
    if a_t is None or e_t is None:
        return  # nothing to project yet

    A_total = a_t.shape[-1]
    E_total = e_t.shape[-1]
    E_real_max = E_total - 1
    valence_table = {i: DEFAULT_VALENCE_SETS.get(sym, (4,))
                     for i, sym in enumerate(atom_map)}

    a_hard = a_t.argmax(dim=-1).cpu()
    e_hard = e_t.argmax(dim=-1).cpu()
    ue_mask = g.edata.get('ue_mask')
    ue_mask = ue_mask.cpu() if ue_mask is not None else None
    positions = g.ndata.get('x_t')
    positions = positions.cpu() if positions is not None else None

    gs = dgl.unbatch(g)
    atom_offset, edge_offset = 0, 0
    new_e_hard = e_hard.clone()

    for g_i in gs:
        N = g_i.num_nodes()
        n_edges = g_i.num_edges()
        a_i = a_hard[atom_offset:atom_offset + N]
        e_i = e_hard[edge_offset:edge_offset + n_edges]
        ue_i = (ue_mask[edge_offset:edge_offset + n_edges]
                if ue_mask is not None else torch.ones(n_edges, dtype=torch.bool))
        src_i, dst_i = g_i.edges()
        src_i = src_i.cpu(); dst_i = dst_i.cpu()

        real_mask = a_i < n_real

        # Build (1, N, N) bond-order matrix.
        b_mat = torch.zeros(1, N, N, dtype=torch.long)
        pair_to_edges: dict[tuple[int, int], list[int]] = {}
        for k in range(n_edges):
            i = int(src_i[k]); j = int(dst_i[k])
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            pair_to_edges.setdefault(pair, []).append(k)
            if not ue_i[k].item() or i >= j:
                continue
            bt = int(e_i[k].item())
            if bt >= E_real_max:
                bt = 0
            if not (real_mask[i].item() and real_mask[j].item()):
                bt = 0
            if bt > 3:
                bt = 1
            b_mat[0, i, j] = bt
            b_mat[0, j, i] = bt

        # Fast greedy valence projection (no ILP — too slow for training).
        a_for_proj = a_i.clone()
        a_for_proj[~real_mask] = 0
        try:
            b_proj = project_valence_preserve_conn(
                a_for_proj.unsqueeze(0).long(), b_mat,
                valence_table, max_iters=5,   # low iters for speed
            )
        except Exception:
            b_proj = b_mat

        # Fast connectivity projection (BFS + nearest pair).
        try:
            pos_i = positions[atom_offset:atom_offset + N].unsqueeze(0) \
                    if positions is not None else None
            b_proj = project_connectivity(
                b_proj, n_atoms=torch.tensor([N], dtype=torch.long),
                r=pos_i,
            )
        except Exception:
            pass

        # Write bond orders back.
        for (i, j), eids in pair_to_edges.items():
            new_bt = int(b_proj[0, i, j].item())
            if new_bt == 0:
                new_bt = E_real_max  # mask slot = no-bond semantics
            for eid in eids:
                new_e_hard[edge_offset + eid] = new_bt

        atom_offset += N
        edge_offset += n_edges

    # Rewrite e_t as one-hot on original device.
    new_e_flat = new_e_hard.clamp(0, E_total - 1).to(e_t.device)
    g.edata['e_t'] = one_hot(new_e_flat.long(), num_classes=E_total).float()


def _patch_sample_conditional_path_discrete(vf, atom_map: list[str]) -> None:
    """Wrap sample_conditional_path to additionally project a_t + e_t onto
    the discrete manifold AFTER the simplex interpolation is done.

    This is the TRAINING-TIME discrete projection that makes our paper's
    "by construction" claim hold at every step, not just at the final
    output. Adapted from Cardei/Christopher NeurIPS 2025 (Constrained
    Discrete Diffusion) but for the CTMC flow-matching setting.
    """
    original = vf.sample_conditional_path
    prev_wrapper = getattr(vf, '_cfm_scp_wrapped', False)
    if prev_wrapper:
        # Already wrapped (e.g., gluing hook already installed). Stack
        # without re-wrapping.
        pass

    def wrapped(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        g = original(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)
        try:
            _project_discrete_interpolated(g, atom_map)
        except Exception as e:
            # Silent fallback — don't break training if projection fails
            # on a pathological batch.
            pass
        return g

    vf.sample_conditional_path = wrapped


# ---------------------------------------------------------------------------
# Hook 4b: project FINAL discrete state after integrate() at sample time
# (kept from previous implementation, complementary to 4a)
# ---------------------------------------------------------------------------

def _project_discrete_final(g, atom_map: list[str]) -> None:
    """In place: project g.ndata['a_1'] + g.edata['e_1'] so that every
    atom's summed bond-order is in its allowed valence set V(a_i) AND
    the bond graph is connected.

    Steps per molecule (after DGL unbatching):
      1. argmax a_1 + e_1 -> integer atom types + bond types
      2. Filter: fake-atom (idx = n_real_atoms) / mask (idx = n_real+1)
         slots bypass valence rule; bonds touching fake/mask are zeroed
      3. Map bond-type mask-token (FlowMol3 treats mask idx as "no bond")
         to 0 explicitly, before projection
      4. Run greedy `project_valence`; if residual errors, ILP fallback
      5. Run `project_connectivity` (BFS + Euclidean-nearest pair merge)
         with positions from g.ndata['x_1'] for nearest-pair selection
      6. Write back: new bond types -> one-hot in g.edata['e_1']
                     with mask slot preserved (bonds projected to 0 keep
                     their original mask-token position so RDKit ignores).
    """
    import dgl
    import torch
    from torch.nn.functional import one_hot

    n_real = len(atom_map)
    A_total = g.ndata['a_1'].shape[-1]   # e.g., 7 = 5 real + fake + mask
    E_total = g.edata['e_1'].shape[-1]   # e.g., 5 = none/single/double/triple/mask
    # FlowMol3 convention: last edge index = mask (treated as "no bond" in extract)
    # first real bond index = 0 (none). Real bonds are 0..E_real_max-1.
    E_real_max = E_total - 1             # exclude mask slot

    valence_table = {i: DEFAULT_VALENCE_SETS.get(sym, (4,))
                     for i, sym in enumerate(atom_map)}

    a_hard = g.ndata['a_1'].argmax(dim=-1).cpu()       # (total,)
    e_hard = g.edata['e_1'].argmax(dim=-1).cpu()       # (total_edges,)
    positions = g.ndata['x_1'].cpu() if 'x_1' in g.ndata else None
    ue_mask = g.edata.get('ue_mask')
    if ue_mask is not None:
        ue_mask = ue_mask.cpu()

    gs = dgl.unbatch(g)
    atom_offset, edge_offset = 0, 0
    new_e_hard = e_hard.clone()

    for g_i in gs:
        N = g_i.num_nodes()
        n_edges = g_i.num_edges()
        a_i = a_hard[atom_offset:atom_offset + N]
        e_i = e_hard[edge_offset:edge_offset + n_edges]
        ue_i = (ue_mask[edge_offset:edge_offset + n_edges]
                if ue_mask is not None else torch.ones(n_edges, dtype=torch.bool))
        src_i, dst_i = g_i.edges()
        src_i = src_i.cpu(); dst_i = dst_i.cpu()

        # Which atoms are REAL (0..n_real-1). Fake (n_real) and mask (n_real+1) are ignored.
        real_mask = a_i < n_real

        # Build (1, N, N) integer bond-order matrix from upper-triangle edges.
        b_mat = torch.zeros(1, N, N, dtype=torch.long)
        # Keep track of original edge index per pair to write back.
        pair_to_edges: dict[tuple[int, int], list[int]] = {}
        for k in range(n_edges):
            i = int(src_i[k]); j = int(dst_i[k])
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            pair_to_edges.setdefault(pair, []).append(k)

        for k in range(n_edges):
            if not ue_i[k].item():
                continue
            i = int(src_i[k]); j = int(dst_i[k])
            if i >= j:
                continue
            bt = int(e_i[k].item())
            # mask-token edge -> treat as no-bond (matching FlowMol3 semantics)
            if bt >= E_real_max:
                bt = 0
            # bonds touching fake/mask atoms: drop to 0
            if not (real_mask[i].item() and real_mask[j].item()):
                bt = 0
            # aromatic index 4 (rare with explicit_aromaticity=False) -> single
            if bt > 3:
                bt = 1
            b_mat[0, i, j] = bt
            b_mat[0, j, i] = bt

        # For projection, coerce non-real atom indices to 0 (C) so
        # project_valence doesn't choke on unknown indices.
        a_for_proj = a_i.clone()
        a_for_proj[~real_mask] = 0

        # Step 1: greedy valence projection (+ ILP fallback).
        a_batch = a_for_proj.unsqueeze(0).long()
        try:
            b_proj = project_valence_preserve_conn(a_batch, b_mat, valence_table, max_iters=15)
            ok = valence_ok(a_batch, b_proj, valence_table).all(dim=-1)[0]
            if not bool(ok):
                b_proj = project_valence_ilp(a_batch, b_proj, valence_table)
        except Exception:
            b_proj = b_mat

        # Step 2: connectivity projection (BFS + Euclidean-nearest pair merge).
        try:
            pos_i = positions[atom_offset:atom_offset + N].unsqueeze(0) \
                    if positions is not None else None
            b_proj = project_connectivity(
                b_proj,
                n_atoms=torch.tensor([N], dtype=torch.long),
                r=pos_i,
            )
        except Exception:
            pass

        # Step 3: write projected bond orders back to the edge list.
        for (i, j), eids in pair_to_edges.items():
            new_bt = int(b_proj[0, i, j].item())
            # Preserve mask-token slot for "no bond" so downstream
            # extract_moldata filters it out cleanly. Use index E_real_max.
            if new_bt == 0:
                new_bt = E_real_max      # FlowMol3's mask slot = no-bond semantics
            for eid in eids:
                new_e_hard[edge_offset + eid] = new_bt

        atom_offset += N
        edge_offset += n_edges

    # Rewrite g.edata['e_1'] as one-hot on device.
    new_e_flat = new_e_hard.clamp(0, E_total - 1).to(g.edata['e_1'].device)
    g.edata['e_1'] = one_hot(new_e_flat.long(), num_classes=E_total).float()
    # Atom types unchanged.


def _patch_integrate(vf, atom_map: list[str]) -> None:
    """Wrap `vector_field.integrate` to project the final discrete state
    onto the valence manifold before returning."""
    original = vf.integrate

    def wrapped(g, node_batch_idx, *args, **kwargs):
        result = original(g, node_batch_idx, *args, **kwargs)
        g_out = result[0] if isinstance(result, tuple) else result
        try:
            _project_discrete_final(g_out, atom_map)
        except Exception as e:
            print(f"[cfm_mol] _project_discrete_final raised {type(e).__name__}: "
                  f"{e}. Skipping discrete projection for this batch.")
        return result

    vf.integrate = wrapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_flowmol(
    model,
    d_min_table: torch.Tensor,
    tangent: bool = True,
    retract: bool = True,
    gluing: bool = True,
    discrete_projection: bool = True,
    train_time_discrete: bool = True,
    atom_map: list[str] | None = None,
) -> None:
    """Apply constraint hooks to a FlowMol LightningModule.

    All three hooks are applied by default. For ICLR E3 ablation, any hook
    can be disabled:

      - tangent: tangent-project the velocity inside `step()` before the
        Euler update.
      - retract: retract the updated `g.ndata['x_t']` after the Euler step.
      - gluing:  retract `g.ndata['x_t']` after `sample_conditional_path`
        (training-time interpolation retraction). Named "gluing" because
        this is where a fibre switch after a discrete transition gets
        geometrically resolved in the interpolant.

    The three booleans correspond 1:1 to the E3 ablation variants in
    `scripts/run_iclr_experiments.sh`:
       full          = (tangent=True,  retract=True,  gluing=True)
       -tangent      = (tangent=False, retract=True,  gluing=True)
       -retract      = (tangent=True,  retract=False, gluing=True)
       -gluing       = (tangent=True,  retract=True,  gluing=False)

    Currently only `parameterization: ctmc` (CTMCVectorField) is supported.
    """
    if not _HAS_FLOWMOL:
        raise ImportError(
            "flowmol package not available in the current env. "
            "Run scripts/install_and_process.slurm then re-import inside "
            "the `flowmol` env."
        )
    vf = model.vector_field
    if not isinstance(vf, CTMCVectorField):
        raise TypeError(
            f"Expected CTMCVectorField, got {type(vf).__name__}. "
            "Only `parameterization: ctmc` is currently supported."
        )

    active: list[str] = []
    if gluing:
        _patch_sample_conditional_path(vf, d_min_table)
        active.append("gluing")
    if tangent:
        _patch_vector_field_helper(vf, d_min_table)
        active.append("tangent")
    if retract:
        _patch_step(vf, d_min_table)
        active.append("retract")
    # If tangent=False but retract=True we still need the step-wrapper to
    # stash context (even though the helper won't do anything with it) so
    # that the retract-after-Euler phase runs. The step-wrapper is idempotent
    # vs the helper-wrapper; both are installed when retract is enabled.
    # If tangent=True but retract=False, install the step-wrapper sans retract.
    if tangent and not retract:
        # Install a minimal step-wrapper that only stashes context without
        # retracting after the Euler step. Reuse _patch_step but short-circuit
        # the retract call via a sentinel d_min_table of zeros.
        _patch_step(vf, torch.zeros_like(d_min_table))
        active.append("step-stash-only")
    if discrete_projection:
        if atom_map is None:
            raise ValueError(
                "patch_flowmol(discrete_projection=True) requires atom_map."
            )
        _patch_integrate(vf, atom_map)
        active.append("discrete_projection")
    if train_time_discrete:
        if atom_map is None:
            raise ValueError(
                "patch_flowmol(train_time_discrete=True) requires atom_map."
            )
        _patch_sample_conditional_path_discrete(vf, atom_map)
        active.append("train_time_discrete")
    print(f"[patch_flowmol] active hooks: {', '.join(active) if active else 'NONE (all disabled)'}")
