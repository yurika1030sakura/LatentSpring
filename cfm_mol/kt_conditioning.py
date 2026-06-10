"""Temperature (kT) conditioning patch for FlowMol3 vector_field.

Standard BGFM is trained at a fixed kT. This module adds runtime kT
conditioning: the model receives kT as an input and learns a single
density family p_θ(x | atom_types, kT) that produces Boltzmann-shaped
samples at any user-specified kT at inference.

Architecture choice:
  Rather than retrain from scratch, we monkey-patch the vector_field
  forward to add a RESIDUAL kT signal to node scalar features after the
  existing scalar_embedding. The kT projection is zero-initialized at the
  final layer so the patched model behaves IDENTICALLY to the unpatched
  model at init -- safe to resume from any pretrained checkpoint.

  kT_proj(log kT) -> (n_hidden_scalars,) -> broadcast to atoms -> add.

Why log kT: training samples kT log-uniformly in [0.025, 1.0] eV. Log
makes the embedding linear in inverse temperature β = 1/kT (the
physically natural variable).

Usage:
  # at model construction (e.g. in patch_flowmol)
  patch_kT_conditioning(model.vector_field, n_hidden_scalars=256)

  # at training step
  kT_tensor = sample_kT(B, device, kT_min=0.025, kT_max=1.0)  # (B,)
  vf_out = model.vector_field(g, t, node_batch_idx=..., upper_edge_mask=..., kT=kT_tensor)

  # at inference
  kT = torch.full((B,), 0.025, device=device)  # T=300K
  v = model.vector_field(g, t, ..., kT=kT)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def sample_kT(batch_size: int, device, kT_min: float = 0.025,
              kT_max: float = 1.0) -> torch.Tensor:
    """Log-uniform sample of kT per molecule in batch.

    Args:
        kT_min: min temperature (e.g. 0.025 eV = T=300K)
        kT_max: max temperature (e.g. 1.0 eV = T=11600K)

    Returns:
        (batch_size,) tensor in eV
    """
    log_min = math.log(kT_min)
    log_max = math.log(kT_max)
    u = torch.rand(batch_size, device=device)
    return torch.exp(log_min + u * (log_max - log_min))


class _kTProjection(nn.Module):
    """log(kT) -> per-atom residual feature of dim n_hidden_scalars.

    Final layer zero-initialized so output is 0 at init -- model behaves
    identically to non-conditional FlowMol3 at start of training.
    """
    def __init__(self, n_hidden_scalars: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_hidden_scalars),
        )
        # Zero-init final layer so kT signal starts at 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, kT: torch.Tensor) -> torch.Tensor:
        # log kT for natural inverse-temperature scaling
        return self.net(kT.log().unsqueeze(-1))


def patch_kT_conditioning(vector_field, n_hidden_scalars: int):
    """Monkey-patch vector_field.forward to apply residual kT conditioning.

    After patching:
      - vector_field.forward accepts kwarg `kT` (shape (B,) in eV)
      - If `kT` is provided, a residual kT signal is added to node scalar
        features after the original scalar_embedding
      - If `kT` is None, behavior is identical to unpatched model

    Args:
        vector_field: the model.vector_field instance (nn.Module)
        n_hidden_scalars: must match vector_field.scalar_embedding output dim
    """
    if hasattr(vector_field, "_kT_projection"):
        print("[kT_conditioning] already patched, skip")
        return

    # Add the projection module
    vector_field._kT_projection = _kTProjection(n_hidden_scalars).to(
        next(vector_field.parameters()).device)
    vector_field._kT_conditioning_enabled = True

    # Save the original forward
    original_forward = vector_field.forward

    def patched_forward(self, *args, **kwargs):
        # Extract kT kwarg if present
        kT = kwargs.pop("kT", None)

        # If no kT provided, behave exactly as original
        if kT is None or not getattr(self, "_kT_conditioning_enabled", False):
            return original_forward(*args, **kwargs)

        # We can't easily intercept the inner forward without rewriting,
        # so use a graph-side stash: store kT on the graph, the inner
        # forward picks it up via a hook attached below.
        # ASSUMPTION: first positional arg is the graph g.
        if args:
            g = args[0]
            # Stash on the graph as a per-graph attribute (broadcast lazily)
            g._bgfm_kT = kT
        # Call original; the post-hook on scalar_embedding adds residual.
        result = original_forward(*args, **kwargs)
        # Clean stash so it doesn't leak
        if args:
            args[0]._bgfm_kT = None
        return result

    import types
    vector_field.forward = types.MethodType(patched_forward, vector_field)

    # Attach a forward post-hook on the scalar_embedding that adds the
    # kT residual to its output. This is the simplest way to intercept
    # without modifying the inner forward.
    def scalar_emb_post_hook(module, inputs, output):
        # The graph is in the local-scope of the vector_field forward, so
        # we get it from the patched_forward's stash via the vector_field
        # itself. But we don't have direct access here -- use the alt
        # approach: store a registration hook on vector_field that captures
        # current node_batch_idx + kT during forward.
        nonlocal_state = getattr(vector_field, "_kT_runtime_state", None)
        if nonlocal_state is None or nonlocal_state.get("kT") is None:
            return output
        kT = nonlocal_state["kT"]
        node_batch_idx = nonlocal_state["node_batch_idx"]
        if kT is None or node_batch_idx is None:
            return output
        kT_emb = vector_field._kT_projection(kT)        # (B, n_hidden_scalars)
        return output + kT_emb[node_batch_idx]          # broadcast to atoms

    vector_field.scalar_embedding.register_forward_hook(scalar_emb_post_hook)

    # Re-patch forward to set runtime state instead of stashing on graph
    def patched_forward_v2(self, *args, **kwargs):
        kT = kwargs.pop("kT", None)
        node_batch_idx = kwargs.get("node_batch_idx", None)
        # Set runtime state for the hook
        self._kT_runtime_state = {"kT": kT, "node_batch_idx": node_batch_idx}
        try:
            result = original_forward(*args, **kwargs)
        finally:
            self._kT_runtime_state = None
        return result

    vector_field.forward = types.MethodType(patched_forward_v2, vector_field)

    n_params = sum(p.numel() for p in vector_field._kT_projection.parameters())
    print(f"[kT_conditioning] patched vector_field: {n_params} kT-projection params, zero-init so initially no-op")
