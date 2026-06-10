"""Admissible manifold definitions (Appendix A Defs 1-4).

The admissible set is
    M = { (r, a, b) : (a, b) in M_conn, r in M_ster(a) }
    M_conn = { (a, b) : valence_ok(a, b) and graph(a, b) connected }
    M_ster(a) = { r : ||r_i - r_j|| > d_min(a_i, a_j) for all i != j }

All functions are batched over a leading batch dimension B and operate on
standard MiDi tensor conventions: r has shape (B, N, 3); a has shape (B, N)
with integer atom-type indices; b has shape (B, N, N) with integer bond-order
indices in {0=none, 1=single, 2=double, 3=triple, 4=aromatic}.
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Valence
# ---------------------------------------------------------------------------

# Default valence sets. Index by atom-type integer (MiDi convention: 0=H, ...).
# Extend when we start tmQM (transition metals) training in Week 6.
DEFAULT_VALENCE_SETS: dict[str, tuple[int, ...]] = {
    "H":  (1,),
    "C":  (4,),
    "N":  (3,),          # extend to (3, 5) later for nitro / N-oxide
    "O":  (2,),
    "F":  (1,),
    "P":  (3, 5),
    "S":  (2, 4, 6),
    "Cl": (1,),
    "Br": (1,),
    "I":  (1,),
    # Transition metals (tmQM benchmark). Start coarse; refine with literature.
    "Pd": (2, 4),
    "Ni": (2, 4),
    "Rh": (2, 3),
    "Ir": (3, 6),
}

# Bond-order integer to chemistry bond order (float).
# MiDi uses {0=none, 1=single, 2=double, 3=triple, 4=aromatic (treated as 1.5)}.
BOND_ORDER_VALUE: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 1.5)


def valence_ok(
    a: torch.LongTensor,
    b: torch.LongTensor,
    valence_table: dict[int, tuple[int, ...]],
) -> torch.BoolTensor:
    """Per-atom valence check.

    Args:
        a: (B, N) atom-type indices.
        b: (B, N, N) bond-order indices.
        valence_table: mapping {atom_type_index: allowed_valences_tuple}.

    Returns:
        (B, N) bool tensor; True where atom's summed bond order is in its valence set.
    """
    # Convert bond-order indices to floats, sum per-atom.
    bond_values = torch.tensor(BOND_ORDER_VALUE, device=b.device, dtype=torch.float)
    b_float = bond_values[b]                           # (B, N, N)
    valence = b_float.sum(dim=-1)                      # (B, N)

    # Check each atom against its allowed set.
    ok = torch.zeros_like(a, dtype=torch.bool)
    for atom_idx, allowed in valence_table.items():
        mask = a == atom_idx                           # (B, N)
        if not mask.any():
            continue
        atom_val = valence[mask]
        in_set = torch.zeros_like(atom_val, dtype=torch.bool)
        for v in allowed:
            in_set = in_set | (torch.isclose(atom_val, torch.full_like(atom_val, float(v))))
        ok[mask] = in_set
    return ok


# ---------------------------------------------------------------------------
# Steric exclusion
# ---------------------------------------------------------------------------

def steric_ok(
    r: torch.Tensor,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
) -> torch.BoolTensor:
    """Per-pair steric exclusion check.

    Args:
        r: (B, N, 3) coordinates.
        a: (B, N) atom-type indices.
        d_min_table: (A, A) lookup of d_min per atom-pair (A = # atom types).

    Returns:
        (B, N, N) bool tensor, True where pair distance > d_min(a_i, a_j).
        Diagonal (i == j) returned as True for convenience.
    """
    B, N, _ = r.shape
    diffs = r.unsqueeze(-2) - r.unsqueeze(-3)          # (B, N, N, 3)
    dists = diffs.norm(dim=-1)                         # (B, N, N)

    # lookup d_min per pair
    ai = a.unsqueeze(-1).expand(B, N, N)               # (B, N, N)
    aj = a.unsqueeze(-2).expand(B, N, N)
    d_min = d_min_table[ai, aj]                        # (B, N, N)

    ok = dists > d_min
    # diagonal trivially ok
    eye = torch.eye(N, dtype=torch.bool, device=r.device).unsqueeze(0).expand(B, N, N)
    ok = ok | eye
    return ok


def steric_all_ok(
    r: torch.Tensor,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
) -> torch.BoolTensor:
    """Reduce per-pair check to per-molecule bool. Shape (B,)."""
    return steric_ok(r, a, d_min_table).reshape(r.shape[0], -1).all(dim=-1)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def connectivity_ok(b: torch.LongTensor, n_atoms: torch.LongTensor) -> torch.BoolTensor:
    """Check whether the bond graph is a single connected component per molecule.

    Args:
        b: (B, N, N) bond-order indices (0 = no bond).
        n_atoms: (B,) actual atom counts per molecule (for padding).

    Returns:
        (B,) bool.
    """
    B, N, _ = b.shape
    out = torch.zeros(B, dtype=torch.bool, device=b.device)
    adj = (b > 0).to(torch.bool)                       # (B, N, N)
    for i in range(B):
        n = int(n_atoms[i].item())
        if n == 0:
            out[i] = True
            continue
        A = adj[i, :n, :n].to(torch.float)
        # BFS from atom 0.
        visited = torch.zeros(n, dtype=torch.bool, device=b.device)
        visited[0] = True
        frontier = torch.zeros(n, dtype=torch.bool, device=b.device)
        frontier[0] = True
        while frontier.any():
            new = (A @ frontier.float() > 0) & ~visited
            visited = visited | new
            frontier = new
        out[i] = visited.all()
    return out


# ---------------------------------------------------------------------------
# Default d_min table
# ---------------------------------------------------------------------------

def default_d_min_table(n_atom_types: int, scale: float = 0.7,
                         atom_map: list[str] | None = None) -> torch.Tensor:
    """Covalent-radius-based d_min(a_i, a_j) = scale * (r_cov(a_i) + r_cov(a_j)).

    If `atom_map` is provided, look up radii by element symbol (works for
    both QM9 atom_map [C, H, N, O, F] and tmQM broad atom_map including
    transition metals). Otherwise use FlowMol3 QM9 default ordering
    [C, H, N, O, F] and pad with 1.0 for unknown slots.

    TODO[Week 1]: run `scripts/calibrate_d_min.py` to verify 0% of
    training-set molecules violate the constraint at the chosen scale.
    """
    # Pyykkö single-bond covalent radii (Å), keyed by element.
    r_cov_by_sym: dict[str, float] = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
        "B": 0.84, "Si": 1.11, "As": 1.19, "Se": 1.20,
        # 3d transition metals.
        "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
        "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
        # 4d.
        "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47,
        "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
        # 5d.
        "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44,
        "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    }

    if atom_map is not None:
        r_cov = torch.tensor(
            [r_cov_by_sym.get(sym, 1.0) for sym in atom_map[:n_atom_types]],
            dtype=torch.float,
        )
    else:
        # FlowMol3 QM9 default order: [C, H, N, O, F].
        default_order = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        r_cov = torch.tensor(
            [r_cov_by_sym[sym] for sym in default_order[:max(10, n_atom_types)]],
            dtype=torch.float,
        )
        r_cov = torch.cat([r_cov, torch.full((max(0, n_atom_types - r_cov.numel()),), 1.0)])
        r_cov = r_cov[:n_atom_types]

    d_min = scale * (r_cov.unsqueeze(0) + r_cov.unsqueeze(1))
    return d_min
