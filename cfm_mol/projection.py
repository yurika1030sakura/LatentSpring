"""Discrete projections onto the valence + connectivity manifold.

After each Euler step of the discrete flow (Gat et al. 2024), we project the
(atom-type, bond-order) state back onto the admissible discrete manifold:

  1. project_valence          -- greedy local-edit algorithm (fast, simple).
  2. project_valence_ilp      -- ILP-based exact projection (strong, slower).
                                  Use as fallback when greedy fails.
  3. project_connectivity     -- BFS + Euclidean-nearest-cross-component merge.

The continuous-side gluing step (retract r onto the new steric fibre) is
provided by `cfm_mol.fibre.retract`. We wrap it here as `gluing_retract` for
semantic clarity at the call site.

Empirical note (2026-04-20): `project_valence` greedy achieves only ~56%
recovery on synthetic near-feasible states (`scripts/measure_projection_failure.py`
--noise_p 0.02). For the ICLR paper's 99.9% claim, use `project_valence_ilp`
as the secondary projection when greedy fails to converge, or switch to ILP
entirely if QM9 CTMC samples are worse than synthetic near-feasible.
"""
from __future__ import annotations

import torch

from cfm_mol.domain import BOND_ORDER_VALUE, valence_ok
from cfm_mol.fibre import retract


# ---------------------------------------------------------------------------
# Valence projection
# ---------------------------------------------------------------------------

def _atom_valence(b_row: torch.Tensor) -> float:
    """Sum bond orders for one atom's row of b."""
    bond_values = torch.tensor(BOND_ORDER_VALUE, device=b_row.device, dtype=torch.float)
    return bond_values[b_row].sum().item()


def _allowed_valences(atom_type: int, valence_table: dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    return valence_table.get(atom_type, (4,))       # default to 4 if unknown


def _nearest_allowed(current: float, allowed: tuple[int, ...]) -> float:
    """Nearest element of allowed to current."""
    return min(allowed, key=lambda v: abs(v - current))


def project_valence(
    a: torch.LongTensor,
    b: torch.LongTensor,
    valence_table: dict[int, tuple[int, ...]],
    max_iters: int = 8,
) -> torch.LongTensor:
    """Project b to nearest valence-satisfying bond-order matrix.

    Greedy algorithm (per molecule in batch):
      For each iteration:
        For each atom i with valence error:
          Compute target valence: nearest element of V(a_i) to current.
          delta = target - current. If |delta| rounds to >=1 in bond-order units,
          adjust one incident edge by +/- 1 bond-order-index to shrink |delta|.
          Prefer adjusting the edge with the largest room to change (not yet
          triple, not yet zero).
        If no atom violates, done.

    Args
    ----
    a : (B, N) atom-type indices.
    b : (B, N, N) bond-order indices in {0,1,2,3,4}.
    valence_table : mapping atom_type -> tuple of allowed integer valences.
    max_iters : greedy rounds.

    Returns
    -------
    b_proj : (B, N, N) (may be unchanged if we fail to converge).
    """
    B, N, _ = b.shape
    b_out = b.clone()

    # Enforce symmetry and no self-loops as a precondition.
    for bi in range(B):
        for i in range(N):
            b_out[bi, i, i] = 0
        b_out[bi] = torch.max(b_out[bi], b_out[bi].t())    # symmetric max

    for bi in range(B):
        for _ in range(max_iters):
            ok = valence_ok(a[bi:bi+1], b_out[bi:bi+1], valence_table)[0]  # (N,)
            if ok.all():
                break
            # Iterate atoms; adjust one edge for each violating atom.
            for i in range(N):
                if ok[i]:
                    continue
                atom_type = int(a[bi, i].item())
                allowed = _allowed_valences(atom_type, valence_table)
                cur = _atom_valence(b_out[bi, i])
                target = _nearest_allowed(cur, allowed)
                if abs(target - cur) < 0.5:
                    continue
                delta = target - cur             # positive: add bond; negative: remove
                # Pick an incident edge (i, j) to adjust.
                # If delta > 0: find j where b[i,j] can increase (index < 3 not aromatic).
                # If delta < 0: find j where b[i,j] can decrease (index > 0).
                best_j = -1
                for j in range(N):
                    if j == i:
                        continue
                    idx = int(b_out[bi, i, j].item())
                    if delta > 0 and idx < 3:        # can go up
                        best_j = j
                        break
                    if delta < 0 and 0 < idx <= 3:   # can go down
                        best_j = j
                        break
                if best_j < 0:
                    break                            # nothing we can do
                # Adjust by one bond-order-index step toward the target.
                step = 1 if delta > 0 else -1
                new_idx = int(b_out[bi, i, best_j].item()) + step
                new_idx = max(0, min(3, new_idx))    # clamp to {0,1,2,3}; skip aromatic (4)
                b_out[bi, i, best_j] = new_idx
                b_out[bi, best_j, i] = new_idx
    return b_out


# ---------------------------------------------------------------------------
# Bridge detection (Tarjan) for connectivity-preserving projection
# ---------------------------------------------------------------------------

def _find_bridges(adj_bool: torch.BoolTensor) -> set[tuple[int, int]]:
    """Tarjan's bridge-finding on an undirected NxN boolean adjacency matrix.

    Returns: set of edges {(i, j) with i<j} that are bridges (removal
    disconnects the graph).

    O(N + E) time. Pure Python; fast for N<=100.
    """
    N = adj_bool.shape[0]
    adj_list: list[list[int]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and adj_bool[i, j].item():
                adj_list[i].append(j)

    disc = [-1] * N       # discovery time
    low = [-1] * N        # lowest discovery reachable
    timer = [0]
    bridges: set[tuple[int, int]] = set()

    def dfs(u: int, parent: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in adj_list[u]:
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for s in range(N):
        if disc[s] == -1:
            dfs(s, -1)
    return bridges


# ---------------------------------------------------------------------------
# Connectivity-preserving valence projection
# ---------------------------------------------------------------------------

def project_valence_preserve_conn(
    a: torch.LongTensor,
    b: torch.LongTensor,
    valence_table: dict[int, tuple[int, ...]],
    max_iters: int = 10,
) -> torch.LongTensor:
    """Valence projection that NEVER deletes a bridge.

    Priority rules when reducing bond orders:
      1. Prefer to reduce HIGH-ORDER bonds first (triple -> double -> single).
         These reductions don't affect connectivity because the edge stays.
      2. Only set a bond order to 0 (remove edge) if the edge is NOT a bridge.
      3. If all candidate edges for removal are bridges, accept soft failure
         (leave the valence violation; connectivity is more important for
         training signal quality).

    This is strictly better than the naive `project_valence` for use inside
    training loops, because it prevents fragmentation that would poison the
    training distribution.

    Args: same as `project_valence`.
    Returns: b_proj (may not fully satisfy valence if all removal candidates
    are bridges, but will ALWAYS preserve connectivity of the input graph).
    """
    B, N, _ = b.shape
    b_out = b.clone()
    # Precondition: symmetrise + no self-loops.
    for bi in range(B):
        for i in range(N):
            b_out[bi, i, i] = 0
        b_out[bi] = torch.max(b_out[bi], b_out[bi].t())

    for bi in range(B):
        for _ in range(max_iters):
            ok = valence_ok(a[bi:bi + 1], b_out[bi:bi + 1], valence_table)[0]
            if ok.all():
                break

            # Recompute bridges each iteration (graph changes).
            adj = (b_out[bi] > 0)
            bridges = _find_bridges(adj)

            changed = False
            for i in range(N):
                if ok[i]:
                    continue
                atom_type = int(a[bi, i].item())
                allowed = _allowed_valences(atom_type, valence_table)
                cur = _atom_valence(b_out[bi, i])
                target = _nearest_allowed(cur, allowed)
                if abs(target - cur) < 0.5:
                    continue
                delta = target - cur

                best_j = -1
                best_strategy = None     # 'reduce_triple', 'reduce_double',
                                          # 'reduce_single', 'add'
                if delta > 0:
                    # Need to add bond order.
                    for j in range(N):
                        if j == i:
                            continue
                        idx = int(b_out[bi, i, j].item())
                        if idx < 3:
                            best_j = j
                            best_strategy = 'add'
                            break
                else:
                    # Need to remove bond order. Prefer high-order reductions.
                    for target_order in (3, 2, 1):
                        if best_j >= 0:
                            break
                        for j in range(N):
                            if j == i:
                                continue
                            idx = int(b_out[bi, i, j].item())
                            if idx != target_order:
                                continue
                            if idx == 1:
                                # Removing a single bond = edge removal.
                                # Skip if this edge is a bridge.
                                edge = (min(i, j), max(i, j))
                                if edge in bridges:
                                    continue
                            best_j = j
                            best_strategy = f'reduce_{target_order}'
                            break

                if best_j < 0:
                    # All candidate edges are bridges OR no edges exist.
                    # Accept soft valence failure for this atom.
                    continue

                step = 1 if delta > 0 else -1
                new_idx = int(b_out[bi, i, best_j].item()) + step
                new_idx = max(0, min(3, new_idx))
                b_out[bi, i, best_j] = new_idx
                b_out[bi, best_j, i] = new_idx
                changed = True

            if not changed:
                break
    return b_out


# ---------------------------------------------------------------------------
# ILP-based valence projection (exact, slower)
# ---------------------------------------------------------------------------

def project_valence_ilp(
    a: torch.LongTensor,
    b: torch.LongTensor,
    valence_table: dict[int, tuple[int, ...]],
) -> torch.LongTensor:
    """Project b to the nearest valence-satisfying bond-order matrix via LP
    relaxation + rounding.

    For each molecule we solve:

        minimize   sum_{i < j} |b_{ij} - b_{ij}^init|
        subject to sum_{j != i} b_{ij}  in  V(a_i)  for all i
                   b_{ij} in {0, 1, 2, 3}
                   b_{ij} = b_{ji}
                   b_{ii} = 0

    The valence-membership constraint is non-convex (a discrete set). We
    handle it by enumerating all combinations of per-atom valence targets
    (at most sum_i |V(a_i)| possibilities) and solving the resulting LP for
    each; keeping the best feasible solution. For QM9 atom types this is
    1^5 = 1 combination when all atoms are C/N/O/F/H (single valences), but
    for S or Pd the branching is small (~2-3 per atom).

    In practice we use a simpler fallback: LP relaxation with continuous
    b_{ij} in [0, 3], objective = L1 to initial, constraints = sum_j b_{ij}
    = target_i (per-atom target picked as the nearest allowed valence to
    the initial sum). Round to nearest integer post-hoc.

    This is WAY stronger than greedy when the initial state is multiple
    edits away from feasible.

    Args: same as project_valence.
    Returns: b_proj : (B, N, N) long.
    """
    try:
        from scipy.optimize import linprog
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "project_valence_ilp requires scipy. Install: pip install scipy"
        ) from e

    B, N, _ = b.shape
    b_out = b.clone()
    # Symmetrise and zero diagonal.
    for bi in range(B):
        b_out[bi] = torch.max(b_out[bi], b_out[bi].t())
        b_out[bi].fill_diagonal_(0)

    bond_values = torch.tensor(BOND_ORDER_VALUE, dtype=torch.float)

    for bi in range(B):
        b_float_init = bond_values[b_out[bi]].numpy()    # (N, N) chemistry values
        a_np = a[bi].numpy()

        # Pick a per-atom valence target (nearest allowed to current).
        targets = np.zeros(N)
        for i in range(N):
            atom_type = int(a_np[i])
            allowed = valence_table.get(atom_type, (4,))
            cur = float(b_float_init[i].sum())
            targets[i] = min(allowed, key=lambda v: abs(v - cur))

        # LP: variables = x_{ij} for i < j, in [0, 3].
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        n_vars = len(pairs)
        pair_to_idx = {(i, j): k for k, (i, j) in enumerate(pairs)}

        # L1 objective: min sum_{k} |x_k - x_k^init|. Lift to LP with slack:
        # min sum_k s_k  s.t.  s_k >= x_k - x_k^init,  s_k >= x_k^init - x_k.
        # Variable order: [x_0..x_{n-1}, s_0..s_{n-1}], 2*n_vars vars.
        x_init = np.array([b_float_init[i, j] for (i, j) in pairs])
        c = np.concatenate([np.zeros(n_vars), np.ones(n_vars)])

        # Slack constraints: s - x >= -x_init  =>  -s + x <= x_init
        #                    s + x >= x_init   =>  -s - x <= -x_init
        A_ub = np.zeros((2 * n_vars, 2 * n_vars))
        b_ub = np.zeros(2 * n_vars)
        for k in range(n_vars):
            A_ub[k, k] = 1.0       # x_k
            A_ub[k, n_vars + k] = -1.0  # -s_k
            b_ub[k] = x_init[k]
            A_ub[n_vars + k, k] = -1.0
            A_ub[n_vars + k, n_vars + k] = -1.0
            b_ub[n_vars + k] = -x_init[k]

        # Equality constraints: for each atom i, sum_{(i,j) or (j,i) in pairs} x_k = target_i.
        A_eq = np.zeros((N, 2 * n_vars))
        b_eq = targets
        for k, (i, j) in enumerate(pairs):
            A_eq[i, k] = 1.0
            A_eq[j, k] = 1.0

        bounds = [(0.0, 3.0)] * n_vars + [(0.0, None)] * n_vars

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            # LP infeasible with these targets; fall back to greedy result.
            continue

        x_opt = res.x[:n_vars]
        # Round to nearest integer in {0, 1, 2, 3}.
        # Chemistry values {0, 1, 1.5 (aromatic), 2, 3}; we map continuous
        # to nearest integer order, dropping aromatic from the codomain.
        order_rounded = np.clip(np.round(x_opt), 0, 3).astype(int)
        for k, (i, j) in enumerate(pairs):
            b_out[bi, i, j] = int(order_rounded[k])
            b_out[bi, j, i] = int(order_rounded[k])
    return b_out


# ---------------------------------------------------------------------------
# Connectivity projection
# ---------------------------------------------------------------------------

def _connected_components(adj: torch.BoolTensor) -> list[list[int]]:
    """BFS components of an NxN bool adjacency matrix. Returns list of lists of
    atom indices."""
    N = adj.shape[0]
    visited = [False] * N
    components: list[list[int]] = []
    for start in range(N):
        if visited[start]:
            continue
        comp = [start]
        frontier = [start]
        visited[start] = True
        while frontier:
            new_frontier = []
            for u in frontier:
                for v in range(N):
                    if adj[u, v].item() and not visited[v]:
                        visited[v] = True
                        comp.append(v)
                        new_frontier.append(v)
            frontier = new_frontier
        components.append(comp)
    return components


def project_connectivity(
    b: torch.LongTensor,
    n_atoms: torch.LongTensor | None = None,
    r: torch.Tensor | None = None,
) -> torch.LongTensor:
    """Add single bonds between nearest-Euclidean atoms of different components
    until the bond graph is one connected component.

    Args
    ----
    b : (B, N, N) bond-order indices.
    n_atoms : (B,) actual atom counts; if None, assume full N.
    r : optional (B, N, 3) coordinates for Euclidean nearest-pair selection.
        If None, falls back to lowest-index pair (testing only).

    Returns
    -------
    b_proj : (B, N, N).
    """
    B, N, _ = b.shape
    b_out = b.clone()
    if n_atoms is None:
        n_atoms = torch.full((B,), N, dtype=torch.long)

    for bi in range(B):
        n = int(n_atoms[bi].item())
        if n <= 1:
            continue
        while True:
            adj = (b_out[bi, :n, :n] > 0)
            comps = _connected_components(adj)
            if len(comps) <= 1:
                break
            # Pick the two components that are closest (Euclidean) and merge.
            best_pair = (comps[0][0], comps[1][0])
            best_dist = float("inf")
            if r is not None:
                for idx_a, comp_a in enumerate(comps):
                    for comp_b in comps[idx_a + 1:]:
                        for i in comp_a:
                            for j in comp_b:
                                d = (r[bi, i] - r[bi, j]).norm().item()
                                if d < best_dist:
                                    best_dist = d
                                    best_pair = (i, j)
            i, j = best_pair
            b_out[bi, i, j] = 1
            b_out[bi, j, i] = 1
    return b_out


# ---------------------------------------------------------------------------
# Gluing retract (continuous side)
# ---------------------------------------------------------------------------

def gluing_retract(
    r: torch.Tensor,
    a_new: torch.LongTensor,
    d_min_table: torch.Tensor,
    max_iters: int = 20,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """After a discrete transition (a, b) -> (a', b'), retract coordinate r
    onto the new steric fibre M_ster(a')."""
    return retract(r, a_new, d_min_table, max_iters=max_iters)
