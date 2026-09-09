"""Stateful loader for the BGFM energy-consistency (Boltzmann) term.

Yields per-call: a batched DGL graph holding K perturbations of each of
B parent molecules (=> B*K virtual molecules), with the corresponding
OMol25 energies and parent ids ready for energy_consistency_loss_per_mol.

Lives outside Lightning's main DataLoader: it's iterated directly inside
the BGFM training hook every `energy_every_k_steps` steps. This avoids
forcing the main DataLoader to multiplex per-mol perturbations alongside
the regular per-mol stream (which complicates collation and breaks the
FM/force step's per-step batching invariant).

Storage convention (matching scripts/precompute_energy_perturbations.py):
  perturbation_*.pt stores K *consecutive* virtual molecules per parent;
  parent_id[i*K + k] == i for all k. Energies in eV.
"""
from __future__ import annotations

from pathlib import Path

import dgl
import torch
from torch.nn.functional import one_hot


class PerturbationLoader:
    def __init__(
        self,
        shard_paths,
        n_atom_types: int,
        n_extra_atom_classes: int = 0,
        n_charge_classes: int = 6,
        n_bond_types: int = 4,
        b_parents: int = 4,
        device: str = "cuda",
        seed: int = 0,
        max_atoms_per_parent: int | None = None,
        perturbation_indices: list[int] | None = None,
    ):
        if isinstance(shard_paths, (str, Path)):
            shard_paths = [shard_paths]
        # Load + concatenate all shards. Each shard's parent ids are local;
        # we re-base them so concatenated parent ids are globally unique.
        positions = []
        atom_types_idx = []
        atom_charges = []
        nia = []
        energies = []
        group_id = []
        K = None
        node_offset = 0
        parent_offset = 0
        for sp in shard_paths:
            d = torch.load(str(sp), map_location="cpu", weights_only=False)
            if K is None:
                K = int(d["K"])
            elif int(d["K"]) != K:
                raise ValueError(f"Mixed K across shards: {K} vs {d['K']}")
            positions.append(d["positions"])
            atom_types_idx.append(d["atom_types"].long())
            atom_charges.append(d["atom_charges"].long())
            nia.append(d["node_idx_array"] + node_offset)
            energies.append(d["energies"])
            group_id.append(d["group_id"] + parent_offset)
            node_offset += int(d["positions"].shape[0])
            parent_offset += int(d["group_id"].max().item()) + 1
        self.positions = torch.cat(positions, dim=0)
        self.atom_types_idx = torch.cat(atom_types_idx, dim=0)
        self.atom_charges = torch.cat(atom_charges, dim=0)
        self.nia = torch.cat(nia, dim=0)
        self.energies = torch.cat(energies, dim=0)
        self.group_id = torch.cat(group_id, dim=0)
        self.K = K
        self.perturbation_indices = list(range(K)) if perturbation_indices is None else list(perturbation_indices)
        if perturbation_indices is not None and (len(self.perturbation_indices)<2 or
                len(set(self.perturbation_indices))!=len(self.perturbation_indices) or
                any(not isinstance(i,int) or i<0 or i>=K for i in self.perturbation_indices)):
            raise ValueError('Select at least two distinct valid perturbation indices')
        self.M = int(self.group_id.max().item()) + 1

        self.b_parents = b_parents
        self.n_atom_types = n_atom_types
        self.n_extra_atom_classes = n_extra_atom_classes
        self.n_charge_classes = n_charge_classes
        self.n_bond_types = n_bond_types
        self.device = device

        # Optional: keep only parents whose first perturbation has <= N atoms.
        # The fully-connected perturbation graph is O(N^2) edges, and at
        # training time the FFJORD trajectory builds a backward graph with
        # create_graph=True over n_ode_steps -- so big parents OOM fast.
        # All K perturbations of a parent share atom count, so checking
        # perturbation k=0 is sufficient.
        if max_atoms_per_parent is not None:
            kept = []
            for p in range(self.M):
                ns = int(self.nia[p * K, 0])
                ne = int(self.nia[p * K, 1])
                if (ne - ns) <= max_atoms_per_parent:
                    kept.append(p)
            if not kept:
                raise ValueError(
                    f"PerturbationLoader: no parents with <= {max_atoms_per_parent} "
                    f"atoms in {len(shard_paths)} shard(s)."
                )
            self._eligible_parents = kept
            print(f"[bgfm] PerturbationLoader: filtered to {len(kept)}/{self.M} parents "
                  f"with <= {max_atoms_per_parent} atoms")
        else:
            self._eligible_parents = list(range(self.M))

        self._gen = torch.Generator().manual_seed(seed)
        self._ptr = 0
        self._order = self._fresh_order()

        # Sanity: K-contiguous-per-parent storage.
        if not torch.equal(self.group_id[:K], torch.zeros(K, dtype=torch.int64)):
            raise ValueError(
                "Perturbation shard does not follow K-contiguous-per-parent layout"
            )

    def _fresh_order(self):
        # Random permutation over the *eligible* parents only.
        idx = torch.randperm(len(self._eligible_parents), generator=self._gen)
        return [self._eligible_parents[int(i)] for i in idx]

    def _next_parent_ids(self):
        if self._ptr + self.b_parents > len(self._order):
            self._order = self._fresh_order()
            self._ptr = 0
        out = self._order[self._ptr:self._ptr + self.b_parents]
        self._ptr += self.b_parents
        return out

    def next_batch(self):
        """Return (g_pert, energies, parent_id, node_batch_idx, upper_edge_mask).

        - g_pert       : batched DGL graph of B*K virtual molecules, set up
                         for log_density_via_flow (x_1_true, x_t, a_t, c_t,
                         e_t all in place).
        - energies     : (B*K,) float32 in eV
        - parent_id    : (B*K,) int64 in 0..B-1 (local, for the loss to group)
        - node_batch_idx, upper_edge_mask: standard helpers (on device)
        """
        from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

        parent_indices = self._next_parent_ids()  # list of B global parent ids
        graphs = []
        energies_list = []
        pid_per_virtual = []  # 0..B-1
        K = self.K

        n_total_a = self.n_atom_types + self.n_extra_atom_classes
        for new_pid, par in enumerate(parent_indices):
            base = par * K
            for k in self.perturbation_indices:
                ns = int(self.nia[base + k, 0])
                ne = int(self.nia[base + k, 1])
                pos = self.positions[ns:ne]
                at_idx = self.atom_types_idx[ns:ne]
                ac = self.atom_charges[ns:ne]
                E = float(self.energies[base + k].item())

                n_atoms = pos.shape[0]
                # Fully-connected graph (bond-free; vector field handles all-to-all)
                src_grid, dst_grid = torch.meshgrid(
                    torch.arange(n_atoms), torch.arange(n_atoms), indexing="ij"
                )
                m = src_grid != dst_grid
                g = dgl.graph((src_grid[m], dst_grid[m]), num_nodes=n_atoms)

                a_oh = one_hot(at_idx, num_classes=n_total_a).float()
                c_oh = one_hot((ac + 2).long(), num_classes=self.n_charge_classes).float()
                g.ndata["x_1_true"] = pos.float()
                g.ndata["x_t"] = pos.float().clone()
                g.ndata["a_1_true"] = a_oh
                g.ndata["a_t"] = a_oh.clone()
                g.ndata["c_1_true"] = c_oh
                g.ndata["c_t"] = c_oh.clone()

                n_e = g.num_edges()
                e_oh = torch.zeros(n_e, self.n_bond_types)
                e_oh[:, 0] = 1.0   # bond-free: all-none class 0
                g.edata["e_1_true"] = e_oh
                g.edata["e_t"] = e_oh.clone()

                graphs.append(g)
                energies_list.append(E)
                pid_per_virtual.append(new_pid)

        g_batched = dgl.batch(graphs).to(self.device)
        energies_t = torch.tensor(energies_list, dtype=self.energies.dtype, device=self.device)
        parent_id_t = torch.tensor(pid_per_virtual, dtype=torch.long, device=self.device)
        node_batch_idx, _ = get_batch_idxs(g_batched)
        uem = get_upper_edge_mask(g_batched)
        # Stash the parent_indices on the loader so a downstream caller can
        # recover the (one-per-parent) invariant features for the anchor loss.
        self._last_parent_indices = parent_indices
        return g_batched, energies_t, parent_id_t, node_batch_idx, uem

    def parent_invariant_features(self, g_pert, parent_id):
        """For the most recently emitted batch, return ONE COPY of the
        invariant per-parent features needed by LogZPredictor:

          (parent_atom_type_idx, parent_node_batch_idx, parent_atom_charges)

        Each tensor has length = sum over parents of n_atoms_in_that_parent
        (NOT the M*K virtual molecules). parent_node_batch_idx maps atoms to
        their parent index in 0..B-1, matching the labels in parent_id.

        Implementation: read directly from the shard via the parent indices
        we cached in next_batch(). Avoids re-walking the batched virtual
        graph (whose discrete channels are one-hot anyway).
        """
        if not hasattr(self, "_last_parent_indices"):
            raise RuntimeError("Call next_batch() first.")
        parent_indices = self._last_parent_indices
        K = self.K
        at_idx_list = []
        ac_list = []
        nbi_list = []
        for new_pid, par in enumerate(parent_indices):
            # Use perturbation index 0 of each parent to get atom info
            # (all K perturbations of a parent share atom_types and atom_charges).
            ns = int(self.nia[par * K, 0])
            ne = int(self.nia[par * K, 1])
            n = ne - ns
            at_idx_list.append(self.atom_types_idx[ns:ne])
            ac_list.append(self.atom_charges[ns:ne])
            nbi_list.append(torch.full((n,), new_pid, dtype=torch.long))
        atom_type_idx = torch.cat(at_idx_list, dim=0).to(self.device)
        atom_charges = torch.cat(ac_list, dim=0).to(self.device)
        parent_nbi = torch.cat(nbi_list, dim=0).to(self.device)
        return atom_type_idx, parent_nbi, atom_charges
