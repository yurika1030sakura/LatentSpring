"""Mass-preserving minima endpoints for the existing composition-only FM.

This module does NOT produce a thermal teacher or certify one. It consumes a
frozen, graph-global particle measure supplied by a separately audited teacher.
Optimization changes coordinates, never particle weights. Graph identity is
sampling/provenance metadata, not a feature supplied to the student.

Files are JSON + numeric NPZ (allow_pickle=False). A failed positive-mass
optimization blocks training by default; conditioning within each graph requires
an explicit opt-in and is a different target. Unknown weights (NaN) always fail.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import numpy as np

FORMAT = "latentspring_weighted_endpoints_v1"
KB_EV_PER_K = 8.617333262145e-5


def composition_key(numbers: tuple[int, ...], charge: int, spin: int) -> str:
    """Permutation-invariant composition/electronic-state key, NOT a graph key."""
    payload = json.dumps([sorted(numbers), charge, spin], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def logsumexp(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=np.float64)
    m = float(np.max(a))
    if not math.isfinite(m):
        if m == -math.inf:
            return m
        raise ValueError("Nonfinite log weight")
    return m + math.log(float(np.exp(a - m).sum()))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class EndpointGroup:
    """One fixed graph, atom ordering, electronic state and thermal target.

    ``mixture_mass`` is an explicitly chosen outer prior (composition x p_chem
    x temperature), NOT a graph partition function inferred from log weights.
    ``success`` concerns the declared optimization/graph-preservation mapping.
    Unsupported thermal proposals have log_weight=-inf; unknown energies must
    NOT be mislabeled -inf. ``basin_id`` is the declared optimization label.
    """
    group_id: str
    graph_id: str
    numbers: tuple[int, ...]
    charge: int
    spin_multiplicity: int
    temperature_K: float
    mixture_mass: float
    minimum_positions: np.ndarray
    log_weight: np.ndarray
    success: np.ndarray
    basin_id: tuple[str, ...]
    particle_id: tuple[str, ...]
    target: dict[str, Any]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        supplied = np.asarray(self.numbers)
        if (supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.number)
                or not np.isfinite(supplied).all() or np.any(supplied != np.round(supplied))):
            raise ValueError("Atomic numbers must be finite integers")
        self.numbers = tuple(int(z) for z in supplied)
        if not 2 <= len(self.numbers) <= 200 or not all(1 <= z <= 118 for z in self.numbers):
            raise ValueError("Require 2--200 atoms and valid atomic numbers")
        if (type(self.charge) is not int or type(self.spin_multiplicity) is not int
                or self.spin_multiplicity < 1):
            raise ValueError("Charge and spin multiplicity must be integers")
        electrons = sum(self.numbers) - self.charge
        if electrons < self.spin_multiplicity - 1 or (electrons - self.spin_multiplicity + 1) % 2:
            raise ValueError("Electron count and multiplicity are incompatible")
        if not self.group_id or not self.graph_id:
            raise ValueError("Explicit group and graph identifiers are required")
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0:
            raise ValueError("Temperature must be positive and finite")
        if not math.isfinite(self.mixture_mass) or self.mixture_mass <= 0:
            raise ValueError("Outer mixture mass must be positive and finite")
        self.minimum_positions = np.array(self.minimum_positions, dtype=np.float64, copy=True)
        self.log_weight = np.array(self.log_weight, dtype=np.float64, copy=True)
        status = np.asarray(self.success)
        if status.dtype != np.bool_:
            raise ValueError("success must be a boolean array, not arbitrary status integers")
        self.success = status.copy()
        m = len(self.log_weight)
        if (m == 0 or self.log_weight.shape != (m,) or self.success.shape != (m,)
                or self.minimum_positions.shape != (m, len(self.numbers), 3)):
            raise ValueError("Invalid endpoint array shapes")
        if (np.isnan(self.log_weight).any() or np.isposinf(self.log_weight).any()
                or not np.isfinite(self.log_weight).any()):
            raise ValueError("Unknown, infinite, or all-zero thermal weights")
        if not np.isfinite(self.minimum_positions[self.success]).all():
            raise ValueError("Successful minima must have finite coordinates")
        if len(self.basin_id) != m or len(self.particle_id) != m or len(set(self.particle_id)) != m:
            raise ValueError("Every particle needs a unique ID and a basin/status label")
        if any(not self.basin_id[i] or self.basin_id[i].startswith("__") for i in np.flatnonzero(self.success)):
            raise ValueError("Successful particles need nonempty basin labels")
        required = {"oracle_id", "base_measure", "support", "optimizer", "weight_scope", "split"}
        if not required.issubset(self.target) or any(not self.target[k] for k in required):
            raise ValueError("Missing target declaration")
        if self.target["weight_scope"] != "graph_global":
            raise ValueError("Per-anchor-normalized local weights are not graph-global weights")
        if self.target["split"] != "train":
            raise ValueError("Endpoint store accepts training targets only")
        # Freeze metadata into a JSON-serializable copy; do not retain mutable aliases.
        self.target = json.loads(json.dumps(self.target, allow_nan=False))
        self.provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        for a in (self.minimum_positions, self.log_weight, self.success):
            a.setflags(write=False)

    @property
    def composition_id(self) -> str:
        return composition_key(self.numbers, self.charge, self.spin_multiplicity)

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_weight - logsumexp(self.log_weight))

    @property
    def failed_mass(self) -> float:
        return float(self.weights[~self.success].sum())

    def masses(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, w in enumerate(self.weights):
            key = self.basin_id[i] if self.success[i] else "__FAILED__"
            out[key] = out.get(key, 0.0) + float(w)
        return out

    def condition(self) -> dict[str, Any]:
        # Explicit whitelist: no graph ID, basin, bonds or target coordinates.
        return dict(atomic_numbers=list(self.numbers), numbers=list(self.numbers),
                    n_atoms=len(self.numbers), charge=self.charge,
                    spin_multiplicity=self.spin_multiplicity,
                    requested_kT_eV=KB_EV_PER_K * self.temperature_K,
                    temperature_K=self.temperature_K, composition_hex=self.composition_id)

    def aggregate_identical_minima(self, tolerance_A: float = 1e-8) -> "EndpointGroup":
        """Merge explicit basin labels ONLY if representative coordinates agree.

        No automatic atom matching or chemical deduplication is guessed here.
        Failed rows retain individual records. Do this before endpoint smoothing.
        """
        if not math.isfinite(tolerance_A) or tolerance_A < 0:
            raise ValueError("Invalid representative tolerance")
        indices: dict[tuple, list[int]] = {}
        for i in range(len(self.log_weight)):
            key = ("basin", self.basin_id[i]) if self.success[i] else ("failure", i)
            indices.setdefault(key, []).append(i)
        xyz, lw, success, basins, ids, lineage = [], [], [], [], [], []
        for k, rows in indices.items():
            first = rows[0]
            ref = self.minimum_positions[first]
            if self.success[first] and any(
                    not np.allclose(self.minimum_positions[j], ref, atol=tolerance_A, rtol=0)
                    for j in rows):
                raise ValueError("Same basin label has different aligned representatives; resolve upstream")
            xyz.append(ref)
            lw.append(logsumexp(self.log_weight[rows]))
            success.append(bool(self.success[first])); basins.append(self.basin_id[first])
            ids.append(f"merged-{len(ids)}")
            lineage.append([self.particle_id[j] for j in rows])
        return EndpointGroup(self.group_id, self.graph_id, self.numbers, self.charge,
            self.spin_multiplicity, self.temperature_K, self.mixture_mass,
            np.asarray(xyz), np.asarray(lw), np.asarray(success, dtype=bool), tuple(basins), tuple(ids),
            self.target, {**self.provenance, "merged_particle_ids": lineage,
                          "aggregation": "sum original thermal mass; no energy reweighting"})


@dataclass
class EndpointDraw:
    group_id: str
    particle_id: str
    basin_id: str
    condition: dict[str, Any]
    positions: np.ndarray


class WeightedEndpointStore:
    """Precomputed graph-global CDFs; no minibatch self-normalization."""
    def __init__(self, groups: list[EndpointGroup], *, failure_policy: str = "error"):
        if not groups or len({g.group_id for g in groups}) != len(groups):
            raise ValueError("Nonempty uniquely identified groups are required")
        if failure_policy not in {"error", "condition_within_group"}:
            raise ValueError("Unknown failure policy")
        total = sum(g.mixture_mass for g in groups)
        if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Declared outer mixture masses must sum to one")
        identities = {(g.composition_id, g.graph_id, g.temperature_K) for g in groups}
        if len(identities) != len(groups):
            raise ValueError("Duplicate graph/temperature groups: consolidate graph-global teacher weights first")
        self.groups, self.failure_policy = groups, failure_policy
        self._outer_cdf = np.cumsum([g.mixture_mass for g in groups]); self._outer_cdf[-1] = 1.
        self._cdf: list[np.ndarray] = []
        for g in groups:
            if failure_policy == "error" and np.any((~g.success) & np.isfinite(g.log_weight)):
                raise ValueError(f"{g.group_id}: failed positive thermal mass {g.failed_mass:.9g}; repair or explicitly condition")
            # Success mass needs to exist even if all failed weights underflow.
            supported = g.success & np.isfinite(g.log_weight)
            if not np.any(supported):
                raise ValueError(f"{g.group_id}: no successful positive-mass endpoints")
            lw = np.where(g.success, g.log_weight, -np.inf)
            cdf = np.cumsum(np.exp(lw - logsumexp(lw))); cdf[-1] = 1.
            self._cdf.append(cdf)

    @property
    def failed_mass(self) -> float:
        return sum(g.mixture_mass * g.failed_mass for g in self.groups)

    def draw(self, n: int, rng: np.random.Generator, *, sigma_A: float = 0.) -> list[EndpointDraw]:
        if not isinstance(n, int) or n < 1 or not math.isfinite(sigma_A) or sigma_A < 0:
            raise ValueError("Require positive draw count and nonnegative finite smoothing width")
        group_indices = np.searchsorted(self._outer_cdf, rng.random(n), side="right")
        u = rng.random(n)
        out = []
        for k, gi in enumerate(group_indices):
            g = self.groups[int(gi)]
            j = int(np.searchsorted(self._cdf[int(gi)], u[k], side="right"))
            if not g.success[j] or not np.isfinite(g.log_weight[j]):
                raise RuntimeError("CDF selected a zero/failed-mass endpoint")
            x = g.minimum_positions[j].copy()
            x -= x.mean(0)
            if sigma_A:
                noise = rng.normal(size=x.shape); noise -= noise.mean(0)
                x += sigma_A * noise
            out.append(EndpointDraw(g.group_id, g.particle_id[j], g.basin_id[j], g.condition(), x))
        return out

    def audit(self) -> dict[str, Any]:
        return dict(format=FORMAT, failure_policy=self.failure_policy, failed_mass=self.failed_mass,
            target_changed_by_conditioning=(self.failure_policy != "error" and self.failed_mass > 0),
            groups=[dict(group_id=g.group_id, graph_id=g.graph_id, composition_id=g.composition_id,
                mixture_mass=g.mixture_mass, temperature_K=g.temperature_K,
                particles=len(g.log_weight), failed_mass=g.failed_mass,
                endpoint_record_ess_not_teacher_ess=float(1 / np.dot(g.weights, g.weights)), basin_mass=g.masses(),
                target=g.target, provenance=g.provenance) for g in self.groups])

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        records = []
        for i, g in enumerate(self.groups):
            filename = f"group_{i:04d}.npz"; path = directory / filename
            np.savez_compressed(path, minimum_positions=g.minimum_positions, log_weight=g.log_weight,
                                success=g.success)
            records.append(dict(group_id=g.group_id, graph_id=g.graph_id, numbers=g.numbers,
                charge=g.charge, spin_multiplicity=g.spin_multiplicity, temperature_K=g.temperature_K,
                mixture_mass=g.mixture_mass, basin_id=g.basin_id, particle_id=g.particle_id,
                target=g.target, provenance=g.provenance, array_file=filename, sha256=file_sha256(path)))
        manifest = dict(format=FORMAT, failure_policy=self.failure_policy, groups=records)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")

    @classmethod
    def load(cls, directory: str | Path, *, failure_policy: str = "error") -> "WeightedEndpointStore":
        directory = Path(directory).resolve()
        meta = json.loads((directory / "manifest.json").read_text())
        if meta.get("format") != FORMAT:
            raise ValueError("Unknown endpoint format")
        groups = []
        for record in meta["groups"]:
            args = dict(record)
            path = (directory / args.pop("array_file")).resolve()
            if path.parent != directory:
                raise ValueError("Unsafe endpoint path")
            expected = args.pop("sha256")
            if file_sha256(path) != expected:
                raise ValueError("Endpoint array checksum mismatch")
            with np.load(path, allow_pickle=False) as values:
                args.update({k: values[k] for k in ("minimum_positions", "log_weight", "success")})
            for key in ("numbers", "basin_id", "particle_id"):
                args[key] = tuple(args[key])
            groups.append(EndpointGroup(**args))
        # Do not silently adopt a conditioning policy merely because a file set it.
        return cls(groups, failure_policy=failure_policy)


def endpoints_from_smc(population: Any, *, minimum_positions: np.ndarray,
                       success: np.ndarray, basin_id: tuple[str, ...],
                       group_id: str, graph_id: str, numbers: tuple[int, ...],
                       charge: int, spin_multiplicity: int, temperature_K: float,
                       mixture_mass: float, target: dict, provenance: dict) -> EndpointGroup:
    """Bridge the repository's TemperedPopulation into a minima endpoint group.

    Uses CURRENT normalized particle weights, not a product of ancestral works
    after resampling. The caller performs/audits the fixed optimization map and
    supplies statuses; the function does not turn arbitrary SMC into equilibrium.
    """
    if not population.history or not math.isclose(float(population.history[-1]['beta']), 1., abs_tol=1e-12):
        raise ValueError('Only a completed final-target SMC population may be exported')
    lw = population.log_weights.detach().cpu().double().numpy()
    if not math.isclose(logsumexp(lw), 0., abs_tol=1e-8):
        raise ValueError('Expected current normalized SMC weights')
    if len(population.positions) != len(lw):
        raise ValueError('SMC population shape mismatch')
    return EndpointGroup(group_id, graph_id, numbers, charge, spin_multiplicity,
        temperature_K, mixture_mass, minimum_positions, lw, success, basin_id,
        tuple(f'smc-particle-{i}' for i in range(len(lw))), target,
        {**provenance, 'teacher_kind': 'repository_tempered_smc',
         'log_normalizer_estimate': float(population.log_normalizer_estimate),
         'initial_ancestors': population.ancestors.detach().cpu().tolist(),
         'target_evaluations': int(population.target_evaluations),
         'weight_source': 'current particle log_weights after any resampling'})
