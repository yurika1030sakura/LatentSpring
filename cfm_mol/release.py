"""Portable inference for the published LatentSpring models and controls."""
import hashlib
import json
import time
from pathlib import Path

import torch

from . import connectivity_feedback as feedback
from . import matched_egnn as base
from .geometry_recovery_field import GeometryRecoveryField, GeometryThenPhysical
from .hydrogen_completion import complete as complete_hydrogens
from .matched_physical_connection import PhysicalFieldTransform
from .physical_connection import make_physical_connection


class Generator:
    """Load a weights directory and generate unfiltered coordinates in angstroms."""

    def __init__(self, weights, device="cpu", upstream=None):
        self.weights = Path(weights).resolve()
        self.manifest = json.loads((self.weights / "manifest.json").read_text())
        if self.manifest["format"] != "latentspring_release_v2":
            raise ValueError("Expected a LatentSpring v2 release manifest")
        self.device = torch.device(device)
        self.upstream = Path(upstream).resolve() if upstream else (
            self.weights / self.manifest["upstream"]
        ).resolve()
        self.loaded = {}
        self.modules = {}
        for name, checksum in self.manifest["files"].items():
            if Path(name).name != name:
                raise ValueError("Weight filenames must be local to the bundle")
            if hashlib.sha256((self.weights / name).read_bytes()).hexdigest() != checksum:
                raise ValueError("Weight checksum mismatch: " + name)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    @property
    def models(self):
        return tuple(self.manifest["models"])

    def _module(self, name, kind):
        key = (name, kind)
        if key in self.modules:
            return self.modules[key]
        record = torch.load(self.weights / name, map_location="cpu", weights_only=True)
        if kind == "backbone":
            spec = dict(record["spec"], upstream=str(self.upstream))
            model = base.initialize(spec, self.device)
            if spec.get("context") == "distance":
                feedback.install(model)
        elif kind == "geometry":
            spec = None
            model = GeometryRecoveryField(**record["configuration"]).to(self.device)
        elif kind == "physical":
            spec = None
            model = make_physical_connection(**record["configuration"]).to(self.device)
        else:
            raise ValueError(kind)
        model.load_state_dict(record["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        if base.state_hash(model) != record["state_sha256"]:
            raise ValueError("Loaded tensor state mismatch: " + name)
        self.modules[key] = model, spec
        return model, spec

    def load(self, name):
        if name not in self.manifest["models"]:
            raise ValueError("Unknown model: " + name)
        if name not in self.loaded:
            item = self.manifest["models"][name]
            model, spec = self._module(item["parent"], "backbone")
            source = base.HarmonicSource(spec.get("edge_log_width", .2))
            context = feedback.GeometryContext(source, "distance") if spec.get("context") == "distance" else None
            physical = self._module(item["physical"], "physical")[0] if item.get("physical") else None
            geometry = self._module(item["geometry"], "geometry")[0] if item.get("geometry") else None
            if geometry is not None and physical is None:
                raise ValueError("The released geometric-field recipes require a physical adapter")
            self.loaded[name] = dict(parent=model, spec=spec, source=source, context=context,
                                     physical=physical, geometry=geometry, recipe=item)
        return self.loaded[name]

    @torch.no_grad()
    def generate(self, numbers, *, model="latentspring_s0", charge=0,
                 spin_multiplicity=1, samples=16, seed=79151, stream=0,
                 hydrogen=False):
        if charge != 0 or spin_multiplicity != 1:
            raise ValueError("Released checkpoints use charge zero and singlet multiplicity")
        if not isinstance(numbers, list) or not 2 <= len(numbers) <= 200 or any(type(z) is not int for z in numbers):
            raise ValueError("Provide 2--200 integer atomic numbers")
        if any(type(v) is not int for v in [samples, seed, stream]) or samples < 1 or min(seed, stream) < 0:
            raise ValueError("Samples must be positive; seed and stream must be nonnegative integers")
        if sum(numbers) % 2:
            raise ValueError("A neutral singlet requires an even electron count")
        entry = self.load(model)
        parent, spec, source, context = (entry[k] for k in ["parent", "spec", "source", "context"])
        if not {1, 6} <= set(numbers) or not set(numbers) <= set(spec["atomic_numbers"]):
            raise ValueError("Composition is outside the released element vocabulary")
        recipe = entry["recipe"]
        transform = None
        physical = None
        if entry["physical"] is not None:
            physical = PhysicalFieldTransform(parent, spec, entry["physical"],
                recipe.get("physical_strength", 4.), strength_limit=4.)
            transform = physical
        geometry_before = entry["geometry"].forward_calls if entry["geometry"] is not None else 0
        if entry["geometry"] is not None:
            transform = GeometryThenPhysical(physical, entry["geometry"], recipe.get("geometry_strength", 1.))
        decoder = decoder_spec = None
        if hydrogen:
            if not recipe.get("hydrogen"):
                raise ValueError("Hydrogen readout is available on the initial-model study recipes")
            decoder, decoder_spec = self._module(recipe["hydrogen"], "backbone")
        coordinates, starts, changed = [], [], []
        count = [0]
        hook = parent.dynamics.egnn.register_forward_hook(lambda *_: count.__setitem__(0, count[0] + 1))
        hydrogen_calls = 0
        tick = time.perf_counter()
        try:
            for begin in range(0, samples, 8):
                size = min(8, samples - begin)
                local_seed = seed * 1000003 + stream * 100003 + begin
                if context is None:
                    x, x0 = base.sample(parent, numbers, spec["kind"], spec, source, local_seed,
                                        size, 128, field_transform=transform)
                else:
                    x, x0 = feedback.sample(parent, numbers, spec, source, context, local_seed,
                                            size, 128, field_transform=transform)
                if hydrogen:
                    z = torch.tensor(numbers, device=self.device)[None].expand(size, -1)
                    x, info = complete_hydrogens(decoder, x, z, {"network_spec": decoder_spec},
                        mode="molecule", start_time=0., steps=4, velocity_cap=2.)
                    hydrogen_calls += info["network_calls"]
                    changed.append(info["changed"].cpu())
                else:
                    changed.append(torch.zeros(size, dtype=torch.bool))
                coordinates.append(x.cpu().double())
                starts.append(x0.cpu().double())
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        finally:
            hook.remove()
        batches = (samples + 7) // 8
        evaluations = 128 // (2 if context is not None and spec.get("two_pass") else 1)
        assert count[0] == 128 * batches
        assert physical is None or physical.calls == evaluations * batches
        geometry_calls = entry["geometry"].forward_calls - geometry_before if entry["geometry"] is not None else 0
        assert entry["geometry"] is None or geometry_calls == evaluations * batches
        return dict(positions=torch.cat(coordinates), initial_positions=torch.cat(starts),
            hydrogen_changed=torch.cat(changed),
            condition=dict(numbers=numbers, charge=0, spin_multiplicity=1),
            settings=dict(model=model, samples=samples, seed=seed, stream=stream, hydrogen_readout=hydrogen),
            costs=dict(attempted=samples, returned=samples, backbone_batch_calls=count[0],
                geometry_head_batch_calls=geometry_calls, physical_head_batch_calls=physical.calls if physical else 0,
                hydrogen_batch_calls=hydrogen_calls, energy_queries=0, geometry_optimizer_steps=0,
                seconds=time.perf_counter() - tick))
