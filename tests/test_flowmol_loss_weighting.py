from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "flowmol3"))

from flowmol.models.flowmol import _apply_loss_weight  # noqa: E402


def test_apply_loss_weight_keeps_cross_entropy_vector_shape():
    loss = torch.arange(5, dtype=torch.float32)
    weight = torch.ones(5, 1)

    weighted = _apply_loss_weight(loss, weight)

    assert weighted.shape == (5,)
    assert torch.equal(weighted, loss)


def test_apply_loss_weight_broadcasts_coordinate_loss_per_atom():
    loss = torch.ones(5, 3)
    weight = torch.arange(1, 6, dtype=torch.float32).unsqueeze(-1)

    weighted = _apply_loss_weight(loss, weight)

    assert weighted.shape == (5, 3)
    assert torch.equal(weighted[:, 0], torch.arange(1, 6, dtype=torch.float32))
