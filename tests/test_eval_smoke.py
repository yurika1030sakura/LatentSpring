"""Smoke tests for the four evaluation scripts (no GPU, no xtb).

Verifies that:
  - eval_qm9_ebmol_protocol.evaluate_samples returns all six metric keys
    on a synthetic batch and is robust to malformed samples.
  - eval_geomdrugs_ebmol_protocol.evaluate_samples returns the
    GEOM-Drugs metric set including vendi_diversity (NaN if RDKit
    fingerprints unavailable).
  - eval_xtb_relaxation summary helpers compute correct percentiles
    and failure rates on synthetic per-sample records.
  - eval_cross_model_ranking correlations match hand-computed values
    on a tiny pool.

Run with:
  python tests/test_eval_smoke.py
or via pytest.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_qm9_eval_returns_metric_keys():
    from scripts.eval_qm9_ebmol_protocol import evaluate_samples
    # Trivial water + methane to ensure the script runs without crashing
    # even if rdkit's xyz->mol path is unavailable: the function should
    # still return a metric dict with the right keys.
    samples = [
        {"atomic_numbers": [8, 1, 1],
         "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]},
        {"atomic_numbers": [6, 1, 1, 1, 1],
         "positions": [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                       [0.63, -0.63, -0.63], [-0.63, 0.63, -0.63]]},
        {"atomic_numbers": [],  # malformed
         "positions": []},
    ]
    metrics = evaluate_samples(samples)
    for k in ("n_samples", "atom_stability", "molecule_stability",
              "validity", "uniqueness", "valid_and_unique", "novelty"):
        assert k in metrics, f"missing metric {k} in {metrics}"
    assert metrics["n_samples"] == 3
    assert 0.0 <= metrics["validity"] <= 1.0
    print(f"[smoke] qm9 metrics OK: {metrics}")


def test_geomdrugs_eval_returns_metric_keys():
    from scripts.eval_geomdrugs_ebmol_protocol import evaluate_samples
    samples = [
        {"atomic_numbers": [6, 6, 1, 1, 1, 1, 1, 1],
         "positions": [[0, 0, 0], [1.54, 0, 0],
                       [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                       [0.63, -0.63, -0.63], [2.17, 0.63, 0.63],
                       [2.17, -0.63, -0.63], [1.54, 0, -1.0]]},
    ]
    metrics = evaluate_samples(samples, vendi_subset=8)
    for k in ("n_samples", "atom_stability", "molecule_stability",
              "validity", "valid_connected", "uniqueness",
              "valid_and_unique", "novelty", "vendi_diversity"):
        assert k in metrics, f"missing metric {k} in {metrics}"
    print(f"[smoke] geomdrugs metrics OK: {metrics}")


def test_xtb_summary_helpers():
    from scripts.eval_xtb_relaxation import summarize, _percentile
    records = [
        {"index": 0, "delta_E_kcalmol": 1.0, "max_force_evA": 0.05,
         "rmsd_A": 0.1, "steps": 10, "converged": True, "failure_reason": None},
        {"index": 1, "delta_E_kcalmol": 2.0, "max_force_evA": 0.10,
         "rmsd_A": 0.2, "steps": 20, "converged": True, "failure_reason": None},
        {"index": 2, "delta_E_kcalmol": 5.0, "max_force_evA": 0.50,
         "rmsd_A": 0.5, "steps": 40, "converged": False, "failure_reason": None},
        {"index": 3, "failure_reason": "single_point_failed"},
    ]
    summary = summarize(records)
    assert summary["n_total"] == 4
    assert summary["n_ok"] == 3
    assert math.isclose(summary["failure_rate"], 0.25)
    assert math.isclose(summary["delta_E_kcal_median"], 2.0)
    assert math.isclose(summary["delta_E_kcal_mean"], 8.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 90.0), 4.6)
    print(f"[smoke] xtb summary OK: {summary}")


def test_cross_model_ranking_correlations():
    from scripts.eval_cross_model_ranking import (
        _pearson, _spearman, _auroc, _correlate_scorer,
    )
    # Hand-checked correlations on a tiny set.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.1, 5.9, 8.0, 9.8]
    assert _pearson(x, y) > 0.99
    assert _spearman(x, y) > 0.99
    # AUROC: scores low for negatives, high for positives.
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1, 1, 0, 0]
    assert math.isclose(_auroc(scores, labels), 1.0)

    # Per-scorer correlation on a synthetic ΔE distribution.
    records = [
        {"score_omol25": s, "delta_E_kcalmol": -s + 0.05}
        for s in [1.0, 2.0, 3.0, 4.0, 5.0]
    ]
    stats = _correlate_scorer(records, target_key="delta_E_kcalmol",
                              scorer_key="score_omol25")
    assert stats["n"] == 5
    # Strong negative Pearson because score increases as target decreases.
    assert stats["pearson"] < -0.99
    print(f"[smoke] cross-model ranking OK: {stats}")


if __name__ == "__main__":
    test_qm9_eval_returns_metric_keys()
    test_geomdrugs_eval_returns_metric_keys()
    test_xtb_summary_helpers()
    test_cross_model_ranking_correlations()
    print("\n[smoke] ALL EVAL SMOKE TESTS PASSED")
