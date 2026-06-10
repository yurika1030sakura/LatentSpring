"""Baseline generative models used in the ICLR paper.

These are **reference implementations inside our repo**, not forks of the
authors' full stacks. Purpose: run them on EXACTLY our data pipeline +
EXACTLY our evaluation code so the comparison is fair, not literature-to-
literature (which is noisy due to atom maps, featurisation, metrics).

Frozen author-code baselines live under `baselines/` at the project root
(MiDi, FlowMol3, SafeDiffuser, MCTD, etc.) for reference.

Modules:
  reflected_sde : Lou-Ermon (ICML 2023) reflected diffusion restricted to
                  the coordinate channel -- THE comparison for E2
                  (convergence-rate experiment).
"""
