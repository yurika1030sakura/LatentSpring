# Product-Manifold Constrained Flow Matching for Molecules

Implementation of the method proposed in `g3_proposal/methods_derivation.tex`
Section 3 + Appendix A.

## Goal

A 3D molecular flow-matching model whose samples satisfy valence, steric
exclusion, and connectivity **by construction at every step of the flow ODE**
— no post-hoc RDKit sanitization, no discarded samples.

Novelty vs prior constrained generative work:

| Prior                                                  | Limitation                                        |
| ------------------------------------------------------ | ------------------------------------------------- |
| Lipman / Chen–Lipman 2023/24 (flow matching on fixed manifolds) | fibre doesn't depend on a discrete state |
| Gat et al. 2024 (discrete flow matching)               | discrete only, no continuous fibre                |
| SafeFlow (ICLR 2025) (CBF + flow matching, robotics)   | fixed obstacle set, continuous only               |
| Lou-Ermon ICML 2023, Fishman TMLR 2023, Christopher NeurIPS 2024 (reflected/projected diffusion) | stochastic, fixed domain |

Ours: **a coupled discrete–continuous fibre bundle**
$\mathcal{M} = \bigsqcup_{(a,b) \in \mathcal{M}_{\mathrm{conn}}} \{(a,b)\} \times \mathcal{M}_{\mathrm{ster}}(a)$
with a flow whose velocity is tangent-projected onto the fibre and whose
coordinate is retracted onto the new fibre whenever the discrete base state
jumps. Consistency theorem + gluing lemma in Appendix A of the proposal.

## Repo layout

```
yulili_cfm_mol/
├── baselines/                  # frozen external repos
│   ├── flowmol3/               # Dunni3/FlowMol — THE skeleton for our fork (joint FM)
│   ├── midi/                   # cvignac/MiDi — discrete-diffusion reference
│   ├── reflected_diffusion/    # louaaron/Reflected-Diffusion — reflection code reference
│   ├── safediffuser/           # Xiao et al. ICLR 2025 — CBF-invariant denoising
│   ├── mctd/                   # Ahn-ML ICML 2025 — Monte Carlo Tree Diffusion
│   └── diffusion_mrmp/         # RAISE-Lab UVA — projected diffusion robotics
├── cfm_mol/                    # OUR code
│   ├── domain.py               # valence / steric / connectivity checks (App A Defs 1-4)
│   ├── fibre.py                # tangent projection, retraction, interpolant, Euler step
│   ├── flow.py                 # CFM regression loss + toy velocity net for pipeline tests
│   ├── projection.py           # discrete projections; gluing retract
│   ├── flow_model.py           # TODO: FlowMol3 fork
│   ├── sampling.py             # TODO: ODE integrator with discrete flow + gluing
│   ├── train.py                # TODO: training entry point
│   └── test_fibre.py           # sanity + end-to-end tests
├── notes/
│   ├── MODIFICATION_PLAN.md    # file-by-file plan for FlowMol3 fork
│   ├── gaps.md                 # unresolved math/engineering issues
│   └── robotics_analogy.md     # robotics safe-diffusion/FM analogy
└── configs/
    └── qm9_cfm.yaml            # TODO: training config
```

## Status (Apr 2026)

**Working**:
- `domain.py` — valence / steric / connectivity checks (Appendix A Defs 1-4).
- `fibre.py` — tangent projection, retraction, straight-line CFM interpolant,
  Euler step on the fibre, prior sampler.
- `flow.py` — CFM regression loss; `LinearVelocityNet` toy net for pipeline
  tests.
- `test_fibre.py` — 7 tests passing including gradient flow through `cfm_loss`.

**TODO**:
- `flow_model.py` — fork FlowMol3 velocity head; wire `cfm_loss`.
- `projection.py::project_valence` (Week 3) and `project_connectivity` (Week 4).
- `sampling.py` — full ODE integrator with discrete flow + gluing.
- `configs/qm9_cfm.yaml`, `train.py` — training entry points.

## Quick start

```bash
cd /n/holylabs/ryl_lab/Lab/yulili_cfm_mol
conda run -n chemistry python -m cfm_mol.test_fibre
```

Expected: 7 "ok:" lines followed by "all tests passed."

## Milestones (13-week preprint target)

- [x] Skeleton + 7 sanity tests passing.
- [ ] **Week 1** Reproduce FlowMol3 QM9 baseline (99.9% validity).
- [ ] **Week 2** Wire `cfm_loss` into FlowMol3 training; one QM9 epoch; verify
  loss converges and constrained samples are valid.
- [ ] **Week 3** Implement `project_valence`; train full QM9 with discrete
  projections active.
- [ ] **Week 4** Implement `project_connectivity` + gluing retract in sampler.
  **Kill switch**: if validity on QM9 doesn't match FlowMol3 at equal compute,
  pause.
- [ ] **Week 5** Scale to GEOM-Drugs.
- [ ] **Weeks 6-7** Curate OOD slices (tmQM, kraken, hypervalent, radicals).
- [ ] **Weeks 8-10** OOD benchmarks vs EDM, MiDi, DiGress, VEDA, FlowMol3,
  Branching Flows. **Kill switch**: if baselines don't degrade on OOD, pivot
  to chemistry-only framing.
- [ ] **Weeks 11-13** Write preprint.

## Scoop risk

Fioretto / Christopher, Ahn-ML (MCTD), MCTD-ME protein authors all adjacent.
Weekly arxiv monitor:

- `site:arxiv.org "flow matching" molecule valid`
- `site:arxiv.org "product manifold" flow`
- `site:arxiv.org "constrained flow matching"`
- Fioretto / Christopher / Fishman / Lipman / Chen / Ahn-ML author feeds

## Environment

```bash
conda activate chemistry   # torch, rdkit already installed
```

## License

Our code: MIT. Baselines retain their original MIT licenses.
