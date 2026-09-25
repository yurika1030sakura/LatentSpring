# LatentSpring

**Physics-Informed Molecular Flow Matching from Atomic Composition**

LatentSpring generates molecular coordinates from atomic composition using a
harmonic source with latent spatial connectivity. Recovery training and learned
geometric and physical endpoint corrections improve the raw structures. Sampling
uses neural networks without energy queries or coordinate optimization.

- [Current manuscript (PDF)](paper/latentspring.pdf)
- [LaTeX entrypoint](paper/main.tex)
- [Title and abstract](paper/submission_abstract.txt)
- [Reproducibility guide](paper/REPRODUCIBILITY.md)

## Method and evidence

The source marginalizes over random harmonic trees, giving a normalized density
without supplied chemical bonds. The main experiments train on 20,000 OMol25
structures and evaluate 64 unseen compositions. Five fitted models reach 35.35%
graph-and-force yield and 9.63% geometry-and-force yield; every generation attempt
remains in the denominator. Frozen correction networks also improve EDM and GAGA
under the shared evaluation setup.

A separate molecular-rotor study examines local nonequilibrium work correction.
The manuscript distinguishes raw structural quality, force criteria, and local
distribution recovery. Detailed protocols, costs and limitations are included.

## Build the paper

```bash
bash paper/build.sh /tmp/latentspring-paper-build
```

The main document is `paper/main.tex`. `paper/bgfm_paper.tex` and its PDF remain
compatible entrypoints for previous links and now display LatentSpring.
A self-contained Overleaf source archive is provided at
[paper/latentspring_overleaf.zip](paper/latentspring_overleaf.zip).

## Code and experiments

Core implementation is in `cfm_mol/`; controlled experiment scripts and frozen
protocols are under `scripts/research/` and `research/evidence/`. Start with
[CLAUDE.md](CLAUDE.md) and [the current handoff](CLAUDE_HANDOFF.md) for architecture,
environments and experiment provenance. Training/inference and energy evaluation
use separate environments because of their PyTorch dependencies.

This repository developed from BGFM. Earlier implementations and exploratory
results are retained for reproducibility; the current manuscript and default
paper entrypoints are LatentSpring.
