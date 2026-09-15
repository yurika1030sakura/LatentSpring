# LatentSpring

**Physics-Informed Molecular Flow Matching from Atomic Composition**

LatentSpring generates molecular coordinates from atomic composition using a
harmonic source with latent spatial connectivity. Physical feedback enters through
training targets, while generation uses a single neural model.

- [Current manuscript (PDF)](paper/latentspring.pdf)
- [LaTeX entrypoint](paper/main.tex)
- [Title and abstract](paper/submission_abstract.txt)
- [Reproducibility guide](paper/REPRODUCIBILITY.md)

## Method and evidence

The source marginalizes over random harmonic trees, giving a normalized density
without supplied chemical bonds. Matched OMol25 continuations improve raw graph
support over Gaussian sources across three composition panels. A physical training
update lowers raw energy and force under eSEN and independent GFN2-xTB evaluation.
A controlled molecular-rotor study tests complete escorted-work correction against
resolved local target distributions.

The experiments use pretrained backbones and selected OMol25 training subsets.
Generator quality, local work correction, and full molecular equilibrium are
separate evaluation questions. The paper states the task and comparison settings
for each reported result.

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
