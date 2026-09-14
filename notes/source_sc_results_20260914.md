# Source controls under geometry self-conditioning

| Training seed | Unit Gaussian | Shell tree | Harmonic tree |
|---|---:|---:|---:|
| 1 | 357/768 | 417/768 | 436/768 |
| 2 | 377/768 | 342/768 | 388/768 |

All arms have identical geometry-SC structure,3000 two-pass updates and128 primitive denoiser calls per output. Four control models trained; two existing shell models and1536 shell outputs reused. New outputs3072. All4608 comparison records replay.

- fixed minus gaussian: +1.63pp; conditional paired95[-1.50,+4.88], descriptive composition95[-2.08,+5.99].
- fixed minus harmonic_tree: -4.23pp; conditional paired95[-7.29,-1.11], descriptive composition95[-8.53,-0.20].
- harmonic_tree minus gaussian: +5.86pp; conditional paired95[+2.67,+8.98], descriptive composition95[+0.52,+12.17].

The prespecified shell-superiority gate fails. Harmonic tree is a stronger candidate on this reused12-composition development panel; it improves over unit Gaussian in both seeds, with unequal gain sizes. Its radial Gaussian edges match shell covariance analytically. This is a post-development candidate choice, not proof of originality or a Boltzmann law.

## Frozen confirmation completed

All10 remaining reference-qualified pool compositions,17-28 atoms, were evaluated
with the same frozen models. Exact elemental-count checks found no overlap with
3,902,107 processed training rows or39,415 validation rows. All10 original
reference assays and2,560 source/output records replay. No new training or
molecular oracle calls were used for this confirmation.

| Existing model seed | Unit Gaussian | Harmonic tree |
|---|---:|---:|
| 1 | 326/640 | 413/640 |
| 2 | 349/640 | 375/640 |

Graph support improves from52.73% to61.56%, a gain of8.83 percentage points;
conditional paired95[5.08,12.50], descriptive composition95[4.84,12.66].
Geometric support improves by9.69pp, conditional paired95[6.33,12.89].
All supported graph samples have distinct perceived connectivity within their
composition in this confirmation. Validator errors are zero. Harmonic/Gaussian
generation seconds are175.73/175.43 and178.20/177.74; no speed advantage is claimed.
The frozen confirmation gate passes. These are new compositions but the SAME two
model seeds, not four independently trained replicates. Size/domain coverage
remains narrow. Conditional intervals do not establish population-wide gains.

Current candidate: fixed random-tree harmonic source plus geometry-only
self-conditioning in the bond-free flow-matching generator. The shell-specific
hypothesis loses. The candidate change was made after development and frozen
before this additional composition test; no per-seed winner selection occurred.
A useful improvement is now supported in this setting. Originality relative to
prior source/flow methods and competitiveness against external generators remain
unestablished; no Boltzmann distribution or new physics law is demonstrated.

Both Slurm studies are complete:46497703 and46503573. Four new models were trained,
5,632 new outputs generated, and1,536 old shell outputs reused. No new architecture
or manuscript polishing was added. Keep these selected checkpoints frozen. The
next essential questions are originality and the original native generator
comparison, before additional modules or tuning. ICLR readiness remains false.
