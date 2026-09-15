# Reader-oriented method and component evidence

The user requests accessible explanations of every symbol, formula and technical
term, with experimental justification for each claimed methodological benefit.
Start from the molecular task and purpose of each operation; introduce notation
after its meaning. The final network weights will be denoted theta_final, avoiding
an unexplained star. A component's intended role is not itself evidence of benefit.
Definitions and probability identities need explanation/proof; empirical advantage
claims need direct comparisons. Remove or narrow unsupported advantage claims.

Existing evidence: the harmonic source has matched Gaussian/shell/covariance
controls. Replay, direct force/work continuations, and paired parameter updates
were all evaluated. The previous SC ablation used a shell source and therefore
does not directly establish benefit with the final harmonic source. Jarzynski
has positive local distribution-correction evidence; an extra neural advantage
over force-only updating is not established.

Two additional frozen studies address the user's specific questions:

- No self-conditioning on the harmonic source, with3000 and6000 update readouts.
  The first matches optimizer updates; the second matches the reference's6000
  primitive training forwards. All inference uses128 primitive calls. The same
  3000 rows, warm initialization and source/coupling settings are retained.
- A direct physical update scaled to the norm of the paired update. The scale is
  computed only from saved parameter vectors, before generation. This tests the
  direction benefit of replay subtraction against an update-magnitude control.
  Frozen, direct and paired outputs/energies are already available on the same
  streams; only the new control needs generation and scoring.

The ongoing64-composition study and independent four-method/two-initialization
EGNN benchmark continue. These ablations add essential component evidence and
do not change their frozen protocols. No positive result is assumed in advance.

## Completed evidence

The64-composition experiment is audited: Gaussian24.85% versus harmonic31.76%,
+6.91pp with a stratified composition interval[4.74,9.03]. Pooled gains are
positive in all four size bins. Original jobs finished in47minutes, so the
prepared scheduler continuation was not needed.

The current harmonic-source SC ablation is audited:52.73% for SC,45.51% for
no-SC3000 and40.30% for no-SC6000. SC improves both continuations, at equal
optimizer updates and at equal training-forward counts. Counts do not equate
all operations or hardware timing.

The paired physical update beats the norm-matched direct update in energy by
0.03232 eV/atom and force RMS by0.4170 eV/A on jointly valid outputs. Both
continuations improve, and both intervals exclude zero. Graph validity is similar.
This supports update direction beyond shrinking its magnitude.

The component replay initially stopped on4.44e-16 A CPU-BLAS rounding. It now
allows1e-12 A absolute source replay error, while requiring exact seeds/trees and
unchanged raw sample/readout hashes. The actual maximum error is1.78e-15 A.
No generated structure or reported readout was changed.

## Manuscript changes

Inputs, coordinate space, source probabilities, interpolation time, both network
passes, every weight vector, and local work are explained before use. An appendix
glossary distinguishes graph validity from same-graph teacher restrictions.
Theta_final names the final weights. Unused source-gradient formulas were removed
from the main narrative. The physical displacement is derived by completing the
square in a Boltzmann-tilted Gaussian; constant-work and quadratic-target tests
pass. A new vector figure separates generation from physical learning.

The main draft includes the wider panel and component comparisons. Older source
controls and the differently pretrained EDM comparison remain in the appendix.
The latest checked build has9 scientific main pages and no missing references or
overfull boxes. The independent scratch comparison remains incomplete: first-seed
Gaussian FM182/2048, harmonic FM223/2048, EDM301/2048 and GAGA321/2048 at128 calls.
Do not turn the source ablation gain into superiority over diffusion. Await both
initializations and the full audit before writing its comparative conclusions.
