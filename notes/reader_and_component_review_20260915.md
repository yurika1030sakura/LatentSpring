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
