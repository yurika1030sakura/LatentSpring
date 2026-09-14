# Prospective neutral organic monomer benchmark

The broad32-composition assay accepted only6 original reference graphs, with
12 disconnected references and12 validator errors. This motivates a separately
defined primary single-molecule task. It does not change any earlier result or
make the failed static context adapter successful.

## Domain and reference qualification

The new domain is neutral singlet structures with8--40 atoms, containing carbon
and hydrogen, and drawn from H, B, C, N, O, F, Si, P, S, Cl, Br and I. OMol25
remains the data source and the implementation maximum remains200 atoms. This
scope does not cover all metals, spin states, charges or molecular complexes.

From the664 official development candidates, all54 compositions in the four
declared previous development/training panels are excluded. Metadata eligibility
leaves78 references. Their original raw record, source, numbers, charge, spin and
stored energy identities are checked; energy values never rank or select cases.
Every eligible original geometry is evaluated with the unchanged graph and
connected/nonoverlap assay before generating any new model output.

31 references qualify:5 with8--16 atoms,18 with17--28, and8 with29--40. A fixed
metadata hash selects up to8 per bin, giving21 distinct compositions. Each of
these21 references passes the complete assay. The pool, all failures and selected
panel are saved separately. Reference coordinates are never generator inputs or
training examples. Passing this assay is not an electronic-stability certificate.

The panel is narrower than the old broad stress test. It must not be described
as a whole-OMol25 result or as post hoc filtering of the earlier generated samples.
An independent composition-overlap audit checks the actual checksum-verified
processed training and legacy-validation corpora before an unseen-corpus claim.
That audit does not certify arbitrary unrecorded pretraining or all trajectory
relationships.

## Comparison before any new generated outcomes

The first continuation compares all frozen Gaussian, shell, harmonic, node-prior
and pair-prior models. This directly tests the existing learned source candidates
on a reference-qualified task without further fitting or parameter tuning. The
second continuation compares the frozen Gaussian/shell/harmonic models; a second
node/pair continuation does not exist and must not be invented in the report.
A learned-source result from the first continuation alone is exploratory and
requires independent replication before a stable learned-increment claim.

Each model receives64 fresh samples per composition, with common within-seed
random streams. This is10,752 new outputs across eight frozen checkpoints, no
new training and zero physical-oracle queries. All attempts, overlaps, fragments,
graph rejects, validator exceptions, diversity and inference cost must remain.
Report first-continuation node/pair versus fixed and Gaussian, and the two-seed
fixed/harmonic controls. Include conditional paired and descriptive size-stratified
composition intervals. Do not select a different source separately for each seed.

The candidate AI contribution remains the explicit, learnable bond-free spatial
source and its useful integration with flow matching. General tree ensembles,
harmonic priors and latent-conditioned flows have prior art. Additional neural
modules are not a substitute for meaningful utility and a defensible distinction.
