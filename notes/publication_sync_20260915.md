# GitHub and Overleaf replacement

The user explicitly authorized replacing the previous BGFM publication on GitHub
and the existing Overleaf project with the completed LatentSpring manuscript.
Repository/project URLs are retained. Primary manuscript entrypoints and branding
switch to LatentSpring, with previous Git revisions available for recovery.

GitHub has25 later BGFM-Native commits outside the audited research branch.
The publication branch will preserve that history as a merge parent and a backup
branch, while using the validated LatentSpring source snapshot as its working tree.
This avoids mixing untested native-framework changes into the audited method.
No force push or deletion of the previous history is needed.

The actual18 cited references have been reviewed. The foundational FM citation was
restored, FlowMol3 updated to its2026 Digital Discovery record, and several DOI,
proceedings and author-name details completed. FAB's role is stated as target-density
training, separately from reinforcement-learning fine-tuning. Preprints remain
identified as preprints. Overleaf receives only the bibliography entries used by
the current manuscript, all included sections, three figures and required local
style files. The independent source package must compile before remote replacement.

Learned-generator evidence currently includes matched Gaussian/covariance source
controls and the independently adapted EDM. The causal source comparison remains
the matched-backbone study. Different pretraining histories and task adaptation
prevent treating the EDM result as its published native benchmark or a universal
SOTA comparison. A stronger fully matched contemporary generator study would
support broader competitiveness claims; it is not invented by bibliography editing.

## Completed local verification

Both canonical and standalone manuscripts compile without unresolved references,
overfull boxes or unstable labels. Their extracted PDF text is identical. Scientific
main text ends on page 8, the AI-use statement ends on page 9, and both PDFs contain
14 pages. All 21 exported source files match the recorded manifest; compiler
intermediates are excluded from the ZIP. Current build evidence is
`research/evidence/publication_build_v1.json`. The current and legacy PDF entrypoints
contain the same LatentSpring manuscript. A scan of outgoing history found no
credential patterns, and no outgoing Git blob exceeds the GitHub size limit.

## Remote replacement completed

GitHub main now contains the audited release at
`cb488cfcdbf1ee51f78e3994493b905d63c4c6c8`. The earlier main is retained at
`archive/bgfm-before-latentspring-20260915` and as a merge parent. The default README,
PDF, LaTeX entrypoints and standalone archive all use LatentSpring.

The original Overleaf project's main branch is now
`e2d291cceccf7a3d573f8521ed7e2b69d7e465da`. A fetch after the push confirms its
source commit; all exported files match the independently compiled package's
manifest. The existing HOLY Overleaf checkout was also advanced with a fast-forward
merge and is clean. The Overleaf web project display name and cloud compiler state
were not changed through Git.

- GitHub: https://github.com/yurika1030sakura/bgfm
- Overleaf: https://www.overleaf.com/project/6a4307765d3baef16c61ccb1

These are publication-source updates, not an OpenReview submission. The manuscript
is an anonymous submission draft; author and submission metadata remain with the
user. This synchronization receipt is a subsequent documentation-only commit.

## Expanded reader-oriented release

The follow-up adds the64-composition source comparison, independent unpretrained
EGNN/EDM/GAGA comparisons, matched-budget self-conditioning ablations, a same-norm
physical-update control, and a displacement-only physical-target control. All
planned studies are audited. The paper explains notation before equations, gives
a reader's guide and workflow figure, and retains the full comparator outcomes.

The new canonical and standalone builds both have9 scientific main pages and20
pages total. Their extracted text is identical; all28 exported source files match
the manifest. There are19 verified cited references. Build evidence is
`research/evidence/publication_build_v2.json`. The updated package source is
`runs/publication_20260915/overleaf_source_v3`.

The expanded release was pushed to GitHub main at a1ab01c and to the existing
Overleaf project at2f1f9c4. Remote references were verified, and fetching the
Overleaf source reproduces the28-file export manifest. The original HOLY Overleaf
checkout was fast-forwarded and is clean. This updates paper sources and PDFs;
it does not submit to OpenReview or verify an Overleaf web compiler session.


## Molecular-figure and scientific-language release (preparing publication)

The manuscript now states the composition/electronic-state input, sampled source
coordinates, time-dependent neural inputs, coordinate velocity output and final
coordinates explicitly. Collaborator wording was reviewed against implementation;
set notation, centering, neutral-singlet definitions and the coordinate-head role
were corrected. The source density is consistently q0, the physical-update alpha
is typeset correctly, and the midpoint-time sampler description follows code.

The literature survey covers 32 identified Luo/Shi AI-for-science work families:
30 full texts and two primary overviews. Original molecular artwork replaces the
main schematics; the actual flow trajectory and prescribed rotor scan are also
included as supplementary GIFs. No external paper artwork is republished.

The new draft retains the complete matched-generator quality comparison and
curvature-teacher/neural results. Teacher ESS improves; extra neural improvement
from curvature work remains unestablished. The existing force-update generator
remains the primary method. The separate new GAGA feedback module has passed
three unit tests but has not yet been trained or claimed as a paper result.

The canonical build has 9 scientific main pages, 25 total pages and 24 used
citations, with no unresolved references, overfull boxes or unstable labels.
Standalone export/build and remote verification follow before publication is
recorded as complete. The Overleaf revision will descend from collaborator
commit 0eecd2802fcd235e54e0e3f3ca50e7b3db44cf15.


### Molecular release synchronized and verified

GitHub main received release `398a82511b6aa625306ffcaf26a9b6f38e8c3fbc` (full ID in publication_sync_v3.json),
and Overleaf main received `239932c45525fbf453966431c1be5643d4174655`.
The Overleaf commit directly preserves collaborator `0eecd28` as its parent.
Both remote heads were fetched after pushing. All 34 exported source/media files
and the manifest match the locally compiled standalone package byte for byte.
Canonical and standalone PDF text is identical; both have 9 scientific main pages
and 25 total pages. The source ZIP includes 24 used references and two GIFs.
The original HOLY Overleaf checkout is fast-forwarded and clean. Cloud compilation
and OpenReview submission are not claimed. The GAGA feedback challenge remains
separate ongoing research with three passing module tests and no trained result.
