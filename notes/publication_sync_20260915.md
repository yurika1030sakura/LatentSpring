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
