# Whole-manuscript prose review

Reviewed the main text and all ten existing included section files following the user's
request for natural academic prose without defensive language. Related work now
credits its sources directly. Method and result sections explain the construction
and findings without repeated novelty disclaimers, internal gate language, or
forensic progress reports. Experimental scope is stated through the task, metrics,
comparison settings and a concise discussion.

All numerical tables are unchanged. Relevant controls, failed-calculation counts,
conditional uncertainty, baseline pretraining/budget differences and the defined
local work targets remain explicit. The work-versus-force update comparison keeps
its actual uncertainty. The AI-use disclosure remains factual.

One mathematical presentation correction makes the FM objective match the code:
mean coordinate error for each molecule divides the two Frobenius-squared velocity
errors by3N, with the existing one-half averaging over the two passes. This follows
`cfm_mol/clamped_fm.py` and changes neither implementation nor experimental results.
Energy/force parity notation and the factor exp(-W/kT) are also stated explicitly.

The title now wraps naturally. The complete draft compiles without undefined
references, unstable labels or overfull boxes:8 main pages including the
reproducibility and AI-use statements,14 total. All table data and all referenced
source/figure hashes are checked. Latest evidence:
`research/evidence/submission_manuscript_build_v2.json`.
The plain-text submission abstract is synchronized with the revised manuscript.
No model retraining or new physical calculation was performed for this edit.

A short reproducibility appendix now defines the exact scaffold filter, its degree
caps and distance bounds, and the RDKit version/settings used for graph perception
and connectivity SMILES. These were checked directly against the implementation.
