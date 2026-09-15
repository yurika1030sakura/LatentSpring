# Jarzynski mechanism experiment: completed positive result

All1536 GFN2 rotor-grid calculations succeeded and independently reparsed. Three
FIT-only molecules and three fixed escort amplitudes were selected before energy
outcomes. The source, target, work and numerical-reference protocol are frozen in
`research/evidence/rotor_work_v1.json`; audit is `rotor_work_audit_v1.json`.

Both prespecified checks pass. Independent density-ratio quadrature recovers the
normalizer and moments in all9 molecule/escort cases. At2048 draws, complete work
has lower mean64-bin total-variation error than both energy-only and missing-
Jacobian controls in all6 nonidentity cases, using128 repetitions per case.

Mean TV across those6 cases: complete work0.0635853, energy-only0.1532118,
without Jacobian0.1133699. Relative reduction versus energy-only is58.50%.
Complete-work free-energy RMSE ranges0.158–0.734 meV. Local ESS at2048 draws
ranges788–1892. The identity-escort control makes all three weighted methods
identical. All sample sizes8/32/128/512/2048 and all molecules/escorts remain
reported; plots average all6 nonidentity cases.

The reference is the explicitly defined periodic interpolation of512 GFN2 values
on each fixed molecular rotor. Coarse/fine interpolation and dense quadrature
convergence are checked. This is a meaningful controlled validation of complete
work accounting on molecular energy curves, not a new Jarzynski law, a demonstrated
extra neural-training advantage, or full3D canonical generation.

Figure: `research/figures/rotor_work_v1/rotor_work.pdf`.
Main paper section: `paper/sections/rotor_work_results.tex`.
Details: `paper/sections/rotor_work_details.tex`.
