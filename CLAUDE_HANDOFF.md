# Claude continuation — latent component-mass calibration

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `research/NEXT.md`,
`research/LATENT_MASS_STATE_20260913.json` and
`notes/latent_mass_calibration_v1.md` before further work.

The offline known-target coupling experiment and exact nonlinear control are
complete and audited. Nonlinear coupling helps the constructed nonlinear case;
the simple binned map matches neural offline. The all-query online experiment
46320232 and audit46320415 are complete. The final all-cost summary and figure
are linked in NEXT; all stage jobs are terminal. Check live Slurm before recovery. All target values
used for online fitting remain in the estimates; no post-update remapping of old
samples is allowed. The audit includes48 full-history replays (32 adaptive,16 static) and all
384 histories' weights and prefix normalizers.

There is no new molecular performance result yet. Generic coupled importance
sampling and Gaussian-preserving flow rearrangement are prior art. This module
changes calibration covariance, then mixture probabilities, while keeping
conditional reference shapes fixed. Do not describe it as a complete improved
Boltzmann generator or distinct ICLR-level AI novelty based on the toy result.

Keep the previous routing branch closed to scaling, preserve its exact
zero-learning control and every negative outcome. Never substitute the corrected
clamped q0.95 for the unknown production FM64-plus-noise density. Preserve all722
reserved outcomes, exclude the12/18 evaluated cohorts from fitting and keep the
two environments separate. The user's priority is a useful main-method advance
and rapid evidence; broad cleanup and an independent new physics law are not
requirements. The full ICLR goal remains active and unachieved.

Final online nonlinear-case RMSE:0.012027 neural,0.014159 binned,0.016519
independent at512 total calls including learning. The neural-minus-binned and
neural-minus-independent paired MSE intervals both span zero. Constant/Gaussian
maps have lower point error on the constant case. Keep this as a bounded
estimator prototype; marginal weight collapse and within-component geometry
remain unchanged. The next main-method choice is open, not an implemented
ICLR-level solution. Do not scale toy variants to force a neural win.
