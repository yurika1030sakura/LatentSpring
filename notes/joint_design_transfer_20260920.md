# Both-design transfer to EDM and GAGA

The user requests only original-baseline versus both-design comparisons in the
transfer subsection. The main paper and its existing results remain unchanged
until this new experiment is complete. No extra long-schedule inference is added.

## Consistent structured diffusion

Sample the same auxiliary tree law and atomic edge scales as the existing FM
harmonic source. For that tree, B is the centered edge-to-atom embedding times
the edge standard deviations, and C=B B^T. C has rank N-1 and is positive definite
on the zero-centroid subspace. A draw Bz is exactly the existing harmonic source
conditional on the tree; averaging trees reproduces its mixture.

Retain the baseline scalar VP schedule, and replace white noise by this colored
noise in both training and sampling:

    X_t = alpha_t X_clean + sigma_t B z.

Hold the tree fixed throughout each sampling trajectory. To make the conditional
reverse process identifiable, pass the tree covariance through the existing
second invariant edge channel. The feature is log(1+3(C_ii+C_jj-2C_ij)); it uniquely
specifies the centered covariance and adds no parameters. These are auxiliary
noise dependencies, not supplied chemical bonds. Unlike the main FM, this
diffusion extension uses the tree covariance throughout the process; describe
that adaptation explicitly rather than implying an unchanged isotropic denoiser.

The network predicts physical-coordinate noise eta=Bz. With D the oriented edge
incidence divided by edge standard deviations, C^+=D^T D on the centered space.
Train with ||D(eta_pred-eta)||^2/[3(N-1)]. This is the whitened reconstruction loss
without a numerically expensive or ambiguous node-space square root. The scalar
posterior mean coefficients remain the usual VP coefficients; posterior noise
is multiplied by B. Tests compare these coefficients with Gaussian conditioning.

EDM starts from Bz. GAGA retains its truncated timestep650 and the original
training-only data variance v. Its starting covariance becomes
alpha_T^2*v*P+sigma_T^2*C, where P removes translation. This is the corresponding
Gaussian data approximation conditional on the tree. It does not assert that
the previously selected timestep is optimal for the new corruption process.

The provisional endpoint is (X-sigma*eta_pred)/alpha, in physical coordinates.
The existing FM head is reused verbatim at strength4; the same endpoint-displacement
adapter therefore still applies. No new head fitting, force queries, H readout,
test-time energy ranking, or geometry minimization is introduced.

## Experiment

Two targets, two paired archived baseline initializations each. Retrain each
structured diffusion backbone for30000 one-pass updates of batch32 on exactly
the same20000 OMol25 records and batch schedule as its baseline. Use final EMA;
fixed-noise validation losses at0/10000/30000 diagnose learning and do not select
checkpoints. The physical heads are the two previously completed FM heads.

Use the existing64-composition panel with16 draws/fit/condition,128 backbone
calls and128 small-head calls per output. Original-baseline outcomes are reused
from their audited archives. Only the both-design arm is newly generated/scored.
Four new fits add120000 updates,3840000 training-example forwards,192 diagnostic
validation forwards,4096 evaluation trajectories/GFN2 attempts, and64 zero-head
sampler verification outputs. All fits/outcomes are retained. Primary joint-yield
intervals account for the two target comparisons; report training variation.

This is transfer of two method designs with structured-backbone retraining;
only the physical-head weights transfer without retraining. It is not an
entirely zero-shot transfer and not a new physical law or universal-framework claim.

## Primary background

- Voleti, Oberman, Pal, *Score-based Denoising Diffusion with Non-Isotropic Gaussian
  Noise Models*: https://arxiv.org/html/2210.12254v1 . Non-isotropic Gaussian
  noising and its reverse process are established constructions.
- Qu et al., *GAGA: Gaussianity-Aware Gaussian Approximation for Efficient 3D
  Molecular Generation*, ICLR2026:
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/08de3c1eb4adc7ac3c1949048422d895-Abstract-Conference.html .
  The truncation/variance-approximation motivation is inherited from GAGA.

Our contribution in this experiment is an explicitly tested combination of our
composition-conditioned tree mixture and the frozen learned physical correction.
Positive performance is not assumed by the construction.
