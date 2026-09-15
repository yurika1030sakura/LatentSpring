# Original endpoint-checkpoint comparison

The original30000-step checkpoint restores directly, uses its original endpoint meaning, and is evaluated without new fitting.127 Euler updates plus first-step bootstrap give128 actual denoiser calls. Current categories are clamped for the composition-conditioned task. This is not the published native joint-generator benchmark.

| Sampling stream | Native predicted history | Fixed-category / centered-bootstrap history | Adapted Gaussian | Harmonic tree |
|---|---:|---:|---:|---:|
| 1 | 106/640 | 116/640 | 326/640 | 413/640 |
| 2 | 113/640 | 120/640 | 349/640 | 375/640 |

All2560 new source/structure records replay, with zero validator exceptions. Both original-history variants reuse one checkpoint and two sampling streams. The adapted source models additionally use shared displacement/electronic/pairing adaptation and3000 specialized updates; this is not an equal-total-training-cost comparison or proof of SOTA. The matched Gaussian/covariance-Gaussian source continuations are the primary attribution controls.

The two small-model tests independently verify the endpoint Euler updates, bootstrap denoiser-call accounting, and absence of reference-coordinate leakage. Actual generation checks128 calls per sample batch. Archived model weights are unchanged.

Scope clarification from the frozen helper: clamped_history centers its first explicit bootstrap endpoint, while native_history retains the upstream uncentered bootstrap. The alternate arm is not a pure categorical-history ablation. All output tensors and metrics are unchanged.
