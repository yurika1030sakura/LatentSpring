# Frozen-forward reverse-kernel diagnosis

The three learned/fixed/trace precision arms failed to improve calibrated
sampling. Test an existing auxiliary-model mechanism before another large
forward-training change. This diagnostic is not a proposed novel loss.

Source: completed annealed-joint 1500-step eight-atom proposal. Freeze all
forward parameters and buffers. Draw 2048 training paths (seed9081) and256
independent heldout paths (seed9082), using batches64 and the source's actual
Gaussian kernels. Cache full states and log forward path probabilities. Only
heldout terminal energies are evaluated:256 oracle calls total. Training and
heldout random streams are disjoint. Upstream 25,024 source queries and neural
path generation/refitting costs remain separate.

Starting from the saved backward network, make500 AdamW updates, batch16,
lr1e-5, gradient norm cap1, minimizing conditional Gaussian negative log
likelihood on cached paths. Minibatch seed9083. No pathwise gradient through
sampling, no forward update, no resampling and no additional energy labels.
Refitting the same mean family tests optimization lag, not all possible
backward conditional distributions.

Score identical heldout endpoints/energies before and after refitting. Also
fit a per-time scalar backward variance ratio by mean squared training residual
in reference noise units, clipped to [.25,1.9]. Widths never use heldout data.
Report all four combinations of before/after means and original/fitted widths,
including work, ESS and likelihood. Assert forward tensors and heldout positions
unchanged. No xTB rerun is needed to compare these identical coordinates.

## Why the variance cap matters

For a Gaussian reference path, write its reverse innovation as z~N(0,1).
Replacing an auxiliary backward variance by gamma times reference variance
(with unchanged reference means and endpoint target) gives normalized weight

    A_gamma = gamma^(-1/2) exp[(1-1/gamma) z^2/2].

Its expectation is one for all gamma>0, but its second moment is
1/sqrt(gamma*(2-gamma)) only for gamma<2, and infinite for gamma>=2.
The forward generator can be exactly at the target while its path weights
have infinite variance because of the auxiliary choice. Positive bounded
covariance is therefore not by itself a finite-weight-variance guarantee.
The same calculation applies to narrowing a forward Gaussian conditional by
precision factor gamma while retaining the reference reverse kernel.

The <2 backward-variance condition extends as a sufficient second-moment
condition to the scalar-reference branch with uniformly bounded mean residuals
and a residual energy bounded below: in reverse-reference innovation coordinates,
the squared-weight integrand is bounded by a Gaussian with quadratic coefficients
1/gamma_k-1/2, times an exponential linear factor. This is a conditional mathematical
statement, not a verified global property of every neural potential. The1.9 cap
avoids introducing the simple auxiliary Gaussian tail failure in this diagnostic.
Finite moments still do not guarantee useful finite-sample ESS or mode coverage.

Execution smoke: two updates,128 training paths,64 heldout paths, same other
settings. Require finite likelihood/gradients and unchanged forward/endpoint
state before the full bounded diagnostic.
