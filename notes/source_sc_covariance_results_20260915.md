# Covariance control for the harmonic source

All2816 new source and output records replay. Both training seeds and both panels favor the harmonic mixture over the covariance Gaussian. The declared attribution gate passes.

| Panel | Seed | Covariance Gaussian | Harmonic tree |
|---|---:|---:|---:|
| development | 1 | 332/768 | 436/768 |
| development | 2 | 313/768 | 388/768 |
| confirmation | 1 | 286/640 | 413/640 |
| confirmation | 2 | 305/640 | 375/640 |

development: harmonic minus covariance Gaussian +11.65pp; conditional paired95[8.59,14.78], descriptive composition95[8.07,15.43].

confirmation: harmonic minus covariance Gaussian +15.39pp; conditional paired95[11.64,19.06], descriptive composition95[11.95,18.98].

The covariance source uses128 trees, calibrated against8192 with1.85–6.25% relative Frobenius error across22 conditions. No exact covariance or universal causal attribution is claimed. All source arms share the SC network, warm checkpoint,3000 continuation rows/updates and128 primitive sampling calls. This additional control reuses already examined panels. No new independent model replication of the harmonic reference.

This strengthens the task-specific source-method evidence, not a claim to invent random-tree priors, self-conditioning or Jarzynski. Original native endpoint comparison remains separate and has different training history.
