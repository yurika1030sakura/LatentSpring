# Independent HMC-SMC reference pilot

This is standard Gaussian-start tempered SMC with target-invariant HMC moves,
not a new algorithm or a reference convergence claim. Earlier hot-target SMC
runs used short MALA/RWM moves; their failures remain in STATUS. HMC and longer
annealing at 300 K are a distinct, bounded test while mean-work training runs.

Calibrate first on AgBr2 with the existing independent v2 quadrature. Start 32
iid COM coordinates from normalized N(0,10 I), without FM weights or molecular
reference positions. Use original charge/spin, eSEN checkpoint and .1 eV/A2
restraint, kT=.025851999786435. The initial 10 A2 variance corresponds to the
same restraint at a 1-eV reference scale. The fixed geometric bridge schedule is
beta=(k/128)^2. Each stage reweights before optional multinomial resampling at
ESS<N/2, then applies one HMC trajectory of eight leapfrog steps. Step size is
.05 sqrt(kT/max(beta,kT)); momentum is fully refreshed and score norm capped at
100/kT with true Hamiltonian correction. Seed 9069 is frozen before results.

Cached initial bridge values eliminate duplicate starting-point oracle calls.
Accepted physical values/scores are recovered algebraically from the bridge
at beta>0. A terminal independent re-evaluation checks cache agreement within
.001 reduced-energy unit, allowing previously measured oracle batch-layout
roundoff; analytic Gaussian tests demand agreement within 1e-10. Total budget
is 32*(1+128*8)+32 = 32,832 queries, with terminal checks explicitly included.

Report path normalizer estimate, reference discrepancy, weighted invariant
moments, endpoint ESS and retained ancestry. One resampled population gives no
iid endpoint error bar. Initial-lineage loss is not itself a mixing certificate
or a rigorous effective-sample bound after mutation. As an exploratory screen,
require ESS>=8, at least four initial ancestors, |delta log Z|<.25 and maximum
Ag-Br mean distance error <.1 A before a larger-condition pilot. This screen is
not a confidence interval or proof of convergence. Do not relax it after seeing
the first result; repeated independent populations would still be required.

## Replication of the failed 32-particle pilot

The first population (9069) failed the normalizer screen: delta log Z +.71340,
ESS 23.14/32, five ancestors but weighted ancestry ESS 1.326. Keep its failed
screen unchanged. Before changing particle count, schedule or target, freeze
three further seeds 9070, 9071, 9072 with the identical recipe. Retain all four
populations; pool normalizers on the linear scale and assess between-population
variation. This adds 98,496 oracle queries. It is a diagnostic of repeatability,
not permission to waive the original gate or declare an eight-atom reference.
No single best seed will be selected for promotion.
