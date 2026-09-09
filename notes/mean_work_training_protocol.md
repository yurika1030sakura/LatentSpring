# Mean-work teacher calibration

This is a standard stochastic-normalizing-flow / path-space reverse-KL teacher
objective initialized from the metadata-aware FM model. It does not require an
already reliable importance-weighted teacher, and it is not claimed as a new
identity. It is separate from any final FM student or de novo composition prior.

For forward path Q=q0 product K and auxiliary backward kernels L, minimize

    E_Q[ U(x_T)/kT + log q0(x_0) + sum_k(log K_k - log L_k) ].

This is KL(Q || normalized_target_endpoint * product L) minus log Z. Both the
forward and backward drift networks may be trained. The backward-only control
freezes the forward network and verifies that held-noise generated coordinates
are unchanged. Negative shifted work values are permissible. The source DFT
energy is a fixed logging offset; it does not initialize generated coordinates.

The first Euler preflight used the confinement-width Gaussian at the input of
a model trained at unit width. Four- and sixteen-step paths had poor geometry
(radius of gyration about 14--15 A). This was not a valid performance result.
Two optimizer updates with and without gradient checkpointing give identical
forward and backward parameter tensors. Retain that negative initialization
evidence; the problem was not repaired by declaring finite gradients sufficient.

The next kernel preserves native unit-width input and uses an explicit Gaussian
reference bridge to the confinement width sqrt(kT/kappa). Define s_t geometrically
between these endpoint widths and rho_k=exp(-noise_scale^2 * dt/2). Reference
transitions are

    K0(y|x)=N(rho_k*s_next/s_prev*x, s_next^2*(1-rho_k^2) I),
    L0(x|y)=N(rho_k*s_prev/s_next*y, s_prev^2*(1-rho_k^2) I).

They are exact forward and conditional reverse kernels of the prescribed
Gaussian marginals. Neural residual mean shifts dt*v are added to each side.
The residual drift is smoothly bounded by 20*sqrt(dimension). All Gaussian
densities describe the actual implemented discrete transitions. With zero neural
residual and a normalized Gaussian endpoint target, work is identically zero
at any step count; this is checked for 1, 3 and 16 steps.

Original electronic states are used, and the target remains eSEN plus the
declared .1 eV/A^2 harmonic restraint, at kT=1 eV. This energy scale is not an
ambient-temperature label for the OMol corpus. The native-width Gaussian bridge
preflight improves numerical scale but still has low ESS and poor geometry;
two updates do not establish molecular benefit.

Energy values and forces are queried in the separate omol25 subprocess. A
custom autograd function supplies the frozen oracle's first derivative; higher
energy derivatives are explicitly unavailable. All trajectory/parameter paths
are retained, and checkpoint recomputation re-enters deterministic model mode.
Analytic checks compare full gradients with fixed-noise finite differences,
checkpointed/plain gradients, exact Gaussian work and external-force gradients.

The bounded GPU calibration compares 100 updates of joint forward/backward
work training with 100 updates of backward-only training, starting from the
same checkpoint. Each uses two paths per update, 16 transitions and the same
64 held-noise paths before/after. Report all oracle/model calls, memory, work
statistics, unweighted geometry/energy, and importance weights. The frozen-forward
control isolates auxiliary-path improvement from changed generation. No result
may be promoted to an ICLR contribution without stronger matched baselines,
multiple training seeds and the reserved evaluation protocol.

Both first calibration jobs completed. Joint energy decreased by 13.6534 eV on
64 held-noise draws, but ESS remained 3.485/64 and unweighted overlap increased
from 12.5% to 17.1875%. Backward-only coordinates were bitwise unchanged; its ESS
was 2.666/64. This does not establish useful molecular sampling.

The next matched development run uses 500 updates, batch 16, 16 transitions,
256 evaluation particles before/after and seed 9051. Both arms start from the
same electronic FM checkpoint and use the same oracle and random streams.
Each uses 8,512 potential evaluations, including 512 for evaluation. Compare
full joint mean-work gradients to an intentional forward-energy-only ablation.
For that ablation, detach both states entering each backward conditional density
and the forward log-kernel term, while keeping backward parameters live. Thus
forward parameters receive only terminal energy/restraint gradients; backward
parameters retain conditional negative-log-likelihood gradients. Work values
and sampled paths do not change at fixed parameters. Networks have disjoint
parameters; global norm clipping and optimizer settings match. This is a
gradient control, not an alternative derivation of full mean-work gradients.

Batching in the separate fairchem process uses its AtomicData batching utility,
with the same ASE conversion, state validation and predictor as serial queries.
Serial remains the default; enable batches only after a recorded energy/force
agreement check. Batch size changes roundoff and throughput, not potential counts.
