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

Independent structure assessment uses all stored geometries for contact and
element-pair-distance diagnostics. The latter is permutation/rigid-motion
invariant but incomplete, and is not molecular RMSD or basin coverage. Contact
graphs use RDKit covalent radii, factor 1.25 for contacts and .6 for overlaps;
disconnected components are not automatically failures of the restrained target.
Select 32 sample indices by torch.randperm with seed 9059 before inspecting any
xTB outcomes, using the same indices for every arm. GFN2-xTB uses original total
charge and multiplicity, single-point forces, and tight relaxation (200 cycles).
Retain every attempt and failure. Report success-only strain with its convergence
denominator, and pairwise outcomes including both-failed cases. This protocol
first evaluates the completed 100-update calibration; it also applies unchanged
to the ongoing 500-update paired experiment. Source sample/result hashes and
the evaluator version are recorded. No final reserved condition is used.

The initial/100-update panels pass xTB relaxation 0/32 and 6/32 respectively;
29 and 24 fail initial SCC. Both complete panels have multiple contact components.
To diagnose whether the stochastic bridge damaged otherwise useful FM outputs,
sample the original electronic FM checkpoint on exactly this condition and the
same intrinsic Gaussian initial draws. Midpoint resolutions 16/64 cost 32/128
neural calls per sample; report coordinate differences and the same xTB subset.
No FM density is evaluated, so its importance weights/ESS remain unavailable.
Earlier eight-condition FM results are not a matched control for this condition.

Development manifests may now supply the work trainer's atomic numbers, total
charge and spin directly. No reference-coordinate dataset is loaded in that
branch, and reserved manifests are rejected for training. A two-update CPU
interface check on the first three-atom development condition (manifest row 167,
PbCl2 singlet) is not a training-benefit comparison. Its poor ESS is retained.

The same-condition FM control completed: midpoint-16/64 xTB convergence is
29/32 and 30/32, versus 0/32 for the reference-bridge initialization and 6/32
after 100 work updates. Successful-only FM median strain is 4.76794/4.67530 eV.
Each FM panel has one overlap among 64 geometries; the bridge panels have 8/11.
The maximum FM coordinate change from 16 to 64 steps is .66522 A, so this is not
a solver-convergence certificate. The control diagnoses initialization damage;
FM importance weights remain unavailable and these are not Boltzmann results.

This motivates a prospective native-mean parameterization. In the previous
bridge, inserting the pretrained velocity v into a reference residual gives
mean a_F*x+dt*v, with a_F containing the reference width expansion. That is not
the native FM Euler mean. Define the residual instead as

    r_F = v + (1-a_F)*x/dt,
    r_B = v_B + (1-a_B)*y/dt.

Without residual bounding, the resulting means are exactly x+dt*v and y+dt*v_B.
With the same smooth bound as before, they approximate native means centrally
and retain bounded far-tail residuals. Apply the bound to the entire residual,
including the fixed correction. Keep the same Gaussian noise variances and
evaluate the actual forward/backward densities. This is a parameterization and
initialization change using standard kernels, not a new work identity. Zero
neural output is no longer zero reference residual in native mode.

The first native-mean run is only a two-update calibration: batch two, 16 path
steps, 64 held-noise evaluation samples, original seed/checkpoint/state, kT=1 eV,
and restraint .1 eV/A2. Total potential budget is 132 including evaluation.
Check initial structure quality before longer training. Existing reference-mean
500-update controls continue unchanged. Tests verify exact unbounded native
means and density factors, full finite-difference/checkpoint gradients, and the
energy-only gradient ablation under both mean parameterizations.

The native-mean two-update preflight completed. Before training, xTB convergence
recovers to 30/32, but successful-only median strain is 16.7263 eV and overlap
incidence is 21/64. Weight ESS is 1.928/64. After two updates the corresponding
values are 28/32, 16.9588 eV, 20/64 and 1.781/64. This restores evaluator
convergence, not adequate sample quality or Boltzmann efficiency. Native mean
compensation is not a demonstrated final solution.

The next native-mean calibration mirrors the reference-mean 500-update protocol:
500 updates, batch 16, 16 transitions, 256 held-noise samples before/after,
seed 9051, learning rate 1e-5, batch-16 oracle inference. Compare joint work
gradients with forward-energy-only plus backward conditional likelihood. Every
arm uses 8,512 potential calls; native/reference changes no neural call count.
The same fixed 32 indices enter xTB assessment, and all 256 geometries enter
contact/diversity/weight diagnostics. Both older reference-mean runs continue.
No new temperature, target, checkpoint or electronic state is selected here.
