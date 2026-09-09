# A quantitative version of the empirical group-overlap statement

This is a standard graph-Laplacian argument and is **not claimed as a novel
theorem**. It sharpens the experimental design criterion without asserting
anything about unsampled configurations or the trained flow's convergence.

Let V contain n distinct evaluated geometries, with a shared composition,
and let S_m select the K_m geometries in group m. For positive weights pi_m,
define `C_m = I - 11^T/K_m` and

`L = sum_m (pi_m/K_m) S_m^T C_m S_m`.

For residual vector w, the grouped objective is exactly `w^T L w`.
L is positive semidefinite. Its null space consists of vectors constant
on each connected component of the graph joining geometries appearing
together in a group. Isolated geometries are separate components.

If the graph is connected and lambda_2(L)>0, write
`z=w-mean(w)1`. The Rayleigh quotient gives

`||z||_2^2 <= (w^T L w)/lambda_2(L)`.

Consequently, for every i,j,

`|w_i-w_j| <= sqrt(2 (w^T L w)/lambda_2(L))`.

Proof: `w_i-w_j=(e_i-e_j)^T z`, and `||e_i-e_j||_2=sqrt(2)`.

For *discrete normalised weights on this evaluated set*,
`q_i proportional exp(ell_i)` and `pi_i proportional exp(-E_i/kT)`,
`w_i-w_j = log(q_i/q_j)-log(pi_i/pi_j)`. The bound therefore controls
pairwise log-weight errors on the finite set. It does not bound continuous
basin probabilities without additional coverage/regularity assumptions.

Implication: adding bridge groups should be evaluated both for connectivity
and for spectral gap. A graph with a very weak bridge can be connected yet
give a poor small-loss stability bound. The existing disconnected groups
have a zero gap and cannot provide this finite-set guarantee.
