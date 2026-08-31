# Endpoint semantics of the likelihood integrator — settled from code and analytic tests

Date: 2026-08-18.  Tests: `tests/test_estimator_endpoint_semantics.py` (3 passed).
Implementation read: `cfm_mol/bgfm_density.py::log_density_via_flow` (lines 101–196).

## 1. The question

The integrator places its quadrature nodes at `t_k = 1 - (k + 1/2)/n`, so the largest
node is `1 - 1/(2n)`.  The manuscript read this as evidence that the estimator targets a
smoothed marginal `p_{1-eps}` with `eps = 1/(2n)`, and built the resolution-sensitivity
story on that reading.

## 2. What the code does

```
x   <- x_1
dt  <- 1/n
for k in 0..n-1:
    t_k <- 1 - (k + 1/2) dt          # midpoint of the k-th subinterval
    div <- divergence of v at (x, t_k)
    x   <- x - dt * v(x, t_k)        # reverse step, under no_grad
    I   <- I + dt * div
return log p_prior(x) - I
```

Two separate numerical schemes are in play, and the manuscript conflates them:

* **State**: explicit **Euler**, `x <- x - dt v(x, t_k)`.  `n` steps of size `1/n` traverse
  the whole of `[0,1]`.  Error is `O(dt)`.
* **Divergence quadrature**: the **midpoint rule** on `[0,1]`, `I ~ dt * sum_k f(t_k)`.
  All nodes are interior, but the integration interval is the whole of `[0,1]`.  Error is
  `O(dt^2)`, and the rule is *exact* for integrands linear in `t`.

So `Table A2`'s "reverse-time Euler" and Appendix A.2's "midpoint scheme" are each half
right.  The accurate description is **explicit Euler in the state with midpoint-rule
quadrature for the divergence integral**.

## 3. The tests, and what they decide

| test | integrand | full-interval prediction | truncated-`1-1/(2n)` prediction | measured |
|---|---|---|---|---|
| constant divergence, `v = a x` | `3Na` | `3Na` for every `n` | `3Na(1 - 1/(2n))` | `3.3300000000` at `n = 1, 2, 4, 12, 48, 100` |
| time-varying, `v = t x` | `3Nt` | `3N/2` for every `n` | `3N/2 (1-1/(2n))^2` | `4.5000000000` at every `n` (max dev `2.0e-9`) |
| linear CNF, `v = a x` | — | converges to the analytic `log p_1` | would converge to different values per `n` | `-12.229` , error `3.1e-1 -> 6.9e-4` for `n = 2 -> 800` |

The second test is the sharpest: at `n = 1` the truncated reading predicts `1.125` against a
measured `4.500`.  The third is the most direct: **different step counts converge to one
and the same endpoint quantity**, which is exactly what a family of distinct `p_{1-eps}`
marginals would not do.

## 4. Conclusion

**The estimator targets `p_1`.**  The `eps = 1/(2n)` smoothing-scale interpretation is
wrong and must be removed from the manuscript.  Step-count dependence is **numerical
discretisation error** — dominated by the `O(dt)` Euler state integration, not by the
`O(dt^2)` quadrature — together with whatever endpoint stiffness the learned field has.

### 4.1 No retraining is required

The correction changes the *interpretation* of the step-count dependence, not the quantity
being computed.  Training at `n = 4` and evaluation at `n = 12` targeted the same `log p_1`
throughout; they differ in how much Euler bias they carry (test 2: `1.5e-1` at `n = 4`
against `4.7e-2` at `n = 12` on the analytic problem).  Nothing in the training objective
changes meaning, so the brief's retraining branch is not triggered.

### 4.2 The honest consequence for the resolution ablation

Under the corrected reading the resolution rows say something less comfortable than the
smoothing-scale story did: the measured gap is **largest where the estimator is least
accurate** (`+0.382` at `n = 4`) and **shrinks as the estimator becomes more accurate**
(`+0.177` at `n = 12`).  That ordering is consistent with part of the effect at coarse
resolution being discretisation artefact.  The headline is computed at `n = 12`; the paper
must state the trend and the fact that it cannot separate a genuine scale dependence of the
learned density from solver bias.

## 5. A second finding, from the same read

The backward pass is a **truncated-adjoint / frozen-trajectory surrogate**, not exact
likelihood-gradient descent:

* `x_req = x.detach().requires_grad_(True)` — every step is cut from the previous one;
* the state update runs under `torch.no_grad()`, so the whole reverse trajectory carries no
  gradient;
* `log p_prior(x)` is therefore a constant with respect to the parameters;
* gradient flows **only** through the divergence evaluations.

The manuscript must say so in the main text, and the theory must be scoped to the zero set
and identifiability of the ideal residual condition rather than to the convergence of this
implemented update.

## 6. A third finding, already documented in the source

`make_position_velocity_fn`'s own docstring: "The discrete channels (a_t, c_t, e_t) are held
at their values already set on g_template (data labels).  Only x_t varies."  The measured
object is a **clamped positional-flow density**, not the joint generator's conditional
density.  This confirms the notation repair the revision brief requires.
