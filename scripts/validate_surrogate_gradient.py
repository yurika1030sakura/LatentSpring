"""P1 numerical validation: full-gradient vs frozen-trajectory surrogate gradient.

BACKGROUND (FINDING 2, already settled from code)
-------------------------------------------------
`cfm_mol.bgfm_density.log_density_via_flow` computes the reverse-time
likelihood with

    x_req = x.detach().requires_grad_(True)      # cut from previous step
    ...divergence evaluated at x_req...
    with torch.no_grad():  x = x - dt * v(x, t)  # state update carries NO grad

so during training the gradient reaches the parameters ONLY through the
divergence evaluations.  The reverse trajectory is frozen and
`log p_prior(x_0)` is a constant in the parameters.  The implemented update
is therefore a truncated-adjoint / frozen-trajectory SURROGATE, not the
gradient of the ODE likelihood.

WHAT THIS SCRIPT MEASURES
-------------------------
On a small model and small batch we compute, for the SAME batch, the SAME
Hutchinson probes and the SAME parameters:

  (a) g_full  -- gradient of a fully differentiable ODE likelihood: the
      trajectory is built WITH gradient (no detach, no no_grad) and the
      divergence is taken with create_graph=True, so the prior endpoint and
      every state along the reverse path contribute.
  (b) g_surr  -- the gradient the implemented surrogate actually produces
      (`log_density_via_flow(..., for_training=True)`, unmodified).

Because the two paths perform IDENTICAL arithmetic on the values (detaching
changes only the graph, never the number), log p is bit-comparable between
them; the script asserts this, so any difference reported is purely a
gradient-pathway difference.

Reported per objective:
  * cosine similarity cos(g_full, g_surr) over all parameters
  * relative norm ||g_surr|| / ||g_full||
  * per-parameter-tensor ("per-layer") cosine and norm-ratio distribution
  * an exact additive decomposition
        g_full = g_surr + g_traj + g_prior
    where g_prior is the path through log p_prior(x_0) and g_traj is the path
    through the dependence of later divergence evaluations on earlier states
  * whether -g_surr is a descent direction for the true objective:
    first-order test sign(<g_full, g_surr>) plus a finite-difference probe of
    the true objective along -g_surr / ||g_surr||.

Two objectives are used:
  L_energy : the ACTUAL training loss (within-parent variance of
             log p + E/kT, `within_group_variance_loss`), and
  L_rank   : a smooth within-parent ORDERING objective (pairwise logistic
             loss for "lower energy must get higher log-density"), the
             differentiable stand-in for the paper's within-parent rank
             endpoint.

Two model arms:
  --arm synthetic : a small parametric O(3)-equivariant message-passing
                    velocity field (exact divergence also available).
  --arm real      : the real FlowMol3 vector field from a completed
                    checkpoint, on real precomputed perturbation batches
                    with real OMol25 energies (needs a GPU).

NO TRAINING is performed: parameters are never updated.  The finite-
difference probe evaluates the objective at displaced parameters and always
restores the original values.

Reproduce:
    source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
    conda activate /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
    cd /n/home04/yulili/bgfm
    python -u scripts/validate_surrogate_gradient.py --arm synthetic
    python -u scripts/validate_surrogate_gradient.py --arm real \
        --checkpoint <ckpt> --config configs/sweep/p0_D_energy_s1.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfm_mol.bgfm_density import (  # noqa: E402
    gaussian_prior_log_density,
    log_density_via_flow,
    within_group_variance_loss,
    _n_atoms_per_graph,
)
from cfm_mol.bgfm_loss import (  # noqa: E402
    divergence_exact_atomwise,
    divergence_hutchinson,
)


# ---------------------------------------------------------------------------
# 1. Fully differentiable reference likelihood
# ---------------------------------------------------------------------------
def log_density_full_gradient(
    model,
    g_aux,
    node_batch_idx,
    upper_edge_mask,
    n_ode_steps: int = 4,
    n_hutchinson: int = 1,
    prior_std: float = 1.0,
    kT=None,
    xi_fn=None,
    detach_prior: bool = False,
    detach_trajectory: bool = False,
):
    """Same numerics as `log_density_via_flow`, but the reverse trajectory is
    differentiable (no `.detach()`, no `torch.no_grad()`), so the gradient
    includes the prior endpoint and every earlier state.

    detach_prior / detach_trajectory switch off one path at a time, which
    gives the exact additive decomposition of g_full used in the report:
        full                -> g_full
        detach_prior=True   -> g_full - g_prior            (= g_surr + g_traj)
        both detached       -> g_surr  (matches the implementation)
    """
    from cfm_mol.bgfm_density import make_position_velocity_fn

    device = g_aux.device
    B = g_aux.batch_size
    x = g_aux.ndata['x_1_true'].detach().clone().requires_grad_(True)

    dt = 1.0 / n_ode_steps
    logp_integral = torch.zeros(B, device=device, dtype=x.dtype)
    n_apg = _n_atoms_per_graph(g_aux, node_batch_idx)

    for step in range(n_ode_steps):
        t_val = 1.0 - (step + 0.5) * dt
        t_scalar = torch.full((B,), t_val, device=device, dtype=torch.float32)

        x_in = x.detach().requires_grad_(True) if detach_trajectory else x
        if not x_in.requires_grad:
            x_in = x_in.requires_grad_(True)

        v_fn = make_position_velocity_fn(
            model, g_aux, t_scalar, node_batch_idx, upper_edge_mask, kT=kT)
        if n_hutchinson <= 0:
            div = divergence_exact_atomwise(v_fn, x_in, n_apg, create_graph=True)
        else:
            div = divergence_hutchinson(
                v_fn, x_in, n_apg, n_samples=n_hutchinson, rademacher=True,
                create_graph=True,
                xi_provider=(None if xi_fn is None
                             else (lambda k, xx, _s=step: xi_fn(_s, k, xx))))

        # State update -- WITH gradient (this is the whole point).
        g_aux.ndata['x_t'] = x
        vf_kwargs = dict(node_batch_idx=node_batch_idx,
                         upper_edge_mask=upper_edge_mask)
        if kT is not None:
            vf_kwargs["kT"] = kT
        v = model.vector_field(g_aux, t_scalar, **vf_kwargs)['x']
        x = x - dt * v

        logp_integral = logp_integral + dt * div

    x_prior = x.detach() if detach_prior else x
    logp0 = gaussian_prior_log_density(x_prior, n_apg, prior_std=prior_std)
    return logp0 - logp_integral


# ---------------------------------------------------------------------------
# 2. Objectives
# ---------------------------------------------------------------------------
def loss_energy(log_p, energies, parent_id, kT=1.0):
    """The ACTUAL training loss: within-parent Var(log p + E/kT)."""
    residual = log_p + energies / float(kT)
    loss, _ = within_group_variance_loss(residual, parent_id)
    return loss


def loss_rank(log_p, energies, parent_id, tau=1.0):
    """Smooth within-parent ORDERING objective.

    For every within-parent pair (i, j) with E_i < E_j the lower-energy
    geometry must get the higher log-density.  Pairwise logistic loss
    softplus(-(log p_i - log p_j) * sign(E_j - E_i) / tau), averaged.
    This is the differentiable stand-in for the paper's within-parent
    Spearman endpoint (rank correlation itself has zero gradient a.e.).
    """
    terms = []
    for m in parent_id.unique():
        sel = (parent_id == m).nonzero(as_tuple=True)[0]
        if sel.numel() < 2:
            continue
        lp = log_p[sel]
        en = energies[sel]
        d_lp = lp[:, None] - lp[None, :]
        s = torch.sign(en[None, :] - en[:, None])
        iu = torch.triu_indices(sel.numel(), sel.numel(), offset=1)
        terms.append(torch.nn.functional.softplus(
            -(d_lp[iu[0], iu[1]] * s[iu[0], iu[1]]) / tau).mean())
    if not terms:
        return log_p.sum() * 0.0
    return torch.stack(terms).mean()


OBJECTIVES = {"energy": loss_energy, "rank": loss_rank}


# ---------------------------------------------------------------------------
# 3. Gradient utilities
# ---------------------------------------------------------------------------
def grad_dict(loss, named_params):
    params = [p for _, p in named_params]
    grads = torch.autograd.grad(loss, params, allow_unused=True,
                                retain_graph=False)
    out = {}
    for (name, p), g in zip(named_params, grads):
        out[name] = torch.zeros_like(p) if g is None else g.detach().clone()
    return out


def flat(gd, names):
    return torch.cat([gd[n].reshape(-1).double() for n in names])


def compare(g_a, g_b, names):
    fa, fb = flat(g_a, names), flat(g_b, names)
    na, nb = fa.norm().item(), fb.norm().item()
    cos = float((fa @ fb / (fa.norm() * fb.norm() + 1e-300)).item()) \
        if na > 0 and nb > 0 else float("nan")
    per, degenerate, zero_in_b = {}, [], []
    for n in names:
        a, b = g_a[n].reshape(-1).double(), g_b[n].reshape(-1).double()
        if a.norm() == 0:
            continue
        if b.norm() == 0:
            zero_in_b.append(n)
            continue
        rec = dict(cos=float((a @ b / (a.norm() * b.norm())).item()),
                   ratio=float((b.norm() / a.norm()).item()),
                   numel=int(a.numel()),
                   n_a=float(a.norm().item()), n_b=float(b.norm().item()))
        # Tensors with <4 entries give a cosine that is nearly +/-1 by
        # construction; keep them out of the distribution and count them.
        if a.numel() >= 4:
            per[n] = rec
        else:
            degenerate.append(n)
    return dict(cos=cos, norm_a=na, norm_b=nb,
                rel_norm=(nb / na if na > 0 else float("nan")), per_layer=per,
                n_degenerate_tensors=len(degenerate),
                n_tensors_zero_in_b=len(zero_in_b))


def quantiles(vals):
    if not vals:
        return {}
    t = torch.tensor(sorted(vals), dtype=torch.float64)
    q = lambda f: float(torch.quantile(t, f).item())  # noqa: E731
    return dict(min=float(t[0]), p10=q(0.10), median=q(0.5), p90=q(0.90),
                max=float(t[-1]), n=len(vals))


# ---------------------------------------------------------------------------
# 4. Probe bank (common random numbers across the two implementations)
# ---------------------------------------------------------------------------
def make_xi_bank(shape, n_steps, n_h, seed, device, dtype):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bank = {}
    for s in range(n_steps):
        for k in range(max(n_h, 1)):
            bank[(s, k)] = ((torch.randint(0, 2, shape, generator=gen).to(
                device=device, dtype=dtype)) * 2 - 1)

    def xi_fn(step, k, x):
        return bank[(step, k)].to(device=x.device, dtype=x.dtype)
    return xi_fn


# ---------------------------------------------------------------------------
# 5. Synthetic arm: small parametric equivariant velocity field
# ---------------------------------------------------------------------------
class ToyVectorField(nn.Module):
    """v_i = sum_{j != i, same graph} w(d_ij, t) (x_i - x_j) + a(t) * x_i.

    O(3)-equivariant, nonlinear in x through the distance-dependent weights,
    time dependent, and with a nontrivial (state-dependent) divergence.
    ~5k parameters spread over 6 weight tensors so per-layer variation is
    measurable.
    """

    def __init__(self, hidden=32, n_rbf=8, d_max=6.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_rbf, self.d_max = n_rbf, d_max
        self.edge = nn.Sequential(
            nn.Linear(n_rbf + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        self.selfterm = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.35)
                nn.init.normal_(m.bias, std=0.10)

    def _tfeat(self, t, n, device, dtype):
        t = torch.as_tensor(t, device=device, dtype=dtype)
        return torch.stack([t, t ** 2, torch.sin(3 * t), torch.cos(3 * t)]
                           ).reshape(1, 4).expand(n, 4)

    def forward(self, g, t_scalar, node_batch_idx=None, upper_edge_mask=None,
                **kw):
        x = g.ndata['x_t']
        dev, dt_ = x.device, x.dtype
        t = float(t_scalar.reshape(-1)[0])
        N = x.shape[0]
        diff = x[:, None, :] - x[None, :, :]                 # (N,N,3)
        d2 = (diff ** 2).sum(-1) + 1e-8
        d = d2.sqrt()
        same = (node_batch_idx[:, None] == node_batch_idx[None, :])
        mask = same & ~torch.eye(N, dtype=torch.bool, device=dev)
        centers = torch.linspace(0., self.d_max, self.n_rbf, device=dev,
                                 dtype=dt_)
        rbf = torch.exp(-((d[..., None] - centers) ** 2) / 0.6)
        tf = self._tfeat(t, N * N, dev, dt_).reshape(N, N, 4)
        w = self.edge(torch.cat([rbf, tf], dim=-1)).squeeze(-1)  # (N,N)
        w = w * mask.to(dt_) / (1.0 + d2)
        v = (w[..., None] * diff).sum(dim=1)
        a = self.selfterm(self._tfeat(t, 1, dev, dt_)).reshape(())
        return {"x": v + a * x}


class ToyModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.vector_field = ToyVectorField(**kw)


class StubGraph:
    """Minimal stand-in for the batched DGL graph used by the estimator."""

    def __init__(self, x1, batch_size):
        self._ndata = {"x_1_true": x1.clone(), "x_t": x1.clone()}
        self.batch_size = batch_size
        self.device = x1.device

    @property
    def ndata(self):
        return self._ndata


def synthetic_batch(M=3, K=4, n_atoms=6, sigma=0.12, seed=0, dtype=torch.float64,
                    device="cpu"):
    """M parent geometries x K perturbations, plus a synthetic energy."""
    gen = torch.Generator().manual_seed(seed)
    xs, energies, parent = [], [], []
    for m in range(M):
        base = torch.randn(n_atoms, 3, generator=gen, dtype=dtype) * 1.1
        base = base - base.mean(0, keepdim=True)
        for k in range(K):
            p = base + torch.randn(n_atoms, 3, generator=gen, dtype=dtype) * sigma
            p = p - p.mean(0, keepdim=True)
            xs.append(p)
            d = torch.cdist(p, p) + torch.eye(n_atoms, dtype=dtype) * 1e3
            e = ((1.0 / d.clamp(min=0.6) ** 12) - 2.0 / d.clamp(min=0.6) ** 6)
            energies.append(0.5 * e[torch.triu(torch.ones_like(e),
                                               diagonal=1) > 0].sum())
            parent.append(m)
    x1 = torch.cat(xs, 0).to(device)
    nbi = torch.arange(M * K).repeat_interleave(n_atoms).to(device)
    g = StubGraph(x1, M * K)
    return (g, nbi, None,
            torch.stack(energies).to(device=device, dtype=dtype),
            torch.tensor(parent, device=device))


# ---------------------------------------------------------------------------
# 6. Core measurement
# ---------------------------------------------------------------------------
def measure(model, g, nbi, uem, energies, parent_id, *, n_ode_steps,
            n_hutchinson, kT, seed, objectives, fd_eps, verbose=True):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    x1 = g.ndata['x_1_true']
    xi_fn = make_xi_bank(tuple(x1.shape), n_ode_steps, n_hutchinson, seed,
                         x1.device, x1.dtype)

    def logp(variant):
        g.ndata['x_t'] = g.ndata['x_1_true'].clone()
        if variant == "surrogate":
            return log_density_via_flow(
                model, g, nbi, uem, n_ode_steps=n_ode_steps,
                n_hutchinson=n_hutchinson, prior_std=1.0, for_training=True,
                xi_fn=xi_fn)
        kw = dict(detach_prior=(variant == "no_prior"))
        return log_density_full_gradient(
            model, g, nbi, uem, n_ode_steps=n_ode_steps,
            n_hutchinson=n_hutchinson, prior_std=1.0, xi_fn=xi_fn, **kw)

    cost = {}
    lp = {}
    for v in ("full", "no_prior", "surrogate"):
        if x1.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        _lp = logp(v)
        lp[v] = _lp.detach().clone()
        del _lp          # free this variant's graph before building the next
        cost[v] = dict(
            fwd_seconds=time.time() - t0,
            cuda_peak_MB=(torch.cuda.max_memory_allocated() / 2 ** 20
                          if x1.is_cuda else None),
            proc_max_rss_MB=max(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                rss0))
    val = {v: lp[v].detach().double() for v in lp}
    max_val_dev = max(float((val[v] - val["full"]).abs().max().item())
                      for v in val)

    out = {"cost": cost,
           "logp_max_value_deviation": max_val_dev,
           "logp_mean": float(val["full"].mean().item()),
           "logp_std": float(val["full"].std().item()),
           "objectives": {}}
    if verbose:
        print(f"    log p: mean={out['logp_mean']:.4f} std={out['logp_std']:.4f} "
              f"| max |value(full)-value(variant)| over the three graph "
              f"variants = {max_val_dev:.3e}", flush=True)

    for oname in objectives:
        fn = OBJECTIVES[oname]
        losses, grads = {}, {}
        for v in ("full", "no_prior", "surrogate"):
            # recompute the graph fresh for each backward
            lpv = logp(v)
            L = fn(lpv, energies, parent_id) if oname != "energy" else \
                fn(lpv, energies, parent_id, kT=kT)
            losses[v] = float(L.detach().item())
            grads[v] = grad_dict(L, named)

        g_full, g_surr = grads["full"], grads["surrogate"]
        g_prior = {n: g_full[n] - grads["no_prior"][n] for n in names}
        g_traj = {n: grads["no_prior"][n] - g_surr[n] for n in names}

        cmp_main = compare(g_full, g_surr, names)
        cmp_prior = compare(g_full, g_prior, names)
        cmp_traj = compare(g_full, g_traj, names)

        f_full, f_surr = flat(g_full, names), flat(g_surr, names)
        dot = float((f_full @ f_surr).item())
        dirderiv = -dot / (f_surr.norm().item() + 1e-300)   # dL/ds along -ghat

        D = int(sum(p.numel() for _, p in named))
        rec = dict(
            n_params=D,
            cos_random_direction_reference=math.sqrt(2.0 / (math.pi * D)),
            loss=losses["full"],
            loss_agreement=max(abs(losses[v] - losses["full"]) for v in losses),
            cos_full_surr=cmp_main["cos"],
            rel_norm_surr_over_full=cmp_main["rel_norm"],
            norm_full=cmp_main["norm_a"], norm_surr=cmp_main["norm_b"],
            norm_prior_path=cmp_prior["norm_b"],
            norm_traj_path=cmp_traj["norm_b"],
            cos_full_prior=cmp_prior["cos"], cos_full_traj=cmp_traj["cos"],
            per_layer_cos=quantiles([d["cos"] for d in
                                     cmp_main["per_layer"].values()]),
            per_layer_ratio=quantiles([d["ratio"] for d in
                                       cmp_main["per_layer"].values()]),
            descent_first_order=("descent" if dot > 0 else
                                 ("ascent" if dot < 0 else "orthogonal")),
            dot_full_surr=dot,
            directional_derivative_along_neg_surr=dirderiv,
        )
        if len(names) <= 12:
            rec["per_layer_table"] = cmp_main["per_layer"]
        # Which parameter tensors carry the disagreement?  Rank by share of
        # ||g_full||^2 and report the 8 heaviest with their own cosines.
        pl = cmp_main["per_layer"]
        if pl:
            heavy = sorted(pl.items(), key=lambda kv: -kv[1]["n_a"])[:8]
            rec["heaviest_tensors"] = [
                dict(name=k, share_of_norm_full=v["n_a"] / cmp_main["norm_a"],
                     cos=v["cos"], ratio=v["ratio"], numel=v["numel"])
                for k, v in heavy]
            worst = sorted(pl.items(), key=lambda kv: kv[1]["cos"])[:5]
            rec["lowest_cos_tensors"] = [
                dict(name=k, cos=v["cos"], ratio=v["ratio"],
                     share_of_norm_full=v["n_a"] / cmp_main["norm_a"],
                     numel=v["numel"]) for k, v in worst]
            rec["n_tensors_negative_cos"] = sum(
                1 for v in pl.values() if v["cos"] < 0)
            rec["n_tensors_scored"] = len(pl)

        # --- finite-difference probe of the TRUE objective along -g_surr ---
        if fd_eps:
            rec["finite_difference"] = fd_probe(
                model, named, g_surr, g_full, fn, oname, kT,
                lambda: logp("surrogate"),  # values identical to full
                energies, parent_id, fd_eps, losses["full"])
        out["objectives"][oname] = rec
        if verbose:
            print(f"    [{oname}] cos={rec['cos_full_surr']:+.4f} "
                  f"||surr||/||full||={rec['rel_norm_surr_over_full']:.4f} "
                  f"{rec['descent_first_order']}", flush=True)
    return out


def fd_probe(model, named, g_surr, g_full, fn, oname, kT, logp_call,
             energies, parent_id, eps_list, L0):
    """Evaluate the TRUE objective at theta - eps * dhat for dhat = the
    normalised surrogate direction and, for reference, the normalised full
    gradient.  Parameters are always restored."""
    names = [n for n, _ in named]
    theta_norm = math.sqrt(sum(float((p.detach() ** 2).sum()) for _, p in named))
    res = {"theta_norm": theta_norm, "L0": L0, "steps": []}
    for tag, gd in (("surrogate", g_surr), ("full", g_full)):
        fnorm = flat(gd, names).norm().item()
        if fnorm == 0:
            continue
        for eps_rel in eps_list:
            step = eps_rel * theta_norm / fnorm
            with torch.no_grad():
                for n, p in named:
                    p -= step * gd[n]
            try:
                with torch.enable_grad():
                    lp = logp_call()
                L = fn(lp, energies, parent_id, kT=kT) if oname == "energy" \
                    else fn(lp, energies, parent_id)
                Lnew = float(L.detach().item())
            finally:
                with torch.no_grad():
                    for n, p in named:
                        p += step * gd[n]
            res["steps"].append(dict(direction=tag, eps_rel=eps_rel,
                                     step_len=step, L_new=Lnew,
                                     delta_L=Lnew - L0))
    return res


# ---------------------------------------------------------------------------
# 6b. Self-checks: is the reference implementation trustworthy?
# ---------------------------------------------------------------------------
def selfcheck(args):
    """Two validations of the measurement apparatus itself.

    A. Reference-matches-implementation: `log_density_full_gradient` with BOTH
       paths detached must reproduce, to machine precision, the gradient of
       the unmodified `log_density_via_flow(..., for_training=True)`.  If it
       does, the only difference in the main experiment is the detaching.
    B. Full gradient is the true gradient: central finite differences of the
       objective along a random parameter direction must match
       <g_full, direction>.  Uses float64 + exact divergence.
    """
    model = ToyModel(hidden=args.hidden, seed=7).to(dtype=torch.float64)
    g, nbi, uem, E, pid = synthetic_batch(M=2, K=3, n_atoms=4, seed=11,
                                          dtype=torch.float64)
    named = [(n, p) for n, p in model.named_parameters()]
    names = [n for n, _ in named]
    n_steps, n_h = 4, 0
    xi_fn = make_xi_bank(tuple(g.ndata['x_1_true'].shape), n_steps, n_h, 5,
                         g.ndata['x_1_true'].device, torch.float64)

    def L_of(variant):
        g.ndata['x_t'] = g.ndata['x_1_true'].clone()
        if variant == "impl":
            lp = log_density_via_flow(
                model, g, nbi, uem, n_ode_steps=n_steps, n_hutchinson=n_h,
                prior_std=1.0, for_training=True, xi_fn=xi_fn)
        else:
            lp = log_density_full_gradient(
                model, g, nbi, uem, n_ode_steps=n_steps, n_hutchinson=n_h,
                prior_std=1.0, xi_fn=xi_fn,
                detach_prior=(variant == "ref_surr"),
                detach_trajectory=(variant == "ref_surr"))
        return loss_energy(lp, E, pid, kT=args.kT), lp

    L_impl, _ = L_of("impl")
    g_impl = grad_dict(L_impl, named)
    L_ref, _ = L_of("ref_surr")
    g_ref = grad_dict(L_ref, named)
    c = compare(g_impl, g_ref, names)
    print(f"  [A] loss impl={float(L_impl):.12f} ref-surrogate={float(L_ref):.12f}")
    print(f"  [A] cos(g_impl, g_ref_surrogate) = {c['cos']:.12f}   "
          f"||ref||/||impl|| = {c['rel_norm']:.12f}   "
          f"max abs diff = "
          f"{max(float((g_impl[n]-g_ref[n]).abs().max()) for n in names):.3e}")

    L_full, _ = L_of("full")
    g_full = grad_dict(L_full, named)
    torch.manual_seed(3)
    d = {n: torch.randn_like(p) for n, p in named}
    dn = flat(d, names).norm()
    d = {n: v / dn for n, v in d.items()}
    pred = float(flat(g_full, names) @ flat(d, names))
    print(f"  [B] predicted directional derivative <g_full, d> = {pred:+.10f}")
    for h in (1e-4, 1e-5, 1e-6):
        with torch.no_grad():
            for n, p in named:
                p += h * d[n]
        Lp = float(L_of("impl")[0])
        with torch.no_grad():
            for n, p in named:
                p -= 2 * h * d[n]
        Lm = float(L_of("impl")[0])
        with torch.no_grad():
            for n, p in named:
                p += h * d[n]
        fd = (Lp - Lm) / (2 * h)
        print(f"  [B] h={h:.0e}  central FD = {fd:+.10f}   "
              f"rel err = {abs(fd - pred) / (abs(pred) + 1e-30):.3e}")
    return 0


# ---------------------------------------------------------------------------
# 6c. Cost probe: where the fully differentiable reference breaks
# ---------------------------------------------------------------------------
def memprobe(args):
    """Run ONE variant in this process and report wall time + peak RSS.

    Peak RSS (ru_maxrss) is a process high-water mark, so a clean number
    requires one variant per process -- hence this mode.  Compare
    `--probe_variant full` against `--probe_variant surrogate` at identical
    settings to get the cost of making the trajectory differentiable.
    """
    model = ToyModel(hidden=args.hidden, seed=100).to(dtype=torch.float64)
    g, nbi, uem, E, pid = synthetic_batch(
        M=args.M, K=args.K, n_atoms=args.n_atoms, seed=200,
        dtype=torch.float64)
    n_steps = args.n_ode_steps[0]
    n_h = args.n_hutchinson[0]
    xi_fn = make_xi_bank(tuple(g.ndata['x_1_true'].shape), n_steps, n_h, 300,
                         g.ndata['x_1_true'].device, torch.float64)
    named = [(n, p) for n, p in model.named_parameters()]
    t0 = time.time()
    g.ndata['x_t'] = g.ndata['x_1_true'].clone()
    if args.probe_variant == "surrogate":
        lp = log_density_via_flow(
            model, g, nbi, uem, n_ode_steps=n_steps, n_hutchinson=n_h,
            prior_std=1.0, for_training=True, xi_fn=xi_fn)
    else:
        lp = log_density_full_gradient(
            model, g, nbi, uem, n_ode_steps=n_steps, n_hutchinson=n_h,
            prior_std=1.0, xi_fn=xi_fn)
    t_fwd = time.time() - t0
    L = loss_energy(lp, E, pid, kT=args.kT)
    t1 = time.time()
    grad_dict(L, named)
    t_bwd = time.time() - t1
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    n_at = int(g.ndata['x_1_true'].shape[0])
    print(f"MEMPROBE variant={args.probe_variant} n_ode={n_steps} "
          f"hutch={n_h if n_h > 0 else 'EXACT'} atoms={n_at} "
          f"fwd_s={t_fwd:.2f} bwd_s={t_bwd:.2f} peak_rss_MB={rss:.0f} "
          f"loss={float(L):.6g}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 7. Arms
# ---------------------------------------------------------------------------
def run_synthetic(args):
    torch.manual_seed(0)
    results = []
    for rep in range(args.n_reps):
        model = ToyModel(hidden=args.hidden, seed=100 + rep).to(
            dtype=torch.float64)
        g, nbi, uem, E, pid = synthetic_batch(
            M=args.M, K=args.K, n_atoms=args.n_atoms, seed=200 + rep,
            dtype=torch.float64)
        for n_steps in args.n_ode_steps:
            for n_h in args.n_hutchinson:
                tag = f"rep{rep}_n{n_steps}_h{n_h}"
                print(f"  {tag} (hutchinson={n_h if n_h > 0 else 'EXACT'})",
                      flush=True)
                r = measure(model, g, nbi, uem, E, pid,
                            n_ode_steps=n_steps, n_hutchinson=n_h, kT=args.kT,
                            seed=300 + rep, objectives=args.objectives,
                            fd_eps=(args.fd_eps if (rep == 0 or args.fd_all_reps)
                                    else []))
                r.update(rep=rep, n_ode_steps=n_steps, n_hutchinson=n_h,
                         arm="synthetic")
                results.append(r)
    return results


def run_real(args):
    import dgl  # noqa: F401
    from flowmol.model_utils.load import model_from_config, read_config_file
    from cfm_mol.perturbation_loader import PerturbationLoader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = read_config_file(Path(args.config))
    bgfm_cfg = cfg.get("mol_fm", {}).pop("bgfm", {})
    atom_map = cfg["dataset"]["atom_map"]
    model = model_from_config(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded ckpt missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    shards = args.shards or bgfm_cfg.get("energy_perturbation_shards", [])
    # Loader constructed exactly as in cfm_mol/bgfm_train_hook.py.
    loader = PerturbationLoader(
        shard_paths=shards,
        n_atom_types=int(getattr(model, "n_atom_types", len(atom_map))),
        n_extra_atom_classes=int(bgfm_cfg.get("n_extra_atom_classes", 0)),
        n_charge_classes=6,
        n_bond_types=int(bgfm_cfg.get("n_bond_types", 4)),
        b_parents=args.M, device=device, seed=args.seed,
        max_atoms_per_parent=args.max_atoms)

    results = []
    for rep in range(args.n_reps):
        g, E, pid, nbi, uem = loader.next_batch()
        n_at = int(g.num_nodes())
        print(f"  rep{rep}: {g.batch_size} virtual mols, {n_at} atoms total",
              flush=True)
        for n_steps in args.n_ode_steps:
            for n_h in args.n_hutchinson:
                print(f"  rep{rep} n_ode={n_steps} hutch={n_h}", flush=True)
                r = measure(model, g, nbi, uem, E, pid,
                            n_ode_steps=n_steps, n_hutchinson=n_h, kT=args.kT,
                            seed=300 + rep, objectives=args.objectives,
                            fd_eps=(args.fd_eps if (rep == 0 or args.fd_all_reps)
                                    else []))
                r.update(rep=rep, n_ode_steps=n_steps, n_hutchinson=n_h,
                         arm="real", n_atoms_total=n_at,
                         n_virtual_mols=int(g.batch_size))
                results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--n_ode_steps", type=int, nargs="+", default=[1, 2, 4, 12])
    ap.add_argument("--n_hutchinson", type=int, nargs="+", default=[0, 2],
                    help="0 = exact divergence")
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--M", type=int, default=3, help="parent molecules")
    ap.add_argument("--K", type=int, default=4, help="perturbations per parent")
    ap.add_argument("--n_atoms", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--kT", type=float, default=1.0)
    ap.add_argument("--objectives", nargs="+", default=["energy", "rank"])
    ap.add_argument("--fd_eps", type=float, nargs="*",
                    default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--shards", type=str, nargs="*", default=None)
    ap.add_argument("--max_atoms", type=int, default=14)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--fd_all_reps", action="store_true",
                    help="run the finite-difference probe for every rep")
    ap.add_argument("--memprobe", action="store_true",
                    help="cost probe: one variant per process")
    ap.add_argument("--probe_variant", choices=["full", "surrogate"],
                    default="full")
    ap.add_argument("--selfcheck", action="store_true",
                    help="validate the measurement apparatus and exit")
    args = ap.parse_args()

    if args.memprobe:
        return memprobe(args)

    if args.selfcheck:
        print("=== SELF-CHECK ===", flush=True)
        return selfcheck(args)

    print(f"=== arm={args.arm} ===", flush=True)
    results = run_synthetic(args) if args.arm == "synthetic" else run_real(args)

    print("\n=== SUMMARY ===")
    hdr = (f"{'arm':<10}{'n_ode':>6}{'hutch':>6}{'obj':>8}{'cos':>10}"
           f"{'|surr|/|full|':>15}{'|prior|/|full|':>15}{'|traj|/|full|':>14}"
           f"{'descent':>10}")
    print(hdr)
    for r in results:
        for o, d in r["objectives"].items():
            nf = d["norm_full"]
            print(f"{r['arm']:<10}{r['n_ode_steps']:>6}{r['n_hutchinson']:>6}"
                  f"{o:>8}{d['cos_full_surr']:>10.4f}"
                  f"{d['rel_norm_surr_over_full']:>15.4f}"
                  f"{d['norm_prior_path'] / nf:>15.4f}"
                  f"{d['norm_traj_path'] / nf:>14.4f}"
                  f"{d['descent_first_order']:>10}")

    print("\n=== AGGREGATE OVER REPS (mean +/- sd; frac cos>0) ===")
    agg = {}
    for r in results:
        for o, d in r["objectives"].items():
            agg.setdefault((r["n_ode_steps"], r["n_hutchinson"], o), []).append(d)
    print(f"{'n_ode':>6}{'hutch':>6}{'obj':>8}{'cos mean':>11}{'cos sd':>9}"
          f"{'cos min':>9}{'frac>0':>8}{'relnorm mean':>14}{'cos_random':>12}")
    for (n_s, n_h, o), ds in sorted(agg.items()):
        c = torch.tensor([d["cos_full_surr"] for d in ds], dtype=torch.float64)
        rn = torch.tensor([d["rel_norm_surr_over_full"] for d in ds],
                          dtype=torch.float64)
        sd = float(c.std()) if c.numel() > 1 else 0.0
        print(f"{n_s:>6}{n_h:>6}{o:>8}{float(c.mean()):>11.4f}{sd:>9.4f}"
              f"{float(c.min()):>9.4f}{float((c > 0).double().mean()):>8.2f}"
              f"{float(rn.mean()):>14.4f}"
              f"{ds[0]['cos_random_direction_reference']:>12.2e}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"args": vars(args), "results": results}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
