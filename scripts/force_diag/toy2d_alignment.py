"""2D toy: does endpoint-force supervision hurt only when p_data != pi?

Hypothesis 2 says the force loss and the flow-matching loss have different
population targets whenever the corpus is not drawn from the Boltzmann law the
force refers to.  This is the cheapest causal test of that: keep the model, the
objective and the metric fixed, and change ONLY whether the data distribution
equals the Boltzmann distribution of the energy the force comes from.

Energy            E(x) = 0.5 x^T A x,  so  F(x) = -A x  and  pi = N(0, kT A^-1).
Aligned setting   p_data = pi                      -- force and FM agree.
Misaligned        p_data = N(mu, kT A^-1), mu != 0  -- same shape, wrong location,
                  which is the toy analogue of a corpus that is not sampled from
                  the target law.

Both settings train FM-only and FM+force with identical seeds, budget and
architecture.  We report the paper's own endpoint -- the within-group
correlation between log q_theta and -E/kT on held-out perturbation clouds --
plus the cosine between the two gradients.

No GPU, no teacher, no molecular data.  Runs in about a minute on CPU.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
import torch
import torch.nn as nn

KT = 1.0
A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
A_INV = torch.linalg.inv(A)
QUARTIC = 0.30      # anharmonic term: pi is no longer Gaussian, so a well-trained
                    # FM model on pi does NOT trivially saturate the ordering metric


def energy(x):                       # (N,2) -> (N,)
    return 0.5 * (x @ A * x).sum(-1) + QUARTIC * (x ** 4).sum(-1)


def force(x):                        # -grad E
    return -(x @ A) - 4.0 * QUARTIC * x ** 3


def sample_data(n, misaligned, g, burn=200, step=0.35):
    """Draw from pi ∝ exp(-E/kT) by Metropolis, so 'aligned' really is pi.
    Misaligned shifts the sample, which is the toy analogue of a corpus that is
    not drawn from the law the force refers to."""
    x = torch.randn(n, 2, generator=g) * 0.5
    e = energy(x)
    for _ in range(burn):
        prop = x + step * torch.randn(n, 2, generator=g)
        ep = energy(prop)
        acc = (torch.rand(n, generator=g) < torch.exp(-(ep - e) / KT))
        x = torch.where(acc.unsqueeze(-1), prop, x)
        e = torch.where(acc, ep, e)
    if misaligned:
        x = x + torch.tensor([1.5, 0.75])
    return x


class VF(nn.Module):
    """v_theta(x, t): small MLP, time appended as a feature."""
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], -1))


def fm_loss(model, x1, g):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, generator=g)
    xt = (1 - t) * x0 + t * x1
    return ((model(xt, t) - (x1 - x0)) ** 2).sum(-1).mean()


def force_loss(model, x1, t_val, g):
    """Exactly the paper's construction: score read out of the FM velocity by
    s = (t v - x_t) / ((1-t) sigma^2), matched to the ENDPOINT force F(x_1)/kT."""
    x0 = torch.randn_like(x1)
    t = torch.full((x1.shape[0], 1), t_val)
    xt = (1 - t) * x0 + t * x1
    v = model(xt, t)
    s = (t * v - xt) / ((1 - t) * 1.0 ** 2)
    return ((s - force(x1) / KT) ** 2).sum(-1).mean()


def flat_grad(loss, model):
    gs = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True,
                             allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if q is None else q).reshape(-1)
                      for p, q in zip(model.parameters(), gs)])


@torch.no_grad()
def log_density(model, x, n_steps=64):
    """Reverse the ODE with an exact 2x2 divergence.  x is (N,2)."""
    xc = x.clone()
    dt = 1.0 / n_steps
    integral = torch.zeros(x.shape[0])
    for k in range(n_steps):
        tv = 1.0 - (k + 0.5) * dt
        t = torch.full((xc.shape[0], 1), tv)
        with torch.enable_grad():
            xr = xc.clone().requires_grad_(True)
            v = model(xr, t)
            d0 = torch.autograd.grad(v[:, 0].sum(), xr, retain_graph=True)[0][:, 0]
            d1 = torch.autograd.grad(v[:, 1].sum(), xr)[0][:, 1]
        integral = integral + dt * (d0 + d1)
        xc = xc - dt * model(xc, t)
    logp0 = -0.5 * (xc ** 2).sum(-1) - math.log(2 * math.pi)
    return logp0 - integral


def ordering_r(model, parents, sigma, g, K=8):
    """The paper's endpoint: per-parent corr(log q, -E/kT), averaged."""
    rs = []
    for p in parents:
        pert = p.unsqueeze(0) + sigma * torch.randn(K, 2, generator=g)
        lq = log_density(model, pert)
        u = -energy(pert) / KT
        if lq.std() < 1e-9 or u.std() < 1e-9:
            continue
        rs.append(float(torch.corrcoef(torch.stack([lq, u]))[0, 1]))
    return float(np.mean(rs)), len(rs)


def run(misaligned, use_force, seed, steps, lam, t_val, n_data=4096):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    data = sample_data(n_data, misaligned, g)
    model = VF()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    cosines = []
    for it in range(steps):
        idx = torch.randint(0, n_data, (256,), generator=g)
        xb = data[idx]
        lf = fm_loss(model, xb, g)
        if it % 50 == 0:      # cosine is a diagnostic of the two gradients, so it is
            lgd = force_loss(model, xb, t_val, g)   # measured in BOTH arms
            a, b = flat_grad(lf, model), flat_grad(lgd, model)
            cosines.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
        if use_force:
            loss = lf + lam * force_loss(model, xb, t_val, g)
        else:
            loss = lf
        opt.zero_grad(); loss.backward(); opt.step()
    ge = torch.Generator().manual_seed(seed + 10_000)
    parents = sample_data(24, misaligned, ge)
    r, n = ordering_r(model, parents, 0.25, ge)
    return {"r": r, "n_parents": n,
            "cos_mean": float(np.mean(cosines)) if cosines else None,
            "cos_frac_neg": float(np.mean([c < 0 for c in cosines])) if cosines else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--ts", default="0.80,0.90,0.95,0.97,0.99")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ts = [float(x) for x in a.ts.split(",")]
    res = {}
    for setting, mis in (("aligned", False), ("misaligned", True)):
        for tv in ts:
            row = {}
            for arm, uf in (("fm_only", False), ("fm_plus_force", True)):
                rs, cs, cn = [], [], []
                for sd in range(a.seeds):
                    o = run(mis, uf, sd, a.steps, a.lam, tv)
                    rs.append(o["r"]); cs.append(o["cos_mean"]); cn.append(o["cos_frac_neg"])
                row[arm] = {"r_mean": round(float(np.mean(rs)), 4),
                            "r_sem": round(float(np.std(rs, ddof=1) / len(rs) ** .5), 4),
                            "r_seeds": [round(x, 4) for x in rs],
                            "cos_mean": round(float(np.mean(cs)), 4),
                            "cos_frac_neg": round(float(np.mean(cn)), 3)}
            row["delta"] = round(row["fm_plus_force"]["r_mean"] - row["fm_only"]["r_mean"], 4)
            row["amplification_t_over_1mt"] = round(tv / (1 - tv), 2)
            res[f"{setting}/t={tv}"] = row
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
