"""分辨率敏感性 ablation 的配对统计。各分辨率用同一 eval seed -> 同分子同扰动。"""
import csv, glob, os, math, statistics as st
S = "/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_res"

def group_r(f, drop_ref=True):
    if not os.path.exists(f): return None
    g = {}
    for row in csv.DictReader(open(f)):
        try:
            pid, gid = int(row["pert_id"]), int(row["group_id"])
            lp, E = float(row["log_p_theta"]), float(row["E_eV"])
        except (KeyError, ValueError): continue
        if drop_ref and pid == 0: continue
        g.setdefault(gid, []).append((lp, -E))
    rs = []
    for v in g.values():
        if len(v) < 3: continue
        x = [a for a, _ in v]; y = [b for _, b in v]
        if st.pstdev(x) < 1e-12 or st.pstdev(y) < 1e-12: continue
        mx, my = st.mean(x), st.mean(y)
        num = sum((a-mx)*(b-my) for a, b in v)
        den = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
        if den > 0: rs.append(num/den)
    return st.mean(rs) if rs else None

print("=== 分辨率敏感性 ablation (剔除参考构型; 120 分子/种子) ===")
print(f"{'ODE步':>6} {'ε=1/2n':>8} | {'能量臂':>16} | {'FM-only':>16} | {'gap':>8} {'t':>7}")
per_seed = []   # (ode, arm, [(run_tag, r), ...]) -- printed in full below
for ode in (4, 12, 48):
    vals = {}
    for arm, lab in (("a3_energy_only", "E"), ("a1_fm_only", "C")):
        rs, tags = [], []
        for d in sorted(glob.glob(f"{S}/{arm}*__ode{ode}")):
            r = group_r(f"{d}/boltz_independent_records.csv")
            if r is None: r = group_r(f"{d}/boltz_records.csv")
            if r is not None:
                rs.append(r); tags.append(os.path.basename(d).split("__")[0])
        vals[lab] = rs
        per_seed.append((ode, arm, list(zip(tags, rs))))
    e, c = vals.get("E", []), vals.get("C", [])
    if not e or not c:
        print(f"{ode:>6} {0.5/ode:>8.4f} | {'pending':>16} | {'pending':>16} |")
        continue
    d = st.mean(e) - st.mean(c)
    se = (math.sqrt(st.variance(e)/len(e) + st.variance(c)/len(c))
          if len(e) > 1 and len(c) > 1 else float('nan'))
    t = d/se if se == se and se > 0 else float('nan')
    print(f"{ode:>6} {0.5/ode:>8.4f} | {st.mean(e):+.3f} (n={len(e)}) | "
          f"{st.mean(c):+.3f} (n={len(c)}) | {d:+8.3f} {t:>7.2f}")
print("\n训练时 energy_n_ode_steps=4 (ε=0.125); 主结果评测用 12 步 (ε=0.042)")

# Per-seed values.  The appendix (sections/A2_details.tex, tab:respowered protocol
# paragraph) states that "per-seed values are written to that analysis script's
# output"; this block is what makes that statement true.  Same r as the means
# above, just not aggregated.
print("\n=== 每个 checkpoint 的 per-seed r (同一 eval seed 12345; 剔除参考构型) ===")
for ode, arm, rows in per_seed:
    if not rows: continue
    body = "  ".join(f"{tag}={r:+.3f}" for tag, r in rows)
    sd = st.stdev([r for _, r in rows]) if len(rows) > 1 else float("nan")
    print(f"  n_ode={ode:>2}  {arm:<14} n={len(rows)}  sd={sd:.3f}  {body}")
