"""Pretty-print the JSON written by scripts/p1_estimator_precision.py."""
import json, sys, statistics as st

def d(x, k, f="{:.4g}"):
    v = x.get(k)
    return "n/a" if v is None else f.format(v)

def block(name, s):
    if not s: return
    print(f"  {name:34s} n={s.get('n','-'):4} mean={d(s,'mean')}  median={d(s,'median')}"
          f"  sd={d(s,'sd')}  [{d(s,'min')}, {d(s,'max')}]")

for path in sys.argv[1:]:
    r = json.load(open(path))
    print("=" * 100)
    print(path)
    print(f"settings: {json.dumps(r.get('settings', {}))}")
    if "toy" in r:
        t = r["toy"]
        print("\n-- TOY ANALYTIC CONTROL (linear field, dim %d) --" % t["dim"])
        print(f"   exact autograd trace error   : {t['exact_abs_error']:.3e}")
        print(f"   Hutchinson-2 bias            : {t['hutch2_bias']:+.5f} "
              f"+- {t['hutch2_bias_sem']:.5f} (SEM, {t['n_repeat']} draws)")
        print(f"   Hutchinson-2 SD measured     : {t['hutch2_sd_measured']:.4f}")
        print(f"   Hutchinson-2 SD analytic     : {t['hutch2_sd_analytic']:.4f}")
    if "taskA" in r:
        a = r["taskA"]
        print(f"\n-- TASK A: exact divergence vs {r['settings']['n_hutchinson']}-probe "
              f"Hutchinson, {a['n_repeat']} repeats --")
        for g in a["parents"]:
            extra = ""
            if "r_exact_divergence" in g:
                extra = (f"  r_exact={g['r_exact_divergence']:+.3f} "
                         f"r_hutch={g['r_hutch2_mean']:+.3f}+-{g['r_hutch2_sd']:.3f}")
            print(f"   parent {g['group']:4d}  N={g['n_atoms']:3d} geoms={g.get('n_geoms','-')}"
                  f"  {g['seconds']:6.1f}s{extra}")
        print("   summary:")
        for k, v in a["summary"].items():
            block(k, v)
    if "taskB" in r:
        b = r["taskB"]
        print(f"\n-- TASK B: repeat scoring of log q, {b['n_repeat']} independent probe sets --")
        print(f"   {'grp':>4} {'N':>4} {'noise_sd':>9} {'signal_sd':>10} {'n/s':>6} "
              f"{'r(1 draw)':>10} {'sd(r)':>7} {'r(mean logq)':>13}")
        for g in b["per_group"]:
            print(f"   {g['group']:4d} {g['n_atoms']:4d} {g['noise_sd']:9.3f} "
                  f"{g['signal_sd_corrected']:10.3f} {g['noise_over_signal']:6.3f} "
                  f"{g['r_per_repeat_mean']:+10.3f} {g['r_per_repeat_sd']:7.3f} "
                  f"{g['r_repeat_mean_logq']:+13.3f}")
        print("   summary:")
        for k, v in b["summary"].items():
            if k == "population_mean_r_per_repeat":
                print(f"  {'population mean r per draw':34s} n_parents={v['n_parents']} "
                      f"n_repeats={v['n_repeats']} mean={v['mean']:+.4f} sd={v['sd']:.4f}")
                print("       values: " + " ".join(f"{x:+.4f}" for x in v["values"]))
                continue
            block(k, v)
    if "production_check" in r:
        p = r["production_check"]
        print("\n-- PRODUCTION-PATH CHECK (log_density_via_flow called directly) --")
        print(f"   group {p['group']}")
        print(f"   direct   mean {['%.2f' % x for x in p['log_density_via_flow_mean']]}")
        print(f"   direct   sd   {['%.2f' % x for x in p['log_density_via_flow_sd']]}")
        print(f"   instrum. mean {['%.2f' % x for x in p['instrumented_mean']]}")
        print(f"   instrum. sd   {['%.2f' % x for x in p['instrumented_sd']]}")
