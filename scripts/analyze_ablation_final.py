"""汇总消融实验的最终统计（真实/打乱/无物理 × 多种子 × 某个 n）。"""
import glob, json, statistics as st, math, sys
S="/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours"
prefix=sys.argv[1] if len(sys.argv)>1 else "n240"
def vals(pat):
    out=[]
    for f in sorted(glob.glob(f"{S}/{prefix}_{pat}/boltz_independent.json")):
        try: out.append(json.load(open(f))["mean_pearson_r"])
        except: pass
    return out
real=vals("a3_energy_only_s*")+vals("a3_energy_only")   # 各种子
shuf=vals("a6_energy_only_shuffled_s*")+vals("a6_energy_only_shuffled_stab_s*")
nop =vals("a1_fm_only_s*")
def welch(x,y):
    if len(x)<2 or len(y)<2: return None,None
    mx,my=st.mean(x),st.mean(y); vx,vy=st.variance(x),st.variance(y)
    se=math.sqrt(vx/len(x)+vy/len(y)); 
    return mx-my, (mx-my)/se if se>0 else 0
print(f"=== 最终统计 ({prefix}, n_molecules per eval) ===")
for name,v in [("能量-真实",real),("能量-打乱",shuf),("无物理",nop)]:
    if v:
        sem=st.stdev(v)/math.sqrt(len(v)) if len(v)>1 else float('nan')
        print(f"  {name:10s} n_seeds={len(v)} mean={st.mean(v):+.3f}±{sem:.3f} {[round(x,3) for x in v]}")
for lab,a,b in [("能量项有效: 真实 vs 无物理",real,nop),
                ("能量信息(Y-scramble): 真实 vs 打乱",real,shuf),
                ("正则化成分: 打乱 vs 无物理",shuf,nop)]:
    d,t=welch(a,b)
    if d is not None:
        print(f"  [{lab}] Δ={d:+.3f} t={t:+.2f} {'✅显著' if abs(t)>2 else '⚠️不显著'}")
