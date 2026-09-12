#!/usr/bin/env python3
"""Plot the complete declared-budget and equal-inference scalar-learning controls."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();r=json.loads(a.summary.read_text())
assert r['complete'] and r['all_primary_budgets_matched'] and r['all_48_parents_at_all_caps']
a.out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(8.4,3.1),sharey=True)
for ax,cap,title in zip(axes,['128','final'],['Same inference:128 calls / parent','Including model data costs']):
 for y,(method,label) in enumerate([('learned_work','Work'),('learned_work_force','Work + force')]):
  d=r['readouts'][cap]['paired_potential_change_eV'][method+'_minus_arc_site'];m=d['mean'];lo,hi=d['fixed_composition_parent_bootstrap_95_percent_interval']
  ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#236783',capsize=4,linewidth=1.8)
 ax.axvline(0,color='black',linestyle='--',linewidth=.8)
 ax.set_title(title,fontsize=10);ax.set_xlabel('Learned minus physical potential (eV)',fontsize=9)
 ax.set_yticks([0,1],['Work','Work + force']);ax.set_ylim(-.55,1.55);ax.invert_yaxis()
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
fig.text(.5,.02,'Negative favors learning. Internal development; fixed-composition parent intervals.\nNominal method budgets match;1,392 additional control-repair calls are separately retained.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.14,1,1))
for suffix in ['pdf','png']:
 path=a.out/f'conditional_arc_controls.{suffix}'
 if path.exists():raise FileExistsError(path)
 fig.savefig(path,dpi=180)
print(a.out/'conditional_arc_controls.pdf')
