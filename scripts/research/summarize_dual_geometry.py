#!/usr/bin/env python3
"""All-control summary for parameter-free internal-geometry experiments."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();d=json.loads(a.audit.read_text());assert d['complete']
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);methods=['plain','geometry','fixed_geometry','time_geometry'];labels=['Original','Geometry SC','Fixed geometry','Time-scaled']
 with (a.out/'results.csv').open('w') as f:
  fields=['seed','method','attempts','graph','geometry','distinct_connectivity','validator_errors','parameters','training_seconds','generation_seconds','training_reused','primitive_evaluations_per_sample']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for seed,rows in d['summary'].items():
   for m in methods:w.writerow(dict(seed=seed,method=m,**rows[m],**d['costs'][seed][m]))
 fig,axes=plt.subplots(1,2,figsize=(9,3.1),sharey=True)
 for seed,ax in enumerate(axes):
  rows=d['summary'][str(seed)];bars=ax.bar(range(4),[100*rows[m]['graph']/rows[m]['attempts'] for m in methods],color=['#777777','#5389b5','#c78d45','#007f87'])
  for bar,m in zip(bars,methods):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{rows[m]['graph']}/768",ha='center',fontsize=8)
  ax.set_xticks(range(4),labels,rotation=15,ha='right');ax.set_ylim(0,100);ax.set_title(f'Training seed {seed+1}');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Graph support per attempted draw (%)')
 fig.suptitle('Internal geometry representation;128 primitive denoiser calls per draw',fontsize=10)
 fig.tight_layout();fig.savefig(a.out/'dual_geometry_results.pdf');fig.savefig(a.out/'dual_geometry_results.png',dpi=180);plt.close(fig)
 lines=['# Internal geometry comparison','','| Seed | Original | Geometry SC | Fixed geometry | Time-scaled geometry |','|---|---:|---:|---:|---:|']
 for seed in ['0','1']:
  rows=d['summary'][seed];lines.append('| '+str(int(seed)+1)+' | '+' | '.join(f"{rows[m]['graph']}/768" for m in methods)+' |')
 lines+=['','Graph-rate differences in percentage points:','']
 for key,value in d['comparisons'].items():
  r=value['graph_supported'];ci=r['paired_draw95'];cc=r['descriptive_composition95']
  lines.append(f"- {key}: {100*r['mean']:+.2f}; paired95 [{100*ci[0]:+.2f},{100*ci[1]:+.2f}], composition95 [{100*cc[0]:+.2f},{100*cc[1]:+.2f}].")
 lines+=['',f"Prespecified internal-geometry gate passed: **{d['development_gate_passed']}**.",
  f"All{d['source_and_structural_outputs_replayed']} source/output records replay. Original/fixed/time variants have identical trainable parameter counts and state shapes; their training data, source, warm start and update recipe match.",
  'The geometry-SC reference used higher training compute, which is reported separately. All inference arms use128 primitive denoiser calls. This reused12-composition development panel is not an untouched confirmation. No final-density or Boltzmann-law result is established.']
 (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
