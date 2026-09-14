#!/usr/bin/env python3
"""Present all dynamic connectivity controls and both uncertainty summaries."""
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
 a.out.mkdir(parents=True)
 methods=['fixed','tree_learned','tree_fixed','local_learned']
 names=['No block','Learned tree','Fixed affinities','Learned local']
 with (a.out/'results.csv').open('w') as f:
  fields=['seed','method','attempts','graph','geometry','distinct_connectivity','validator_errors','training_seconds','generation_seconds']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for s,rows in d['summary'].items():
   for m in methods:writer.writerow(dict(seed=s,method=m,**{k:v for k,v in rows[m].items() if k!='learning'},**d['costs'][s][m]))
 fig,axes=plt.subplots(1,2,figsize=(9,3.1),sharey=True)
 for s,ax in enumerate(axes):
  rows=d['summary'][str(s)]
  bars=ax.bar(range(4),[100*rows[m]['graph']/rows[m]['attempts'] for m in methods],color=['#777777','#007f87','#c78d45','#8b6daf'])
  for bar,m in zip(bars,methods):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{rows[m]['graph']}/{rows[m]['attempts']}",ha='center',fontsize=8)
  ax.set_xticks(range(4),names,rotation=18,ha='right');ax.set_title(f'Training seed {s+1}');ax.set_ylim(0,100)
  ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Graph support per attempted draw (%)')
 fig.suptitle('Current-geometry connectivity: reused monomer development panel',fontsize=10)
 fig.tight_layout();fig.savefig(a.out/'dynamic_tree_results.pdf');fig.savefig(a.out/'dynamic_tree_results.png',dpi=180)
 lines=['# Dynamic connection pilot results','',
  '| Seed | No block | Learned tree | Fixed tree affinities | Learned local |','|---|---:|---:|---:|---:|']
 for s in ['0','1']:
  rows=d['summary'][s];lines.append('| '+str(int(s)+1)+' | '+' | '.join(f"{rows[m]['graph']}/{rows[m]['attempts']}" for m in methods)+' |')
 lines+=['','Learned-tree graph-rate differences, percentage points:','']
 for m in ['fixed','local_learned','tree_fixed']:
  r=d['comparisons'][m]['graph_supported'];ci=r['paired_draw95'];cc=r['descriptive_composition95']
  lines.append(f"- vs {m}: {100*r['mean']:+.2f}; paired95 [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}], composition95 [{100*cc[0]:+.2f}, {100*cc[1]:+.2f}].")
 lines+=['',f"Prespecified development gate passed: **{d['development_gate_passed']}**.",
  f"All{d['source_and_structural_attempts_replayed']} source draws/structural assays replay. Training rows match across methods within each seed and independently exclude declared evaluation compositions. Initial adapter tensors match; full checkpoints restore.",
  'This is a reused12-composition development panel with new generation streams and training seeds. It is not an untouched confirmatory result, a new matrix-tree theorem, a complete comparison with independently trained generators, or evidence of Boltzmann sampling.',
  'The local control does not cover every possible attention/degree-normalization rule. Larger-cut versus singleton-coverage effects require further controls if the first mechanism gains. Preserve prior negative source-utility and static-context experiments.']
 (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
