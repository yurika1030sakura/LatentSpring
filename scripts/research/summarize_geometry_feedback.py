#!/usr/bin/env python3
"""Summarize all feedback controls without confusing known SC with new benefit."""
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
 a.out.mkdir(parents=True);methods=['plain','geometry','latent','pooled'];labels=['One pass','Geometry SC','Pair codes','Pooled codes']
 with (a.out/'results.csv').open('w') as f:
  fields=['seed','method','attempts','graph','geometry','distinct_connectivity','validator_errors','edge_head_parameter_change','training_seconds','generation_seconds','training_primitive_forwards','primitive_evaluations_per_sample']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for s,rows in d['summary'].items():
   for m in methods:writer.writerow(dict(seed=s,method=m,**rows[m],**d['costs'][s][m]))
 fig,axes=plt.subplots(1,2,figsize=(9,3.1),sharey=True)
 for s,ax in enumerate(axes):
  rows=d['summary'][str(s)];bars=ax.bar(range(4),[100*rows[m]['graph']/rows[m]['attempts'] for m in methods],color=['#777777','#5389b5','#007f87','#c78d45'])
  for bar,m in zip(bars,methods):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{rows[m]['graph']}/{rows[m]['attempts']}",ha='center',fontsize=8)
  ax.set_xticks(range(4),labels,rotation=15,ha='right');ax.set_title(f'Training seed {s+1}');ax.set_ylim(0,100);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Graph support per attempted draw (%)')
 fig.suptitle('128 primitive denoiser calls per sample; reused monomer development panel',fontsize=10)
 fig.tight_layout();fig.savefig(a.out/'geometry_feedback_results.pdf');fig.savefig(a.out/'geometry_feedback_results.png',dpi=180);plt.close(fig)
 lines=['# Geometry/relation feedback results','','| Seed | One pass | Geometry SC | Pair codes | Pooled codes |','|---|---:|---:|---:|---:|']
 for s in ['0','1']:
  rows=d['summary'][s];lines.append('| '+str(int(s)+1)+' | '+' | '.join(f"{rows[m]['graph']}/768" for m in methods)+' |')
 lines+=['','Graph-rate differences in percentage points; both uncertainty summaries retained:','']
 for key,value in d['comparisons'].items():
  r=value['graph_supported'];ci=r['paired_draw95'];cc=r['descriptive_composition95']
  lines.append(f"- {key}: {100*r['mean']:+.2f}; paired95 [{100*ci[0]:+.2f},{100*ci[1]:+.2f}], composition95 [{100*cc[0]:+.2f},{100*cc[1]:+.2f}].")
 lines+=['',f"Prespecified relation-learning gate passed: **{d['development_gate_passed']}**.",
  f"All{d['source_and_structural_outputs_replayed']} source/structure output records replay. Checkpoints restore, matching initial feedback weights and coordinate-only edge-head updates are verified.",
  'A gain of geometry SC over plain is a known-technique baseline improvement, not evidence of new AI novelty. Primary learned-relation evidence requires gains over geometry-only and pooled-code feedback.',
  'All inference arms use128 primitive denoiser calls. Training updates match but SC training compute is higher; component runtimes and forward counts are retained. The12 compositions are reused development conditions, not untouched confirmatory evidence. No chemical-bond semantics are assigned to the codes and no energy-law claim is made.']
 (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
