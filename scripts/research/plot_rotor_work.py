#!/usr/bin/env python3
"""Publication figure for all prespecified nonidentity molecular rotor cases."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();results=json.loads(a.results.read_text());audit=json.loads(a.audit.read_text())
    assert audit['complete'] and audit['result_sha256']==sha(a.results)
    a.out.mkdir(parents=True,exist_ok=True)
    methods=[('complete_work','Complete work','#2166ac'),('energy_only','Energy only','#d6604d'),
             ('without_jacobian','Without Jacobian','#9970ab'),('unweighted','Unweighted','#777777')]
    sizes=[8,32,128,512,2048]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,2.65),constrained_layout=True)
    for method,label,color in methods:
        grouped=[[r for study in results['studies'] for r in study['comparisons'] if r['method']==method and r['n']==n and r['amplitude']!=0] for n in sizes]
        tv=[np.mean([r['mean_total_variation'] for r in group]) for group in grouped]
        axes[0].plot(sizes,tv,'o-',color=color,label=label,markersize=3)
        if method!='unweighted':
            error=[1000*np.mean([r['free_energy_rmse_eV'] for r in group]) for group in grouped]
            axes[1].plot(sizes,error,'o-',color=color,label=label,markersize=3)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(sizes,[str(n) for n in sizes]);ax.set_xlabel('Samples per estimate');ax.grid(alpha=.18)
    axes[0].set_ylabel('Mean total-variation error');axes[0].set_title('(a) Target distribution')
    axes[1].set_ylabel('Mean free-energy RMSE (meV)');axes[1].set_yscale('log');axes[1].set_title('(b) Target normalizer')
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(a.out/'rotor_work.pdf');fig.savefig(a.out/'rotor_work.png',dpi=200);plt.close(fig)
    (a.out/'provenance.json').write_text(json.dumps(dict(results_sha256=sha(a.results),audit_sha256=sha(a.audit),
        aggregation='Equal average of all3 molecules and2 nonidentity escort amplitudes;128 repetitions per case at each N. Identity controls retained in full results.',
        pdf_sha256=sha(a.out/'rotor_work.pdf')),indent=2)+'\n')


if __name__=='__main__':main()
