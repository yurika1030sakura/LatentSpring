"""Render all five fit pairs and write evidence-linked manuscript sections."""
import argparse,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scripts.research.train_electronic_fm import sha

def interval(d,key='composition_ci95',scale=100):
    lo,hi=np.asarray(d[key])*scale
    return f'[{lo:.2f},{hi:.2f}]'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','audit','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();d=json.loads(a.audit.read_text());assert d['complete'];a.out.mkdir(parents=True,exist_ok=False)
    assert sha(a.audit.with_suffix('.npz'))==d['arrays_sha256'];arrays=dict(np.load(a.audit.with_suffix('.npz')));summary=d['summary'];methods=d['methods']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors={'fm':'#238078','gaga':'#b87640'};fig,axes=plt.subplots(1,2,figsize=(5.5,2.5),sharey=True)
    maximum=max(max(summary[n+'_'+v]['joint_by_fit']) for n in colors for v in ['parent','physical','hydrogen'])*100
    for ax,(name,color) in zip(axes,colors.items()):
        points=np.array([summary[name+'_'+v]['joint_by_fit'] for v in ['parent','physical','hydrogen']]).T*100
        for si,row in enumerate(points):
            col=color if si>=2 else '#8b969d';ax.plot(range(3),row,color=col,lw=1.1,alpha=.85,marker='o',markersize=4,mfc=col if si>=2 else 'white',mew=.8)
        ax.set_xticks(range(3),['Parent','+ physical','+ H flow']);ax.set_title('FM' if name=='fm' else 'GAGA',loc='left',fontsize=10,color=color,pad=8)
        ax.set_ylim(0,max(40,10*np.ceil((maximum+3)/10)));ax.set_xlim(-.2,2.2)
        for spine in ['top','right']:ax.spines[spine].set_visible(False)
        ax.grid(axis='y',color='#e4e8eb',lw=.5);ax.set_axisbelow(True)
    axes[0].set_ylabel('Joint graph/force yield (%)')
    fig.subplots_adjust(left=.13,right=.98,bottom=.24,top=.84,wspace=.25)
    fig.text(.13,.052,'Open gray: earlier fits     Filled color: three new independent fits',fontsize=7.5,color='#59656d')
    for ext in ['pdf','svg','png']:fig.savefig(a.out/('five_fit_replication.'+ext),dpi=350,facecolor='white')
    Image.open(a.out/'five_fit_replication.png').convert('L').save(a.out/'five_fit_replication_grayscale.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.35),sharey=True);thresholds=np.linspace(0,12,121)
    for ax,(family,color) in zip(axes,colors.items()):
        for suffix,label,style in [('parent','Parent','--'),('physical','Physical head','-'),('hydrogen','Head + H flow',':')]:
            mi=methods.index(family+'_'+suffix);valid=arrays['graph'][:,mi]&arrays['success'][:,mi];forces=arrays['force'][:,mi]
            ys=[100*(valid&(forces<=t)).mean() for t in thresholds];ax.plot(thresholds,ys,ls=style,lw=1.5,color=color if suffix!='parent' else '#7d8890',label=label)
        graph_maximum=100*max(max(summary[n+'_'+s]['graph_by_fit']) for n in colors for s in ['parent','physical','hydrogen'])
        ax.set_title('FM' if family=='fm' else 'GAGA',loc='left',fontsize=10,color=color);ax.set_xlim(0,12);ax.set_ylim(0,max(40,10*np.ceil((graph_maximum+3)/10)))
        ax.set_xlabel('Force RMS threshold (eV/Å)');ax.axvline(5,color='#c2c9cd',lw=.7,zorder=-1)
        for spine in ['top','right']:ax.spines[spine].set_visible(False)
    axes[0].set_ylabel('Joint yield (%)');axes[1].legend(frameon=False,fontsize=7,loc='lower right');fig.subplots_adjust(left=.12,right=.98,bottom=.22,top=.86,wspace=.24)
    for ext in ['pdf','svg','png']:fig.savefig(a.out/('force_thresholds.'+ext),dpi=350,facecolor='white')
    plt.close(fig)
    c=d['contrasts'];fm=c['fm_physical_minus_parent']['new_three']['joint'];ga=c['gaga_physical_minus_parent']['new_three']['joint'];hh=c['fm_hydrogen_minus_radial']['new_three']['energy'];compare=c['fm_minus_gaga_hydrogen']['all_five']
    lines=[r'\subsection{Physical correction across independent training runs}\label{sec:seed-replication}',
        'We fix the selected method and add three independently trained FM/GAGA pairs to the two earlier pairs. Each uses the same 20,000 OMol25 structures and 960,000 backbone training-example forward passes. Each physical head receives 1,024 eSEN queries and 20,000 updates; a hydrogen model is shared by the two generators within each pair. All five pairs generate on the same 64 new compositions, with 16 draws per composition. The three new pairs test training replication; the five-pair average also includes the earlier models that informed method development.',
        r'\begin{table}[t]\centering\small',
        r'\caption{Frozen-method comparison on 64 new compositions and five fit pairs, with 5,120 attempts per row. Joint yield requires a valid inferred graph and GFN2 force RMS at most 5 eV/\AA. All outputs are scored at their generated coordinates.}',
        r'\begin{tabular}{llrr}\toprule Parent & Added module & Graph (\%) & Joint (\%)\\\midrule']
    for family in ['fm','gaga']:
        for suffix,label in [('parent','None'),('physical','Physical head'),('hydrogen','Head + H flow')]:
            item=summary[family+'_'+suffix];lines.append(f"{family.upper()} & {label} & {item['graph_rate']*100:.2f} & {item['joint_rate']*100:.2f}"+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\label{tab:seed-replication}\end{table}',
        f"Across the three new FM fits, physical correction changes joint yield by {fm['mean']*100:+.2f} percentage points (95\\% composition-bootstrap interval ${interval(fm)}$). The individual changes are "+', '.join(f'{v*100:+.2f}' for v in fm['by_fit'])+f" points. Resampling both fitted models and compositions gives ${interval(fm,'crossed_fit_composition_ci95')}$; this additionally reflects training variation. GAGA's corresponding change is {ga['mean']*100:+.2f} points (${interval(ga)}$).",
        r'\begin{figure}[t]\centering\includegraphics[width=\linewidth]{../research/figures/seed_replication_v1/five_fit_replication.pdf}',
        r'\caption{Every fitted model in the replication study. Each line follows one fixed parent through physical correction and the hydrogen readout. Gray open markers identify earlier fits; filled markers identify the three new fits. The models are compared on identical compositions and all attempts remain in the denominator.}\label{fig:seed-replication}\end{figure}'])
    if hh['complete']:
        lines.append(f"For the three new FM fits, the H flow changes all-output energy by {1000*hh['mean']:+.2f} meV/atom relative to the fixed radial rule (${interval(hh,scale=1000)}$). This comparison isolates learned placement from a simple attachment rule.")
    else:lines.append(f"The H-flow energy comparison has {hh['missing_changed_pairs']} changed-coordinate pairs with incomplete physical scores; an all-output mean difference is not estimated for that contrast.")
    lines.append(f"With identical added modules, the five-fit FM-minus-GAGA difference is {100*compare['graph']['mean']:+.2f} graph-validity points (${interval(compare['graph'])}$) and {100*compare['joint']['mean']:+.2f} joint-yield points (${interval(compare['joint'])}$). Appendix~\\ref{{app:seed-replication}} reports the new-fit subset, training uncertainty, size strata, radial controls, and force-threshold curves.")
    if not d['primary_new_fit_replication_gate']:lines.append('The independent-training comparison does not meet the prespecified requirement of a positive interval and improvement in every new fit.')
    clean_tables=lambda text:re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}',lambda m:m.group(0).replace('\n\n','\n'),text,flags=re.S)
    section=a.project/'paper/sections/seed_replication_results.tex';assert not section.exists();section.write_text(clean_tables('\n\n'.join(lines))+'\n')
    details=[r'\subsection{Frozen-method training replication}\label{app:seed-replication}',
        'We freeze all architectures, checkpoint schedules, physical strength four, and hydrogen-flow settings before the new fits. Each new FM parent receives 15,000 two-pass updates of batch size 32; GAGA receives 30,000 one-pass updates. Initialization tensors and the batch-schedule prefix match within each pair. This matches backbone training-example forward passes, while optimizer steps, data presentations, and measured runtime differ. All newly trained parents, heads, and H flows use their prescribed final checkpoints.',
        'The new metadata pool excludes every composition in the previous pool. Exhaustive qualification leaves only 20 previously unused compositions with 17--28 atoms. Before any generation, we therefore fix these 20 and the first 44 eligible compositions with 29--40 atoms under the previously specified metadata ranking. The planned 32-per-bin selection and its failed qualification are retained. The final 64 compositions occur in neither processed generator corpus nor the prepared training/validation rows. Reference geometry and overlap checks use no generated outcomes or energy ranks.',
        'For each parent, the strength-zero and strength-four trajectories share source draws and sampler randomness. The zero-strength control evaluates the head but multiplies its output by zero; its recorded call budget therefore includes these unused head calls. Each corrected output also receives the unchanged, fixed radial, and learned hydrogen readouts. Original graph-valid outputs are preserved exactly by the H readout. Identical coordinates reuse their original physical calculation, and all failed calculations remain in graph/force denominators.',
        'The primary independent-training analysis uses the three new pairs. The two earlier pairs participated in method development and are included in the separate five-pair summary. Composition intervals resample the 64 compositions while retaining all fitted models. Crossed intervals resample both fit indices and composition indices; only three new fits, or five total fits, are available to estimate training variation. We additionally give equal-weight averages of the two size bins, so the 20:44 composition mix is explicit.',
        r'\begin{table}[h]\centering\small\caption{Joint yield in every fit (percent). Fits 1--2 reuse earlier weights; fits 3--5 are newly trained.}\begin{tabular}{llrrrrr}\toprule Parent & Module & 1 & 2 & 3 & 4 & 5\\\midrule']
    for family in ['fm','gaga']:
        for suffix,label in [('parent','None'),('physical','Physical'),('radial','Physical + radial'),('hydrogen','Physical + H flow')]:details.append(family.upper()+' & '+label+' & '+' & '.join(f'{100*v:.2f}' for v in summary[family+'_'+suffix]['joint_by_fit'])+r'\\')
    details.extend([r'\bottomrule\end{tabular}\end{table}',r'\begin{figure}[h]\centering\includegraphics[width=\linewidth]{../research/figures/seed_replication_v1/force_thresholds.pdf}\caption{Joint yield as the force threshold varies, averaged over all five fits. The vertical reference marks the prespecified 5 eV/\AA\ endpoint. These curves are descriptive; no threshold is selected from them.}\end{figure}'])
    for name in ['fm_physical_minus_parent','gaga_physical_minus_parent','fm_minus_gaga_hydrogen']:
        z=c[name]['all_five']['joint'];details.append(name.replace('_',' ')+f": pooled change {100*z['mean']:+.2f} points, crossed interval ${interval(z,'crossed_fit_composition_ci95')}$; equal-stratum change {100*z['equal_stratum_mean']:+.2f} points, crossed interval ${interval(z,'equal_stratum_crossed_ci95')}$.")
    costs=d['costs'];details.append(f"The study adds {costs['new_evaluation_parent_trajectories']:,} evaluation parent trajectories, {costs['new_derived_outputs']:,} derived readout outputs, and {costs['new_gfn2_attempts']:,} GFN2 attempts. New physical teachers use {costs['new_fit_trajectories']:,} TRAIN trajectories and {costs['new_esen_queries']:,} eSEN calls. The three new pairs add 135,000 parent updates, 120,000 head updates, and 30,000 hydrogen-model updates. Reused parent and readout calculations are not counted as new oracle calls.")
    path=a.project/'paper/sections/seed_replication_details.tex';assert not path.exists();path.write_text(clean_tables('\n\n'.join(details))+'\n')
    (a.out/'figure_manifest.json').write_text(json.dumps(dict(audit_sha256=sha(a.audit),arrays_sha256=d['arrays_sha256'],all_fit_pairs_shown=True,
        files={f.name:sha(f) for f in a.out.iterdir() if f.is_file()},sections={str(f.relative_to(a.project)):sha(f) for f in [section,path]}),indent=2)+'\n')

if __name__=='__main__':main()
