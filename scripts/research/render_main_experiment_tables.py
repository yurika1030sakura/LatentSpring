"""Render the main comparison and source/head ablation from audited summaries."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--figure-dir',default='research/figures/experimental_story_v4')
    a=p.parse_args();root=a.project.resolve()
    paths=['research/evidence/seed_replication_audit_v2.json',
           'research/evidence/other_baseline_transfer_audit_v1.json',
           'research/evidence/source_head_factorial_v1.json']
    replication,transfer,factorial=[json.loads((root/name).read_text()) for name in paths]
    assert all(d['complete'] for d in [replication,transfer,factorial])
    assert factorial['summary']==transfer['summary']
    s=replication['summary'];t=transfer['summary'];f=lambda x:f'{100*x:.2f}'
    main_rows=[('Gaussian FM',t['gaussian_fm']['parent']['graph'],t['gaussian_fm']['parent']['joint']),
        ('EDM',t['edm']['parent']['graph'],t['edm']['parent']['joint']),
        ('GAGA',s['gaga_parent']['graph_rate'],s['gaga_parent']['joint_rate']),
        ('LatentSpring (ours)',s['fm_physical']['graph_rate'],s['fm_physical']['joint_rate'])]
    main_text=r'''\subsection{Comparison with generative baselines}\label{sec:main-comparison}
We compare LatentSpring with Gaussian-source flow matching, EDM, and GAGA.
LatentSpring combines the harmonic-mixture source and learned physical
correction; the baselines use their own generators without these additions.
Table~\ref{tab:main-generators} reports both measures on the same compositions.

\begin{table}[H]\centering\small
\caption{Generation from 64 unseen compositions. All values are percentages;
higher is better. Gaussian FM and EDM average two training runs (2,048 outputs
per row); GAGA and LatentSpring average five (5,120 outputs per row). The
LatentSpring row uses the source and physical correction, without the optional
hydrogen readout.}
\begin{tabular}{lrr}\toprule
Method & Graph validity & Joint yield\\\midrule
'''
    for i,(label,graph,joint) in enumerate(main_rows):
        if i==3:main_text+=rf'\textbf{{{label}}} & \textbf{{{f(graph)}}} & \textbf{{{f(joint)}}}\\'+'\n'
        else:main_text+=f'{label} & {f(graph)} & {f(joint)}'+r'\\'+'\n'
    main_text+=r'''\bottomrule\end{tabular}\label{tab:main-generators}
\end{table}

LatentSpring has the highest mean graph validity and joint yield in this
comparison. Its joint yield is '''+f(s['fm_physical']['joint_rate'])+r'''\%, compared with '''+f(s['gaga_parent']['joint_rate'])+r'''\% for GAGA,
'''+f(t['edm']['parent']['joint'])+r'''\% for EDM, and '''+f(t['gaussian_fm']['parent']['joint'])+r'''\% for Gaussian FM. The gains over GAGA and
EDM are '''+f(s['fm_physical']['joint_rate']-s['gaga_parent']['joint_rate'])+' and '+f(s['fm_physical']['joint_rate']-t['edm']['parent']['joint'])+r''' percentage points, respectively. The separation is larger
for joint yield than for graph validity: more generated structures satisfy
both the chemical-graph checks and the physical force threshold.
'''
    ablation_rows=[('Gaussian','No','gaussian_fm','parent'),('Harmonic','No','harmonic_fm','parent'),
                   ('Gaussian','Yes','gaussian_fm','transferred'),('Harmonic','Yes','harmonic_fm','transferred')]
    ablation=r'''\subsection{Ablation of the source and physical correction}\label{sec:source-head-ablation}
We separate the two design choices in a four-setting comparison: a Gaussian
or harmonic source, each with or without physical correction. These experiments
use one-pass EGNN flow models without self-conditioning. Within each training
run, the same frozen correction head is applied to both source variants.
This keeps the learned correction fixed while changing the source.

\begin{table}[H]\centering\small
\caption{Source and physical-correction ablation on the same 64 compositions.
Each row contains 2,048 outputs from two training runs. All values are
percentages; the hydrogen readout is omitted.}
\begin{tabular}{llrr}\toprule
Source & Physical correction & Graph validity & Joint yield\\\midrule
'''
    for i,(source,head,target,arm) in enumerate(ablation_rows):
        value=t[target][arm];cells=[source,head,f(value['graph']),f(value['joint'])]
        if i==3:cells=[rf'\textbf{{{v}}}' for v in cells]
        ablation+=' & '.join(cells)+r'\\'+'\n'
    ablation+=r'''\bottomrule\end{tabular}\label{tab:source-head-ablation}
\end{table}

Physical correction improves joint yield with either source: from
'''+f(t['gaussian_fm']['parent']['joint'])+' to '+f(t['gaussian_fm']['transferred']['joint'])+r'''\% for Gaussian FM and from '''+f(t['harmonic_fm']['parent']['joint'])+' to '+f(t['harmonic_fm']['transferred']['joint'])+r'''\% for harmonic FM.
The harmonic source alone gives smaller changes. Its contribution is clearer
when combined with physical correction: graph validity rises from
'''+f(t['gaussian_fm']['transferred']['graph'])+' to '+f(t['harmonic_fm']['transferred']['graph'])+r'''\%, and joint yield rises from '''+f(t['gaussian_fm']['transferred']['joint'])+' to '+f(t['harmonic_fm']['transferred']['joint'])+r'''\%.
'''
    contrast=factorial['contrasts']['joint']['both_vs_physical_only'];ci=contrast['crossed_fit_composition_ci95']
    ablation+='The '+f(contrast['mean'])+r'''-point joint-yield gain has a 95\% interval of $['''+f(ci[0])+','+f(ci[1])+r''']$
when resampling both training runs and compositions. Both runs favor the
combination over either component alone. Appendix~\ref{app:main-implementation}
reports the individual runs and the four-setting analysis.
'''
    def replace_table(text,figure,caption,label):
        start=text.index(r'\begin{table}');end=text.index(r'\end{table}',start)+len(r'\end{table}')
        block=(r'\begin{figure}[H]\centering'+'\n'+r'\includegraphics[width=\linewidth]{../'+a.figure_dir+'/'+figure+'.pdf}\n'+
               r'\caption{'+caption+'}'+r'\label{'+label+'}\n'+r'\end{figure}')
        return text[:start]+block+text[end:]
    main_text=replace_table(main_text,'main_comparison',
        r'Complete generators on 64 unseen compositions. Bars give mean graph validity (A) and joint yield (B); open circles show every training fit. Gaussian FM and EDM use two fits (2,048 outputs each); GAGA and LatentSpring use five (5,120 each). All attempts are counted. LatentSpring uses the source and physical correction, without the optional hydrogen readout.',
        'fig:main-generators').replace(r'Table~\ref{tab:main-generators}',r'Figure~\ref{fig:main-generators}')
    ablation=replace_table(ablation,'source_correction_ablation',
        r'Four-setting FM ablation on the same 64 compositions, without self-conditioning or hydrogen readout. Color and marker shape distinguish Gaussian and harmonic sources; the horizontal axis switches physical correction off or on. Bold lines and value labels show means; thin lines show both training fits. Each configuration contains 2,048 outputs.',
        'fig:source-head-ablation')
    outputs={'paper/sections/main_generator_comparison.tex':main_text,
             'paper/sections/source_head_ablation.tex':ablation}
    for name,text in outputs.items():(root/name).write_text(text)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(complete=True,inputs={p:digest(root/p) for p in paths},
        outputs={p:digest(root/p) for p in outputs},main_rows=main_rows,
        scope='Tables generated from completed audits. Distinct repetition counts are stated. The factorial is a two-fit one-pass-EGNN analysis, not the exact self-conditioned main configuration.')
    (root/'research/evidence/main_experiment_tables_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
