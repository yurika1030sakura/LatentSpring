#!/usr/bin/env python3
"""Render the complete independent-backbone audit without selecting methods."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.audit.read_text())
    assert report['complete'] and report['no_pretraining']
    names = [('gaussian_fm_128', 'Gaussian FM'), ('harmonic_fm_128', 'Harmonic FM'),
             ('edm_128', 'EDM'), ('gaga_128', 'GAGA')]
    table = []
    rates = {}
    for key, name in names:
        rows = [report['summary'][str(seed)][key] for seed in [0, 1]]
        assert all(row['attempted'] == 2048 for row in rows)
        counts = [r['graph_supported'] for r in rows]
        rates[key] = 100*sum(counts)/4096
        table.append(f'{name} & {counts[0]} & {counts[1]} & {rates[key]:.2f}'+r'\\')
    contrast = report['comparisons']['harmonic_fm_128__minus__gaussian_fm_128']['graph_supported']
    delta = 100*contrast['mean']
    lo, hi = [100*v for v in contrast['descriptive_composition95']]
    direction = 'increases' if delta >= 0 else 'decreases'
    diffusion = all(rates[k] > rates['harmonic_fm_128'] for k in ['edm_128', 'gaga_128'])
    text = r'''\subsection{Independent backbone and generator objectives}\label{sec:matched-generators}
We repeat the source comparison on an independent EGNN backbone and compare with
position-conditional EDM and GAGA \citep{edm,qu2026gaga}. All four methods train
from scratch on the same 20,000 OMol25 rows, with identical architecture and
initial weights within each of two repetitions. Each receives 30,000 updates at
batch size 32. The FM arms share their training path and atom pairing; their
source distribution is the only difference. Appendix~\ref{app:matched-generators}
gives the protocol and the transferred GAGA truncation setting.

\begin{table}[t]
\centering
\caption{Chemical-graph validity on the same 64 compositions, with 2,048 attempts
per initialization and 128 network evaluations per output. All models share the
2.38-million-parameter EGNN architecture and have no pretraining.}
\begin{tabular}{lrrr}
\toprule
Method & Valid, seed 1 & Valid, seed 2 & Pooled rate (\%)\\
\midrule
'''+ '\n'.join(table)+r'''
\bottomrule
\end{tabular}\label{tab:matched-generators}
\end{table}

'''
    text += f'The harmonic source {direction} FM validity by {abs(delta):.2f} percentage points,\n'
    text += f'with a composition-bootstrap 95\\% interval of [{lo:.2f},{hi:.2f}] for the signed\n'
    text += 'difference. This comparison tests source transfer independently of pretrained\nbackbone history. '
    if diffusion:
        text += 'The diffusion objectives reach higher validity in this setting.\n'
    else:
        text += 'Table~\\ref{tab:matched-generators} reports the relative performance of all four objectives.\n'
    text += 'The matched FM contrast isolates the source contribution; comparisons with\ndiffusion also change the probability path, objective and sampler.\n'
    args.out.write_text(text)
    print(json.dumps(dict(rates=rates, source_gain_pp=delta, composition_interval_pp=[lo,hi])))


if __name__ == '__main__':
    main()
