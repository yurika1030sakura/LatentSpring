#!/usr/bin/env python3
"""Vector overview separating molecular generation from physical learning."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'font.size':10})
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    ax.set(xlim=(0, 10), ylim=(0, 4.4))
    ax.axis('off')

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.035,rounding_size=0.09',
                                   facecolor=color, edgecolor='#405366', linewidth=1.0))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', linespacing=1.4)

    def arrow(a, b, color='#405366'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
                                    linewidth=1.25,color=color))

    ax.text(.10, 4.15, 'GENERATION', weight='bold', color='#163b65', fontsize=12)
    ax.text(2.05, 4.15, 'One trained network is reused throughout the flow.', color='#405366')
    box(.12, 3.03, 1.8, .86, 'Atomic input\nelements\ncharge and spin', '#edf4fb')
    box(2.26, 3.03, 2.08, .86, 'Latent spring source\nrandom scaffold', '#e7f1e5')
    box(4.68, 3.03, 2.72, .86, 'Flow updates\nprovisional geometry\nthen refined movement', '#e8eef9')
    box(7.75, 3.03, 2.08, .86, '3D coordinates\nraw generator output', '#edf4fb')
    for a,b in [(1.94,2.23),(4.37,4.65),(7.43,7.72)]:
        arrow((a,3.46),(b,3.46))

    ax.plot([.10,9.88],[2.69,2.69],color='#cad3dc',linewidth=.9)
    ax.text(.10, 2.36, 'PHYSICAL LEARNING', weight='bold', color='#7b4928', fontsize=12)
    ax.text(3.2, 2.36, 'Train matched copies; combine their updates.', color='#405366')
    box(.12, .83, 1.8, .97, 'Base generator\n'+r'$\theta_{\mathrm{base}}$', '#edf4fb')
    box(2.37, 1.40, 2.14, .64, 'Force-shifted targets\n'+r'$\theta_{\mathrm{physical}}$', '#faeee1')
    box(2.37, .43, 2.14, .64, 'Unmodified targets\n'+r'$\theta_{\mathrm{replay}}$', '#f4f0e8')
    box(5.03, .83, 2.18, .97, 'Difference of updates\n'+r'$\Delta\theta=\theta_{\mathrm{physical}}-\theta_{\mathrm{replay}}$', '#faeee1')
    box(7.75, .83, 2.08, .97, 'Final generator\n'+r'$\theta_{\mathrm{final}}=\theta_{\mathrm{base}}+\Delta\theta$', '#e7f1e5')
    arrow((1.95,1.46),(2.34,1.71))
    arrow((1.95,1.13),(2.34,.76))
    arrow((4.54,1.71),(5.0,1.47))
    arrow((4.54,.76),(5.0,1.13))
    arrow((7.24,1.31),(7.72,1.31))
    ax.text(3.44,.12,'Both also learn from the same original reference structures.',
            ha='center',fontsize=9,color='#405366')
    ax.text(8.79,.48,'Use for generation.',ha='center',fontsize=9,color='#405366')
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    for suffix in ['pdf','svg','png']:
        fig.savefig(args.out/('method.'+suffix),dpi=200,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)


if __name__ == '__main__':
    main()
