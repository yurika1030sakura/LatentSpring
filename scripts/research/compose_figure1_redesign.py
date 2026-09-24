"""Molecule-led Figure 1 with separate sampling, endpoint and learning panels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];a.out.mkdir(parents=True,exist_ok=True)
    source=root/'research/figures/geometric_method_v1'
    # The same rectangular union crop preserves the paired camera and scale.
    images={n:Image.open(a.render/(n+'_paired_crop.png')).convert('RGBA') for n in ['source','output']}
    assert images['source'].size==images['output'].size
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stixsans'})
    fig=plt.figure(figsize=(5.5,4.85),facecolor='white')
    ink='#25364a';gray='#68798a';blue='#4e7199';teal='#16847f';ochre='#b37438';rule='#dce3e8'
    def text(x,y,s,fs=9,ha='left',color=ink,weight='normal'):
        return fig.text(x,y,s,fontsize=fs,ha=ha,va='center',color=color,weight=weight,linespacing=1.35)
    def arrow(a,b,color=gray,lw=1.1,style='-|>',connectionstyle=None):
        kwargs={} if connectionstyle is None else dict(connectionstyle=connectionstyle)
        fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle=style,mutation_scale=9,lw=lw,color=color,**kwargs))
    def box(x,y,w,h,color,fill='white'):
        fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,boxstyle='round,pad=.006,rounding_size=.013',facecolor=fill,edgecolor=color,lw=1.))
    def picture(im,rect):
        ax=fig.add_axes(rect);ax.imshow(im);ax.axis('off')
    def line(y):fig.add_artist(Line2D([.02,.98],[y,y],transform=fig.transFigure,color=rule,lw=.65))

    text(.02,.966,'a  From composition to 3D coordinates',10.5,weight='bold')
    text(.02,.910,r'Input: $\mathrm{C_7H_{15}N}$  ·  neutral singlet',8.7,color=gray)
    text(.18,.857,'Harmonic source',9,'center',blue)
    text(.82,.857,'Raw generated structure',9,'center',ink)
    picture(images['source'],[.005,.505,.35,.35])
    picture(images['output'],[.645,.505,.35,.35])
    box(.39,.666,.22,.100,blue,'#f5f8fb')
    text(.50,.716,'Flow matching\n+ corrections',9,'center',blue)
    arrow((.333,.716),(.381,.716),blue);arrow((.619,.716),(.668,.716),teal)
    text(.50,.620,'32 integration steps',8.2,'center',gray)
    text(.17,.511,r'$X_0\sim q_0$',10,'center',blue)
    text(.17,.478,'Auxiliary springs',8.1,'center',gray)
    text(.72,.511,r'$X_{\rm gen}$',10,'center')
    picture(Image.open(source/'connectivity.png'),[.80,.465,.175,.11])
    line(.445)

    text(.02,.410,'b  Shared endpoint correction',10.5,weight='bold')
    text(.98,.410,'At each sampling step',8.2,'right',gray)
    text(.025,.353,r'FM: $H=X+(1-p)v$',9)
    text(.025,.285,r'Diffusion: $H=(X-\sigma_t\epsilon)/\alpha_t$',9)
    arrow((.350,.353),(.421,.336),blue)
    arrow((.350,.285),(.421,.336),blue)
    box(.44,.295,.205,.078,teal,'#f1f8f7')
    text(.5425,.351,'Geometry',8.8,'center',teal)
    text(.5425,.319,r'$g=G_\psi(H)$',9,'center',teal)
    arrow((.654,.336),(.740,.336),teal)
    text(.697,.368,r'$H_g$',9,'center',teal)
    box(.755,.295,.215,.078,ochre,'#fcf7f2')
    text(.8625,.351,'Physics',8.8,'center',ochre)
    text(.8625,.319,r'$f=A_\phi(X,H_g)$',9,'center',ochre)
    arrow((.5425,.288),(.602,.265),teal,lw=.95)
    arrow((.8625,.288),(.810,.265),ochre,lw=.95)
    text(.705,.239,r'$\Delta H=(1-p)(g+\eta f)$',10,'center',ink)
    line(.193)

    text(.02,.157,'c  Learn the corrections offline',10.5,weight='bold')
    text(.025,.098,'Perturbed references',8.6,color=gray)
    arrow((.293,.098),(.341,.098),teal)
    text(.355,.098,r'$G_\psi$',10,color=teal)
    text(.535,.098,r'eSEN forces at endpoints',8.6,color=gray)
    arrow((.853,.098),(.909,.098),ochre)
    text(.925,.098,r'$A_\phi$',10,color=ochre)
    text(.50,.032,'14,853 correction parameters  ·  0 sampling energy calls',8.3,'center',gray)

    for ext in ['pdf','svg','png']:fig.savefig(a.out/('figure1.'+ext),dpi=400,facecolor='white')
    Image.open(a.out/'figure1.png').convert('L').save(a.out/'figure1_grayscale.png')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out/'figure_manifest.json').write_text(json.dumps(dict(complete=True,source_scene=str(source/'scenes.json'),source_scene_sha256=sha(source/'scenes.json'),render_manifest=str(a.render/'render_manifest.json'),render_manifest_sha256=sha(a.render/'render_manifest.json'),same_source_output_camera_scale=True,geometry_optimized=False,width_inches=5.5,height_inches=4.85,minimum_font_pt=8.1,files={n:sha(a.out/n) for n in ['figure1.pdf','figure1.svg','figure1.png','figure1_grayscale.png']},molecular_display='Same verified C7H15N raw sample and replayed harmonic source. Changed rendering and layout only.'),indent=2)+'\n')


if __name__=='__main__':main()
