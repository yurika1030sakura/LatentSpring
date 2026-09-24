"""Compose a readable method figure from a checked raw molecular example."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch,Circle
from matplotlib.lines import Line2D
from PIL import Image


def main():
    root=Path(__file__).resolve().parents[2];out=root/'research/figures/round2_method_v1';data=json.loads((out/'scenes.json').read_text());render=out/'outline';images={n:Image.open(render/(n+'.png')).convert('RGBA') for n in ['source','output']}
    boxes=np.array([im.getchannel('A').getbbox() for im in images.values()]);lo=boxes[:,:2].min(0)-20;hi=boxes[:,2:].max(0)+20;side=max(hi-lo);center=(lo+hi)/2;crop=tuple(np.r_[center-side/2,center+side/2]);images={k:v.crop(crop) for k,v in images.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stixsans'})
    fig=plt.figure(figsize=(5.5,4.65),facecolor='white');ink='#273847';teal='#197d78';blue='#487b9b';gold='#bb864b';gray='#6d7b87'
    def text(x,y,s,fs=9,ha='left',color=ink,weight='normal'):
        return fig.text(x,y,s,fontsize=fs,ha=ha,va='center',color=color,weight=weight,linespacing=1.5)
    def arrow(a,b,c=gray):fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=9,lw=1.05,color=c))
    def box(x,y,w,h,label,c=blue):
        fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,boxstyle='round,pad=.006,rounding_size=.008',facecolor='white',edgecolor=c,lw=1.1));text(x+w/2,y+h/2,label,8.7,'center',c)
    def picture(im,rect):ax=fig.add_axes(rect);ax.imshow(im);ax.axis('off')
    text(.02,.96,'A  Composition to raw coordinates',10.5,weight='bold')
    text(.02,.905,r'$\mathrm{C_9H_9NO_2}$  ·  neutral singlet',9,color=gray)
    text(.13,.84,'Harmonic source',8.7,'center');text(.655,.84,'3D output',8.7,'center');text(.9,.84,'Connectivity',8.7,'center')
    picture(images['source'],[.005,.59,.23,.245]);picture(images['output'],[.55,.59,.23,.245]);picture(Image.open(out/'connectivity.png'),[.785,.605,.205,.21])
    box(.305,.752,.19,.062,'Frozen FM',blue);box(.305,.63,.19,.062,'Physical head',teal)
    arrow((.244,.724),(.30,.782),blue);arrow((.40,.748),(.40,.699),blue);text(.42,.724,r'$H$',9,color=blue)
    arrow((.50,.783),(.515,.740),blue);arrow((.50,.661),(.515,.698),teal)
    fig.add_artist(Circle((.520,.719),.014,transform=fig.transFigure,facecolor='white',edgecolor=gray,lw=.8))
    text(.520,.719,'+',10,'center',teal);arrow((.538,.719),(.558,.719),teal)
    text(.40,.573,r'$v^{\rm corr}=v+\eta A_\phi$',10,'center',teal)
    text(.13,.55,r'$X_0\sim q_0(\cdot\mid c)$',10,'center');text(.655,.55,r'$X_{\rm gen}$',10,'center')
    text(.02,.497,'Dashed links: auxiliary springs',8.3,color=gray)
    fig.add_artist(Line2D([.02,.98],[.465,.465],transform=fig.transFigure,color='#dbe2e7',lw=.7))
    text(.02,.425,'B  Learn the correction from forces',10.5,weight='bold')
    box(.025,.318,.21,.058,'Endpoint $H$',blue);box(.305,.318,.22,.058,'eSEN force $F_+(H)$',gold);box(.605,.318,.35,.058,r'Target $\delta(H)/(1-p)$',teal)
    arrow((.24,.347),(.294,.347),gold);arrow((.534,.347),(.594,.347),teal)
    text(.50,.272,'Offline regression of the small correction network',8.7,'center',gray)
    fig.add_artist(Line2D([.02,.98],[.237,.237],transform=fig.transFigure,color='#dbe2e7',lw=.7))
    text(.02,.198,'C  Apply the same coordinate correction',10.5,weight='bold')
    text(.025,.135,r'FM:  $H=X+(1-p)v$',10,color=blue)
    text(.025,.067,r'Diffusion:  $H=(X-\sigma_t\epsilon)/\alpha_t$',9.5,color=blue)
    arrow((.495,.135),(.562,.11),teal);arrow((.495,.067),(.562,.095),teal)
    box(.577,.064,.385,.084,r'$\Delta H=\eta(1-p)A_\phi$',teal)
    text(.975,.020,'7,106 parameters  ·  Sampling energy queries: 0',8.1,'right',gray)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for ext in ['pdf','svg','png']:fig.savefig(out/('method.'+ext),dpi=360,facecolor='white')
    Image.open(out/'method.png').convert('L').save(out/'method_grayscale.png')
    (out/'figure_manifest.json').write_text(json.dumps(dict(complete=True,scene_sha256=sha(out/'scenes.json'),render_sha256=sha(render/'render_manifest.json'),files={n:sha(out/n) for n in ['method.pdf','method.svg','method.png','method_grayscale.png','connectivity.png','connectivity.svg','raw_output.sdf','raw_output_coordinates.txt']},same_source_output_camera_scale=True,geometry_optimized=False,chemistry='True generated coordinates plus the inferred connectivity diagram; no generated-image chemistry used',text_minimum_pt=8.1,width_inches=5.5,height_inches=4.65),indent=2)+'\n')


if __name__=='__main__':main()
