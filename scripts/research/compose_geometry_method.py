"""Compact method overview using a verified, unchanged raw molecular output."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image


def main():
    root=Path(__file__).resolve().parents[2];out=root/'research/figures/geometric_method_v1';render=out/'outline'
    images={n:Image.open(render/(n+'.png')).convert('RGBA') for n in ['source','output']}
    boxes=np.array([im.getchannel('A').getbbox() for im in images.values()]);lo=boxes[:,:2].min(0)-20;hi=boxes[:,2:].max(0)+20;side=max(hi-lo);center=(lo+hi)/2
    crop=tuple(np.r_[center-side/2,center+side/2]);images={k:v.crop(crop) for k,v in images.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stixsans'})
    fig=plt.figure(figsize=(5.5,3.65),facecolor='white');ink='#273847';teal='#197d78';blue='#487b9b';gold='#bb864b';gray='#6d7b87'
    def text(x,y,s,fs=9,ha='left',color=ink,weight='normal'):
        return fig.text(x,y,s,fontsize=fs,ha=ha,va='center',color=color,weight=weight,linespacing=1.45)
    def arrow(a,b,c=gray):fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=9,lw=1.05,color=c))
    def box(x,y,w,h,label,c=blue):
        fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,boxstyle='round,pad=.005,rounding_size=.008',facecolor='white',edgecolor=c,lw=1.));text(x+w/2,y+h/2,label,8.6,'center',c)
    def picture(im,rect):ax=fig.add_axes(rect);ax.imshow(im);ax.axis('off')
    text(.02,.955,'A  From atomic composition to a 3D structure',10.5,weight='bold')
    text(.02,.895,r'$\mathrm{C_7H_{15}N}$  ·  neutral singlet',9,color=gray)
    text(.13,.825,'Harmonic source',8.6,'center');text(.665,.825,'Raw 3D output',8.6,'center');text(.9,.825,'Inferred graph',8.6,'center')
    picture(images['source'],[.005,.57,.25,.24]);picture(images['output'],[.55,.57,.24,.24]);picture(Image.open(out/'connectivity.png'),[.785,.59,.21,.21])
    box(.30,.69,.20,.10,'Flow velocity $v$',blue);box(.30,.54,.20,.10,'Corrections\n$g$ then $f$',teal)
    arrow((.24,.735),(.294,.735),blue);arrow((.40,.685),(.40,.648),teal);arrow((.505,.588),(.56,.645),teal)
    text(.13,.535,r'$X_0\sim q_0$',10,'center');text(.665,.535,r'$X_{\rm gen}$',10,'center')
    text(.13,.472,'Auxiliary springs',8.1,'center',gray);text(.58,.472,r'$v^{\rm corr}=v+g+\eta f$',10,'center',teal)
    fig.add_artist(Line2D([.02,.98],[.423,.423],transform=fig.transFigure,color='#dbe2e7',lw=.7))
    text(.02,.373,'B  Learn geometric and physical corrections',10.5,weight='bold')
    box(.025,.24,.32,.072,'Perturbed reference',blue);box(.50,.24,.45,.072,r'Geometry target $(X_{\rm ref}-H)/(1-p)$',teal)
    arrow((.352,.276),(.49,.276),teal)
    box(.025,.12,.32,.072,'Generated endpoint',blue);box(.50,.12,.45,.072,'eSEN force → physical target',gold)
    arrow((.352,.156),(.49,.156),gold)
    text(.50,.035,'Two small learned fields  ·  No sampling energy queries',8.6,'center',gray)
    for ext in ['pdf','svg','png']:fig.savefig(out/('geometric_method.'+ext),dpi=360,facecolor='white')
    Image.open(out/'geometric_method.png').convert('L').save(out/'geometric_method_grayscale.png')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'figure_manifest.json').write_text(json.dumps(dict(complete=True,scene_sha256=sha(out/'scenes.json'),render_sha256=sha(render/'render_manifest.json'),files={n:sha(out/n) for n in ['geometric_method.pdf','geometric_method.svg','geometric_method.png','geometric_method_grayscale.png','connectivity.png','connectivity.svg','raw_output.sdf','raw_output_coordinates.txt']},same_source_output_camera_scale=True,geometry_optimized=False,text_minimum_pt=8.1,width_inches=5.5,height_inches=3.65),indent=2)+'\n')


if __name__=='__main__':main()
