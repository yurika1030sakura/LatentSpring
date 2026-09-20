"""An endpoint-learning overview composed from measured molecular coordinates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image
from scripts.research.train_electronic_fm import sha

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'research/figures/endpoint_method_v1';data=json.loads((out/'scenes.json').read_text());render=out/'outline';receipt=json.loads((render/'render_manifest.json').read_text());images={};crops={}
    for item in receipt['images']:
        f=render/(item['name']+'.png');assert sha(f)==item['sha256'];images[item['name']]=Image.open(f).convert('RGBA')
    for names in [['source','output'],['endpoint','shifted_endpoint']]:
        bounds=np.array([images[n].getchannel('A').getbbox() for n in names]);lo=bounds[:,:2].min(0)-20;hi=bounds[:,2:].max(0)+20;side=max(hi-lo);center=(lo+hi)/2
        box=(center[0]-side/2,center[1]-side/2,center[0]+side/2,center[1]+side/2)
        for n in names:crops[n]=box
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stixsans'})
    fig=plt.figure(figsize=(5.5,3.55),facecolor='white');ink='#263542';blue='#496f92';orange='#b87640';gray='#657582'
    def text(x,y,s,size=8,color=ink,ha='left',weight='normal'):fig.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',weight=weight)
    def arrow(a,b,color=gray,curve=0):fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=8,lw=.95,color=color,connectionstyle=f'arc3,rad={curve}'))
    def box(x,y,w,h,s,color):
        fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,boxstyle='round,pad=.006,rounding_size=.01',fc='white',ec=color,lw=1.))
        text(x+w/2,y+h/2,s,8,color,ha='center')
    def molecule(name,rect):ax=fig.add_axes(rect);ax.imshow(images[name].crop(crops[name]));ax.axis('off')
    text(.02,.956,'a',10,weight='bold');text(.06,.956,'Composition to 3D coordinates',10,weight='bold')
    text(.98,.956,r'$\mathrm{C_{10}H_5FN_2O_2}$',10,blue,ha='right')
    text(.06,.900,'Input: atomic types, charge and spin',7.5,gray)
    text(.98,.900,'Charge 0; spin multiplicity 1',7.5,gray,ha='right')
    text(.15,.859,'Harmonic source',8.5,ha='center');text(.85,.859,'Generated structure',8.5,ha='center')
    molecule('source',[.014,.52,.273,.305]);molecule('output',[.717,.52,.273,.305])
    text(.15,.512,r'$X_0\sim q_0(\cdot\mid c)$',9,ha='center');text(.85,.512,r'$X_{\rm gen}\in\mathbb{R}^{20\times3}$',9,ha='center')
    box(.35,.735,.29,.094,'Frozen parent prediction',blue)
    box(.35,.595,.29,.094,'Learned correction',orange)
    arrow((.285,.72),(.343,.775),blue);text(.305,.822,r'$X,t,c$',8.5,blue,ha='center')
    arrow((.494,.731),(.494,.695),blue);text(.527,.712,r'$H$',8.5,blue)
    arrow((.648,.644),(.689,.685),orange);arrow((.648,.775),(.689,.719),blue)
    text(.697,.700,'+',11,gray,ha='center');arrow((.713,.700),(.749,.700),orange)
    text(.714,.839,'Integrate',7.6,gray,ha='center')
    text(.498,.535,r'$v^{\rm corr}=v+\eta A_\phi$',10,orange,ha='center')
    fig.add_artist(Line2D([.02,.98],[.46,.46],transform=fig.transFigure,color='#d9e0e5',lw=.7))
    text(.02,.412,'b',10,weight='bold');text(.06,.412,'Physical supervision at predicted endpoints',9.5,weight='bold')
    molecule('endpoint',[.005,.092,.23,.265]);molecule('shifted_endpoint',[.29,.092,.23,.265])
    text(.115,.080,'Provisional endpoint $H$',7.7,ha='center');text(.408,.080,r'Target $H+\delta(H)$',7.7,ha='center')
    arrow((.233,.216),(.289,.216),orange);text(.26,.328,'Force\nshift',7.7,orange,ha='center')
    box(.662,.150,.31,.150,'Symmetric pair features\nBounded scalars\nOpposite pairwise vectors',orange)
    arrow((.529,.216),(.651,.216),orange);text(.584,.295,'Supervise',7.5,orange,ha='center')
    text(.815,.093,r'$A_\phi\approx\delta(H)/(1-p)$',9,orange,ha='center')
    text(.026,.024,'Forces are used during training; sampling uses neural networks.',7.2,gray)
    for ext in ['pdf','svg','png']:fig.savefig(out/('endpoint_method.'+ext),dpi=350,facecolor='white')
    Image.open(out/'endpoint_method.png').convert('L').save(out/'endpoint_method_grayscale.png')
    (out/'figure_manifest.json').write_text(json.dumps(dict(scene_sha256=sha(out/'scenes.json'),render_sha256=sha(render/'render_manifest.json'),
        outputs={ext:sha(out/('endpoint_method.'+ext)) for ext in ['pdf','svg','png']},width_inches=5.5,height_inches=3.55,style='outline',
        geometry='All original atoms and coordinates retained; source tree RNG exactly replayed. Separate generation and training examples. Pairwise shared cameras and crops.'),indent=2)+'\n')

if __name__=='__main__':main()
