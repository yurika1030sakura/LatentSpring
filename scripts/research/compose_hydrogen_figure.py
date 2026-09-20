"""Compose true-coordinate PyMOL panels and paired energy estimates at paper size."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scripts.research.train_electronic_fm import sha

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'research/figures/hydrogen_readout_v1';render=out/'outline_v2'
    scenes=json.loads((out/'scenes.json').read_text());receipt=json.loads((render/'render_manifest.json').read_text())
    assert len(receipt['images'])==3 and all(v['atoms']==23 for v in receipt['images'])
    for v in receipt['images']:assert sha(render/(v['name']+'.png'))==v['sha256']
    names=['base','radial','molecule_start0'];images=[Image.open(render/(name+'.png')).convert('RGBA') for name in names]
    boxes=[i.getchannel('A').getbbox() for i in images];box=(min(x[0] for x in boxes)-18,min(x[1] for x in boxes)-18,max(x[2] for x in boxes)+18,max(x[3] for x in boxes)+18)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7})
    fig=plt.figure(figsize=(5.5,3.25),facecolor='white');ink='#253544';teal='#238078';orange='#b87640'
    fig.text(.035,.95,'a',weight='bold',fontsize=10,color=ink)
    fig.text(.080,.95,'Hydrogen placement around one fixed heavy skeleton',fontsize=9,color=ink)
    for k,(name,title,status) in enumerate(zip(names,['Parent','Radial rule','Conditional flow'],['Detached hydrogen','Graph-valid','Graph-valid'])):
        ax=fig.add_axes([.025+k*.327,.52,.31,.35]);ax.imshow(images[k].crop(box));ax.axis('off')
        fig.text(.18+k*.327,.88,title,ha='center',color=ink,fontsize=9)
        fig.text(.18+k*.327,.505,status,ha='center',color=teal if k else orange,fontsize=8)
    fig.text(.035,.412,'b',weight='bold',fontsize=10,color=ink)
    fig.text(.080,.412,'Energy reduction relative to the radial rule',fontsize=9,color=ink)
    auditfile=ROOT/'research/evidence/hydrogen_physical_confirmation_audit_v1.json';audit=json.loads(auditfile.read_text())
    ax=fig.add_axes([.16,.135,.69,.20]);ax.axvline(0,color='#a8afb3',lw=.7)
    for y,(family,color) in enumerate([('gaga',orange),('fm',teal)]):
        d=audit['contrasts'][family+'_learned_minus_radial']['energy'];mean=-1000*d['mean'];lo,hi=-1000*np.array(d['ci95'])[::-1]
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,markersize=5,capsize=3,lw=1.5)
        ax.plot(-1000*np.array(d['by_seed']),[y+.18,y-.18],marker='|',ls='',color=color,markersize=6)
        ax.text(hi+.6,y,f'{mean:.1f}',va='center',color=color,fontsize=8)
    ax.set_xlim(-.4,25);ax.set_ylim(-.45,1.45);ax.set_yticks([0,1],['GAGA','FM']);ax.set_xticks([0,5,10,15,20,25]);ax.set_xlabel('Lower GFN2 energy (meV/atom)',labelpad=3)
    ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',length=3)
    for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    for extension in ['pdf','svg','png']:fig.savefig(out/('hydrogen_readout.'+extension),dpi=300,facecolor='white')
    Image.open(out/'hydrogen_readout.png').convert('L').save(out/'hydrogen_readout_grayscale.png')
    manifest=dict(scene_sha256=sha(out/'scenes.json'),render_manifest_sha256=sha(render/'render_manifest.json'),audit_sha256=sha(auditfile),
        outputs={ext:sha(out/('hydrogen_readout.'+ext)) for ext in ['pdf','svg','png']},width_inches=5.5,union_crop=box,
        selected_style='outline: restrained edges remain legible at final size; separate tan boron and purple iodine.',
        geometry='Shared heavy-atom frame and camera, no atom movement for rendering; all edges are distance contacts.',
        statistical_unit='Compositions, with two fitted models retained; small marks are fit means.')
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
