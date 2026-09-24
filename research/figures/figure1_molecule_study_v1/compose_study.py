"""Union-crop paired molecular renders without changing coordinates or scale."""
from pathlib import Path
import json, hashlib
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).parent
styles=['outline','soft','polished','sculpted','balanced']
images={(style,name):Image.open(root/style/(name+'.png')).convert('RGBA') for style in styles for name in ['source','output']}
boxes=[im.getchannel('A').getbbox() for im in images.values()]
margin=32
box=(max(0,min(b[0] for b in boxes)-margin),max(0,min(b[1] for b in boxes)-margin),min(1800,max(b[2] for b in boxes)+margin),min(1800,max(b[3] for b in boxes)+margin))
records=[]
for (style,name),im in images.items():
 out=root/style/(name+'_paired_crop.png')
 im.crop(box).save(out)
 records.append({'style':style,'name':name,'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
font=ImageFont.truetype('/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',25)
canvas=Image.new('RGB',(540*len(styles),920),'white');d=ImageDraw.Draw(canvas)
for i,style in enumerate(styles):
 d.text((540*i+22,14),style,fill='#253442',font=font)
 for j,name in enumerate(['output','source']):
  crop=images[style,name].crop(box)
  crop.thumbnail((510,350),Image.LANCZOS)
  canvas.paste(crop,(540*i+15,70+420*j),crop)
  d.text((540*i+22,380+420*j),name,fill='#61717D',font=font)
canvas.save(root/'style_comparison.png')
(root/'crop_manifest.json').write_text(json.dumps({'shared_crop_pixels':box,'original_pixels':[1800,1800],'crop_includes_every_nontransparent_pixel':True,'same_crop_for_every_style_and_paired_scene':True,'geometry_changed':False,'records':records},indent=2)+'\n')
