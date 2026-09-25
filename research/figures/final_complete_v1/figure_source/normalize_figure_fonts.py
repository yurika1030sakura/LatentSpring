"""Rebuild inherited figures from supplied normalized SVGs, without Inkscape.

normalized_svg/ freezes supplied layouts; original_svg/ supplies glyph defs.
Data paths, coordinate scales and intervals are preserved.
"""
from pathlib import Path
import copy, hashlib, json, re, shutil
import cairosvg, fitz
from lxml import etree as E
R=Path(__file__).resolve().parent
OUT=R.parent/'figures'
NS='http://www.w3.org/2000/svg'

def missing_references(root):
    refs=set()
    for node in root.iter():
        for key,value in node.attrib.items():
            refs.update(re.findall(r'url\(#([^)]*)\)',value))
            if key.rsplit('}',1)[-1]=='href' and value.startswith('#'): refs.add(value[1:])
    return sorted(refs-{n.get('id') for n in root.iter() if n.get('id')})

def main():
    OUT.mkdir(exist_ok=True)
    audit=R/'audit';audit.mkdir(exist_ok=True)
    receipts=[]
    for source in sorted((R/'normalized_svg').glob('*.svg')):
        root=E.parse(str(source)).getroot()
        original=E.parse(str(R/'original_svg'/source.name)).getroot()
        lookup={n.get('id'):n for n in original.iter() if n.get('id')}
        defs=E.SubElement(root,'{'+NS+'}defs');restored=[]
        while missing:=missing_references(root):
            absent=[name for name in missing if name not in lookup]
            if absent:raise ValueError((source.name,absent))
            for name in missing:
                defs.append(copy.deepcopy(lookup[name]));restored.append(name)
        width=float(root.get('viewBox').split()[2]);scale=396/width;changes=[]
        if source.stem=='rotor_work':
            # Exact log ticks and y locations; enlarge base glyphs to 8 printed pt.
            for ident,y in [('text_22',255.194845),('text_23',178.852597)]:
                group=root.xpath(f"//*[@id='{ident}']")[0]
                group.set('transform',f'translate(311,{y}) scale({1/scale}) translate(-311,{-y})')
            changes.append('Restored 10^0 and 10^1 log ticks; 8 pt base glyphs with mathematical superscripts.')
        if source.stem=='source_training':
            # Only annotations move, by five printed points; data paths stay fixed.
            shift=5/(scale*0.7500097925003469)
            for text in root.findall('.//{'+NS+'}text'):
                if ''.join(text.itertext()).strip() in ['Mean +3.89 pp','4 of 5 runs improve']:
                    text.set('transform',f'translate({-shift},0) '+text.get('transform',''))
            changes.append('Moved upper-right annotations 5 pt inward.')
        target=OUT/source.name
        if missing_references(root):raise ValueError((source.name,missing_references(root)))
        E.ElementTree(root).write(str(target),encoding='utf-8',xml_declaration=True)
        cairosvg.svg2pdf(url=str(target),write_to=str(target.with_suffix('.pdf')))
        with fitz.open(target.with_suffix('.pdf')) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(4,4),alpha=False).save(target.with_suffix('.png'))
        manuscript=R.parent/'manuscript/figures'
        if manuscript.is_dir():shutil.copyfile(target.with_suffix('.pdf'),manuscript/target.with_suffix('.pdf').name)
        receipts.append({'figure':target.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'restored_definitions':restored,'changes':changes,'unresolved_svg_references':missing_references(root),'curves_and_points':'preserved from supplied normalized SVG'})
    (audit/'font_changes.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print('Rebuilt',len(receipts),'inherited figures with complete SVG references.')

if __name__=='__main__':main()
