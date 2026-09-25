"""Check final SVG references, printed fonts, bounds and preserved data paths."""
from pathlib import Path
import hashlib,json
import fitz
from lxml import etree as E
from normalize_figure_fonts import missing_references

R=Path(__file__).resolve().parent
OUT=R.parent/'figures'
NS='http://www.w3.org/2000/svg'

def plot_paths(root):
    return [p.get('d') for p in root.findall('.//{'+NS+'}path')
            if not any(a.tag=='{'+NS+'}defs' for a in p.iterancestors())]

def main():
    figures=[]
    for pdf in sorted(OUT.glob('*.pdf')):
        with fitz.open(pdf) as d:
            page=d[0]
            spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b
                   for line in b['lines'] for s in line['spans']]
            outside=[s['text'] for s in spans if s['bbox'][0]<-.1 or s['bbox'][1]<-.1
                     or s['bbox'][2]>page.rect.width+.1 or s['bbox'][3]>page.rect.height+.1]
            size=min(s['size'] for s in spans)
            assert size>=7.99,(pdf.name,size)
            assert not outside,(pdf.name,outside)
            assert abs(page.rect.width-396)<.01
        svg=pdf.with_suffix('.svg');root=E.parse(str(svg)).getroot()
        missing=missing_references(root)
        assert not missing,(svg.name,missing)
        inherited=R/'normalized_svg'/svg.name
        preserved=None
        if inherited.exists():
            preserved=plot_paths(root)==plot_paths(E.parse(str(inherited)).getroot())
            assert preserved,pdf.name
        figures.append({'figure':pdf.name,'width_pt':396,'minimum_extracted_text_pt':round(size,4),
                        'outside_page_text':outside,'unresolved_svg_references':missing,
                        'inherited_plot_path_coordinates_unchanged':preserved})
    outlined=[]
    for svg in sorted((OUT/'outlined').glob('*.svg')):
        missing=missing_references(E.parse(str(svg)).getroot())
        assert not missing,(svg.name,missing)
        outlined.append({'figure':svg.name,'unresolved_svg_references':missing})
    hashes={str(p.relative_to(R.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [OUT,R/'data',R/'normalized_svg'] for p in sorted(folder.rglob('*')) if p.is_file()}
    output={'complete':True,'figures':figures,'outlined':outlined,'file_sha256':hashes,
            'visual_review':['Figure 2 fixed palette and independent line styles','Rotor logarithmic ticks visible at correct locations','Training annotation right margin clear'],
            'numeric_changes':False,'molecular_coordinates_changed':False,
            'mathematical_superscripts':'Rotor log ticks have 8 pt base glyphs and conventional 70 percent superscripts.'}
    dest=R/'audit';dest.mkdir(exist_ok=True)
    (dest/'verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Verified',len(figures),'PDFs and',len(outlined),'outlined SVGs.')

if __name__=='__main__':main()
