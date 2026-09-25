"""Export 600-dpi PNGs, font-outlined SVGs, and a vector figure collection."""
from pathlib import Path
import subprocess,shutil,fitz
root=Path(__file__).resolve().parent.parent;figures=root/'figures'
(figures/'outlined').mkdir(exist_ok=True)
for p in figures.glob('*.svg'):
    if shutil.which('inkscape'):
        subprocess.run(['inkscape',str(p),'--export-text-to-path','--export-type=svg','--export-plain-svg','--export-filename='+str(figures/'outlined'/p.name)],check=True)
    elif shutil.which('pdftocairo'):
        subprocess.run(['pdftocairo','-svg',str(p.with_suffix('.pdf')),str(figures/'outlined'/p.name)],check=True)
    else:
        raise RuntimeError('Outlined SVG export requires Inkscape or pdftocairo (Poppler).')
for p in figures.glob('*.pdf'):
    with fitz.open(p) as d:d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(p.with_suffix('.png'))
order=['figure1','main_comparison','components','geometric_transfer','organic_geometry','composition_effects','source_training','matched_connection','force_thresholds','hydrogen_readout','common_model_quality','rotor_work','quality_yield','primary_quality']
d=fitz.open()
for name in order:
    with fitz.open(figures/(name+'.pdf')) as f:d.insert_pdf(f)
d.set_toc([[1,f'Figure {i+1}: {name.replace("_"," ")}',i+1] for i,name in enumerate(order)])
d.save(root/'Figures.pdf',garbage=4,deflate=True)
