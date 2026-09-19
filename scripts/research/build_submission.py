"""Build and verify canonical and standalone paper sources before publication."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(folder,stem,logdir):
    for index,command in enumerate([['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],
            ['bibtex',stem],['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],
            ['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex']]):
        p=subprocess.run(command,cwd=folder,text=True,capture_output=True)
        (logdir/f'{stem}_{index}.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-3000:])
    log=(folder/(stem+'.log')).read_text()
    assert not re.search(r'Overfull \\[hv]box|There were undefined references|Rerun to get cross-references right',log)
    assert not re.search(r'(Citation|Reference) .* undefined',log)
    aux=(folder/(stem+'.aux')).read_text()
    mainpages=int(re.search(r'\\newlabel\{endofmaintext\}\{\{[^}]*\}\{(\d+)\}',aux).group(1))
    info=subprocess.check_output(['pdfinfo',str(folder/(stem+'.pdf'))],text=True)
    total=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    assert mainpages<=9,mainpages
    return dict(main_scientific_pages=mainpages,total_pages=total,pdf_sha256=sha(folder/(stem+'.pdf')),
        unresolved_references=False,overfull_boxes=False,unstable_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--export',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    root=a.project.resolve();paper=root/'paper';export=a.export.resolve();export.mkdir(parents=True,exist_ok=False)
    logdir=export.parent/(export.name+'_logs');logdir.mkdir(exist_ok=True)
    canonical=build(paper,'tree_working',logdir);inputs={};files={};sources={}
    def visit(path,target):
        text=path.read_text();inputs[str(path.relative_to(root))]=sha(path)
        for match in re.finditer(r'\\input\{([^}]+)\}',text):
            relative=match.group(1);visit(paper/(relative+'.tex'),relative+'.tex')
        for match in list(re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)):
            relative=match.group(1);src=(paper/relative).resolve();dst='figures/'+src.name
            if dst in files:assert files[dst]==src
            files[dst]=src;inputs[str(src.relative_to(root))]=sha(src)
            text=text.replace('{'+relative+'}','{'+dst+'}')
        text=re.sub(r'\\bibliography\{[^}]+\}',lambda m:'\\bibliography{refs}',text)
        sources[target]=text
    visit(paper/'tree_working.tex','main.tex')
    bibname=re.search(r'\\bibliography\{([^}]+)\}',(paper/'tree_working.tex').read_text()).group(1)
    bibliographies=[paper/(name.strip()+'.bib') for name in bibname.split(',')]
    sources['refs.bib']='\n'.join(path.read_text() for path in bibliographies)
    for path in bibliographies:inputs[str(path.relative_to(root))]=sha(path)
    for name in ['iclr2027_conference.sty','iclr2027_conference.bst','fancyhdr.sty','natbib.sty']:files[name]=paper/name
    for name,src in [('generation.gif','research/figures/molecular_overview_v2/generation.gif'),
            ('rotor_scan.gif','research/figures/molecular_rotor_v2/rotor_scan.gif')]:files['supplement/'+name]=root/src
    for target,text in sources.items():
        dst=export/target;dst.parent.mkdir(exist_ok=True,parents=True);dst.write_text(text)
    for target,src in files.items():
        dst=export/target;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst)
        inputs[str(src.relative_to(root))]=sha(src)
    (export/'README.md').write_text('LatentSpring submission draft. Compile main.tex with pdfLaTeX, BibTeX, then pdfLaTeX twice.\nAnimations show generation or prescribed rotations, not molecular dynamics.\n')
    exported={str(p.relative_to(export)):sha(p) for p in sorted(export.rglob('*')) if p.is_file()}
    (export/'MANIFEST.json').write_text(json.dumps(dict(files=exported,canonical_inputs=inputs),indent=2)+'\n')
    standalone=build(export,'main',logdir)
    texts=[subprocess.check_output(['pdftotext','-layout',str(p),'-']) for p in [paper/'tree_working.pdf',export/'main.pdf']]
    assert texts[0]==texts[1],'Canonical/standalone text differs'
    for name,digest in exported.items():assert sha(export/name)==digest
    for name in ['latentspring.pdf','main.pdf','bgfm_paper.pdf']:shutil.copy2(paper/'tree_working.pdf',paper/name)
    with zipfile.ZipFile(paper/'latentspring_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(list(exported)+['MANIFEST.json']):z.write(export/name,name)
    source=(paper/'tree_working.tex').read_text()
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',source,re.S).group(1).strip()
    abstract=abstract.replace('\\%','%').replace('\\AA','Å').replace('--','–').replace('\\,',' ')
    (paper/'submission_abstract.txt').write_text('LatentSpring: Physics-Informed Molecular Flow Matching from Atomic Composition\n\n'+re.sub(r'\s+',' ',abstract)+'\n')
    receipt=dict(compiled=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        builds=dict(canonical=canonical,standalone=standalone),input_sha256=inputs,
        canonical_and_standalone_pdf_text_identical=True,export_manifest_verified=True,
        exported_source_files=len(exported),export_path=str(export),
        cited_references=len(re.findall(r'\\bibitem', (paper/'tree_working.bbl').read_text())),
        archive_sha256=sha(paper/'latentspring_overleaf.zip'),ledger='research/evidence/generator_reference_registry_v18.json')
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='input_sha256'}),flush=True)


if __name__=='__main__':main()
