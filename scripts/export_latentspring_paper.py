#!/usr/bin/env python3
"""Export the current manuscript as a self-contained Overleaf source project."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bib_entries(text):
    entries = {}
    for match in re.finditer(r'(?m)^@\w+\{([^,]+),', text):
        opening = text.index('{', match.start())
        depth = 0
        for end in range(opening, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                entries[match.group(1)] = text[match.start():end+1]
                break
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    paper = root/'paper'
    pending = [Path('tree_working.tex')]
    seen = set()
    used = set()
    source_hashes = {}
    figures = {}
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        source = paper/relative
        text = source.read_text()
        source_hashes[str(source.relative_to(root))] = sha(source)
        for stem in re.findall(r'\\input\{([^}]+)\}', text):
            pending.append(Path(stem+'.tex'))
        for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])?\{([^}]+)\}', text):
            used.update(k.strip() for k in group.split(','))
        def image_path(match):
            original = (paper/match.group(2)).resolve()
            target = Path('figures')/original.name
            digest = sha(original)
            if str(target) in figures and figures[str(target)] != digest:
                raise ValueError('Conflicting figure basenames')
            figures[str(target)] = digest
            (out/target).parent.mkdir(exist_ok=True)
            shutil.copyfile(original, out/target)
            source_hashes[str(original.relative_to(root))] = digest
            return match.group(1)+'{'+target.as_posix()+'}'
        text = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}', image_path, text)
        text = text.replace(r'\bibliography{refs,tree_refs}', r'\bibliography{refs}')
        destination = out/('main.tex' if relative.name == 'tree_working.tex' else relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    entries = {}
    for filename in ['refs.bib', 'tree_refs.bib']:
        file = paper/filename
        source_hashes['paper/'+filename] = sha(file)
        for key, entry in bib_entries(file.read_text()).items():
            if key in entries and entries[key] != entry:
                raise ValueError('Duplicate bibliography key: '+key)
            entries[key] = entry
    missing = used-entries.keys()
    if missing:
        raise ValueError('Missing citations: '+str(sorted(missing)))
    (out/'refs.bib').write_text('\n\n'.join(entries[k] for k in sorted(used))+'\n')
    for filename in ['iclr2027_conference.sty', 'iclr2027_conference.bst', 'natbib.sty', 'fancyhdr.sty']:
        shutil.copyfile(paper/filename, out/filename)
        source_hashes['paper/'+filename] = sha(paper/filename)
    for relative in ['research/figures/molecular_overview_v2/generation.gif',
                     'research/figures/molecular_rotor_v2/rotor_scan.gif']:
        source = root/relative
        if source.exists():
            target = out/'supplement'/source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            source_hashes[relative] = sha(source)
    (out/'README.md').write_text('# LatentSpring\n\n'
        'Physics-Informed Molecular Flow Matching from Atomic Composition.\n\n'
        'Set `main.tex` as the main document and compile with pdfLaTeX and BibTeX.\n'
        'This project includes every referenced section, figure, bibliography entry, '
        'and local style file. The manuscript is an anonymous ICLR 2027 submission draft.\n\n'
        'When included, `supplement/generation.gif` shows an actual recorded coordinate-flow '
        'trajectory, and `supplement/rotor_scan.gif` shows a prescribed methyl rotation. '
        'These animations are supplementary files; the PDF uses static snapshots. '
        'Neither animation is a claim of physical molecular-dynamics simulation.\n')
    files = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    manifest = dict(format='latentspring_overleaf_v1',source_commit=commit,
        main_document='main.tex',citation_keys=sorted(used),source_sha256=source_hashes,
        files_sha256=files)
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(out=str(out),files=len(files),citations=len(used))))


if __name__ == '__main__':
    main()
