"""Build anonymous code-and-weight archives from the verified release directory."""
import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','release','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();release=a.release.resolve();a.out.mkdir(parents=True,exist_ok=True)
    # Release source includes the implementation and experiment runners, without
    # cluster launchers, environments, repository metadata, or workspace notes.
    for directory in ['cfm_mol','scripts','tests','configs']:
        for source in sorted((root/directory).rglob('*')):
            if not source.is_file() or source.is_symlink() or '__pycache__' in source.parts:continue
            if source.suffix not in ['.py','.json','.yaml','.yml']:continue
            target=release/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
            text=source.read_text()
            # Site-specific defaults become explicit local placeholders. The
            # source algorithms and saved tensor values are unchanged.
            text=text.replace('/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027','.')
            text=text.replace('/n/home04/yulili/bgfm','.')
            text=text.replace('/n/holylabs/woo_lab/Lab/yulili/bgfm/baselines/edm','vendor/edm')
            text=re.sub(r'/n/(?:home\d+|holylabs|netscratch)/[^\s\"\'\)\]\},;]+', '/path/to/local_resource', text)
            text=re.sub(r'/home/(?:yulili|renhaoz)[^\s\"\'\)\]\},;]*','/path/to/local_resource',text)
            if source.suffix=='.py':ast.parse(text,filename=str(source))
            target.write_text(text)
    shutil.copy2(root/'README.md',release/'README.md')
    shutil.copy2(root/'requirements-inference.txt',release/'requirements-inference.txt')
    project=(root/'pyproject.toml').read_text()
    project=re.sub(r'^authors\s*=.*\n','',project,flags=re.M)
    project=project.replace('Product-manifold constrained flow matching for 3D molecules.',
                            'Molecular generation with harmonic sources and learned endpoint corrections.')
    (release/'pyproject.toml').write_text(project)
    manifest=json.loads((release/'weights/manifest.json').read_text())
    main_models={name:value for name,value in manifest['models'].items() if name.startswith('latentspring_s')}
    needed={v for row in main_models.values() for k,v in row.items() if k in ['parent','geometry','physical','hydrogen']}
    main_manifest=dict(manifest,models=main_models,files={k:v for k,v in manifest['files'].items() if k in needed},
        source_records={k:v for k,v in manifest['source_records'].items() if k in needed})
    payloads={}
    accepted_folders={'cfm_mol','scripts','tests','configs','vendor','results','examples','verification','weights'}
    for source in sorted(release.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts:continue
        name=source.relative_to(release).as_posix();parts=Path(name).parts
        if name=='scripts/research/package_submission_release.py':continue
        if parts[0] not in accepted_folders and name not in ['README.md','requirements.txt','requirements-inference.txt','requirements-evaluation.txt','pyproject.toml']:continue
        if source.suffix in ['.pyc','.log'] or source.name.startswith('verification-'):continue
        data=source.read_bytes()
        if source.suffix in ['.json','.yaml','.yml','.txt','.toml','.py']:
            text=data.decode()
            text=text.replace('/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027','.')
            text=re.sub(r'/n/(?:home\d+|holylabs|netscratch)/[^\s\"\'\)\]\},;]+','/path/to/local_resource',text)
            if re.search(r'yulili|yurika1030sakura|Valarzz|renhaozhang@',text):
                raise ValueError('Remaining author identifier: '+name)
            data=text.encode()
        if name=='README.md':
            text=data.decode()
            text=re.sub(r'Download the archives from the \[model release\]\(https://github\.com/[^)]+\)\.\n\n','',text)
            data=text.encode()
        payloads[name]=data
    # The manuscript link in the public README is omitted from the anonymous archive.
    readme=payloads['README.md'].decode()
    readme=readme.replace('[LatentSpring: Harmonic Sources and Physical Corrections for Molecular Generation](paper/latentspring.pdf)',
                          '*LatentSpring: Harmonic Sources and Physical Corrections for Molecular Generation*')
    payloads['README.md']=readme.encode()
    receipts={}
    for label in ['code','main','full']:
        selected={}
        for name,data in payloads.items():
            if name.startswith('weights/'):
                if label=='code':continue
                if label=='main':
                    base=Path(name).name
                    if base=='manifest.json':data=(json.dumps(main_manifest,indent=2)+'\n').encode()
                    elif base not in needed:continue
            selected[name]=data
        hashes={name:hashlib.sha256(value).hexdigest() for name,value in selected.items()}
        selected['SHA256.json']=(json.dumps(hashes,indent=2)+'\n').encode()
        filename={'code':'latentspring_code.zip','main':'latentspring_supplement.zip','full':'latentspring_supplement_full.zip'}[label]
        path=a.out/filename
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name,value in selected.items():
                info=zipfile.ZipInfo('latentspring/'+name,date_time=(2026,9,25,0,0,0));info.external_attr=0o100644<<16
                z.writestr(info,value,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name,digest in hashes.items():assert hashlib.sha256(z.read('latentspring/'+name)).hexdigest()==digest
        receipts[label]=dict(file=filename,bytes=path.stat().st_size,sha256=sha(path),files=len(selected),
            weight_files=0 if label=='code' else len(needed) if label=='main' else len(manifest['files']))
        print(json.dumps(receipts[label]),flush=True)
    (a.out/'archives.json').write_text(json.dumps(dict(complete=True,archives=receipts,
        exact_tensor_weights=True,optimizer_states_included=False,private_paths_removed=True,
        git_history_included=False,readme_files=1),indent=2)+'\n')


if __name__=='__main__':main()
