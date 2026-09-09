#!/usr/bin/env python3
"""Recover the public 2025-05-14 OMol25 4M archive with explicit provenance.

The training URL is the existing project's public Meta CDN source, verified to
return HTTP 200. This does not access the gated model repository or restricted
electronic-structure archives. Dataset license: CC BY 4.0 (official fairchem docs).
Downloads are resumable only when ETag and expected size still agree.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import time
import urllib.request


BASE_URL='https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514'


def write_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--extract',action='store_true')
    p.add_argument('--split',choices=['train_4M','val'],default='train_4M')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    url=f'{BASE_URL}/{args.split}.tar.gz'
    archive=args.out/f'{args.split}.tar.gz';partial=args.out/f'{args.split}.tar.gz.partial'
    manifest=args.out/'download_manifest.json'
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
        headers=dict(response.headers);length=int(response.headers['Content-Length']);etag=response.headers['ETag']
    previous=json.loads(manifest.read_text()) if manifest.exists() else None
    if previous and (previous['url']!=url or previous['expected_bytes']!=length or previous['etag']!=etag):
        raise ValueError('Remote object changed; refusing to reuse existing download')
    if (partial.exists() or archive.exists()) and previous is None:
        raise ValueError('Existing download has no provenance record')
    report=previous or {'complete':False,'url':url,'split':args.split,'expected_bytes':length,'etag':etag,
        'response_headers':headers,'license':'CC BY 4.0',
        'license_source':'https://fair-chem.github.io/omol25/',
        'first_checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_script_sha256':digest(Path(__file__))}
    write_json(manifest,report)
    if not archive.exists():
        for attempt in range(4):
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>length:raise ValueError('Partial archive larger than advertised object')
            if offset==length:break
            request=urllib.request.Request(url,headers={'Accept-Encoding':'identity',**({'Range':f'bytes={offset}-','If-Match':etag} if offset else {})})
            try:
                with urllib.request.urlopen(request,timeout=30) as response,partial.open('ab' if offset else 'wb') as output:
                    if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                        raise ValueError('Server did not honor the requested resume offset')
                    progress=offset;last_print=offset
                    for chunk in iter(lambda:response.read(8*1024*1024),b''):
                        output.write(chunk);progress+=len(chunk)
                        if progress-last_print>=512*1024*1024:
                            print(json.dumps({'downloaded_bytes':progress,'expected_bytes':length}),flush=True);last_print=progress
                if partial.stat().st_size!=length:raise IOError('Incomplete archive response')
                break
            except (OSError,TimeoutError) as exc:
                print(json.dumps({'attempt':attempt+1,'error':str(exc)}),flush=True)
                if attempt==3:raise
                time.sleep(2)
        if not partial.exists() or partial.stat().st_size!=length:raise IOError('Archive byte count mismatch')
        partial.replace(archive)
    if archive.stat().st_size!=length:raise ValueError('Existing archive size mismatch')
    report.update(archive=str(archive.resolve()),archive_sha256=digest(archive),download_complete=True)
    write_json(manifest,report)
    if args.extract:
        destination=args.out/f'{args.split}_extracted';staging=args.out/f'.{args.split}_extract_partial'
        if not destination.exists():
            staging.mkdir(exist_ok=True);marker=staging/'.archive_sha256'
            if marker.exists() and marker.read_text()!=report['archive_sha256']:raise ValueError('Extraction source changed')
            if not marker.exists() and any(staging.iterdir()):raise ValueError('Unowned partial extraction')
            marker.write_text(report['archive_sha256'])
            members=[]
            with tarfile.open(archive,'r|gz') as source:
                for member in source:
                    name=PurePosixPath(member.name)
                    if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe archive path')
                    target=staging.joinpath(*name.parts)
                    if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
                    if not member.isfile():raise ValueError('Archive links/special files are not supported')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with source.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
                    if target.stat().st_size!=member.size:raise IOError('Truncated extracted member')
                    members.append({'path':str(name),'bytes':member.size})
            write_json(staging/'extraction_manifest.json',{'archive_sha256':report['archive_sha256'],'members':members})
            staging.replace(destination)
        extraction=json.loads((destination/'extraction_manifest.json').read_text())
        if extraction['archive_sha256']!=report['archive_sha256']:raise ValueError('Extracted archive hash mismatch')
        report.update(extracted_directory=str(destination.resolve()),extracted_files=len(extraction['members']))
    report['complete']=True;write_json(manifest,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
