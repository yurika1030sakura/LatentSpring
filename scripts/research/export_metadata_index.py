#!/usr/bin/env python3
"""Export a completed source replay into a checked immutable-reader SQLite index."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    args=p.parse_args();progress=args.source/'progress.json';state=json.loads(progress.read_text())
    if not state['complete']:raise ValueError('The producer must finish before index export')
    source=args.source/'source_index.sqlite';destination=args.source/'source_index_readonly.sqlite'
    temporary=args.source/'source_index_readonly.sqlite.partial'
    if destination.exists() or temporary.exists():raise FileExistsError('Refusing to overwrite an index export')
    wal=source.with_name(source.name+'-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Nonempty WAL remains; wait for clean producer shutdown before export')
    # Immutable mode avoids cross-host WAL shared-memory access. The producer
    # is already complete and its journal must have been checkpointed above.
    src=sqlite3.connect('file:'+str(source.resolve())+'?mode=ro&immutable=1',uri=True)
    dst=sqlite3.connect(temporary)
    src.backup(dst);src.close()
    dst.execute('PRAGMA journal_mode=DELETE')
    integrity=dst.execute('PRAGMA integrity_check').fetchall()
    if integrity!=[('ok',)]:raise ValueError(f'Index integrity failed: {integrity[:5]}')
    rows=dst.execute('SELECT count(*) FROM records').fetchone()[0]
    counts=dict(dst.execute('SELECT split,count(*) FROM records GROUP BY split').fetchall())
    if rows!=state['accepted'] or counts!=state['processed_seen']:raise ValueError('Export count mismatch')
    dst.close();temporary.replace(destination)
    report={'complete':True,'source_progress_sha256':sha(progress),'source_database_sha256':sha(source),
        'export_sha256':sha(destination),'export_path':str(destination.resolve()),'rows':rows,
        'split_counts':counts,'sqlite_integrity_check':'ok','journal_mode':'DELETE',
        'read_mode':'file:<path>?mode=ro&immutable=1','source_protocol':state['protocol']}
    (args.source/'index_export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
