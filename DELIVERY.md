# BGFM ICLR audit delivery — 2026-09-08

Start with [the Chinese review](audit/20260908/REVIEW.md), then the
[working manuscript](paper/main.pdf). This is a revised research working copy,
not a claim of ICLR readiness or a replacement for the required new experiments.

## Contents and provenance

- Original source HEAD: `e00597bc461348026ef81ad768f1052bf16801c6`.
- `audit/20260908/changes.patch`: all changed/new UTF-8 source and audit files,
  excluding this delivery manifest and the patch itself.
- `audit/20260908/changes_manifest.json`: file hashes, additions/modifications,
  and binary artifacts (these are included in this complete snapshot).
- `audit/20260908/validation.json` and `validation_logs/`: actual test/build logs.
- `audit/20260908/evidence.json`: per-seed results and input hashes.
- `audit/20260908/checkpoint_smoke.json`: real-checkpoint numerical/gradient check.

The snapshot includes the original tracked evidence files, the validated
revision, and a symbolic link to the existing laboratory environments. It does
not copy checkpoints, datasets or Python environments. It has no Git metadata;
the base revision and patch make the changes reviewable without altering the
original checkout. Original archived result records were not changed.

The original `/n/home04/yulili/bgfm` checkout remains untouched because its
AGENTS.md prohibits writes into home. Build outputs default to `/tmp`.

## Reproduce from this directory

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export FLOWMOL_PY="$PWD/envs/flowmol/bin/python"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/bgfm-mpl \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$FLOWMOL_PY" -m pytest tests/ -q -o cache_dir=/tmp/bgfm-pytest
python scripts/audit_iclr_evidence.py --out /tmp/bgfm-evidence
bash paper/build.sh /tmp/bgfm-paper-build
```

Validation completed: **74 tests passed**, revised manuscript **9 main pages**;
new builder rejects the unmodified original 10-page manuscript with exit 1.
No new molecular training, optimizer update, SLURM job or external publication
was performed. The new clamped-density smoke configuration remains an
experimental template; read the review's experiment gates before using it.
