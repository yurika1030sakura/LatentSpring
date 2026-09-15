#!/bin/bash
#SBATCH -J lspring-gaga-dispatch
#SBATCH -A ryl_lab
#SBATCH -p serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?immutable confirmation source}"
PROJECT="${2:?project data root}"
RUN_ROOT="${3:?run root}"
# Dependency is supplied at submission. This avoids holding a third gpu_test
# submission slot while the two training seeds run.
python - "$SOURCE" "$PROJECT" "$RUN_ROOT" <<'PY'
import datetime,json,subprocess,sys
from pathlib import Path
source,project,run=map(Path,sys.argv[1:])
for seed in [0,1]:
    for method in ['distance','tree','baselines']:
        assert json.loads((run/'training'/f's{seed}'/method/'complete.json').read_text())['complete']
command=['sbatch','--parsable',str(source/'scripts/research/gaga_feedback_confirm.slurm'),str(source),str(project),str(run)]
result=subprocess.run(command,capture_output=True,text=True)
receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,
             returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
if result.returncode==0:
    receipt['job_id']=result.stdout.strip().split(';')[0]
    assert receipt['job_id'].isdigit()
(run/'confirmation_dispatched.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)
PY
