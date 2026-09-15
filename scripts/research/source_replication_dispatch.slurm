#!/bin/bash
#SBATCH -J lspring-repeat-dispatch
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
python - "$@" <<'PY'
import datetime,json,subprocess,sys
from pathlib import Path
launcher,source,project,run,audit_job=sys.argv[1:]
result=subprocess.run(['sbatch','--parsable',launcher,source,project,run],capture_output=True,text=True)
receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
if result.returncode==0:
    job=result.stdout.strip().split(';')[0];assert job.isdigit();receipt['job_id']=job
Path(project,'runs/source_replication_v1/gpu_test_submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
if result.returncode==0:
    # The audit is submitted with a user hold, so cancelled predecessor jobs
    # cannot invalidate it before the actual GPU job is known.
    subprocess.run(['scontrol','update','JobId='+audit_job,'Dependency=afterok:'+job],check=True)
    subprocess.run(['scontrol','release',audit_job],check=True)
raise SystemExit(result.returncode)
PY
