#!/usr/bin/env python3
"""Reproduce the paired physical contrasts used in the factorial manuscript."""
import json
from pathlib import Path
import numpy as np
from scripts.research.audit_source_sc_energy import masked_intervals
from scripts.research.train_electronic_fm import sha


def main():
    root=Path(__file__).resolve().parents[2];evidence=root/'research/evidence'
    audit=evidence/'source_physical_factorial_audit_v1.json';record=json.loads(audit.read_text());assert record['complete']
    file=evidence/record['arrays_file'];assert sha(file)==record['arrays_sha256'];a=np.load(file)
    comparisons={}
    for kind in ['esen','xtb']:
        comparisons[kind]={}
        for label,left,right in [('gaussian_physical_minus_gaussian',2,0),('harmonic_physical_minus_harmonic',3,1)]:
            success=a[kind+'_success'][:,left]&a[kind+'_success'][:,right]
            common=success&a['graph'][:,left]&a['graph'][:,right];value={}
            for key in ['energy','force']:
                diff=a[kind+'_'+key][:,left]-a[kind+'_'+key][:,right]
                value[key]=dict(common_graph=masked_intervals(diff,common),by_seed=[masked_intervals(diff[s:s+1],common[s:s+1],45301+s) for s in [0,1]])
            comparisons[kind][label]=value
    (evidence/'source_physical_factorial_energy_v1.json').write_text(json.dumps(dict(source_audit_sha256=sha(audit),comparisons=comparisons),indent=2)+'\n')
    idx=np.random.default_rng(45311).integers(0,24,size=(20000,24))
    cells=(a['graph'][:,3].astype(float)-a['graph'][:,2].astype(float)).mean(-1)
    summary=dict(primary_audit_sha256=sha(audit),graph_source_with_physics=dict(mean=float(cells.mean()),by_seed=cells.mean(1).tolist(),
        composition_ci95=np.quantile(cells.mean(0)[idx].mean(1),[.025,.975]).tolist()),
        graph_supported_gfn2_failures=int((a['graph']&~a['xtb_success']).sum()),scope='Derived comparisons of the already replayed arrays; no new outcomes or parameter choices.')
    (evidence/'source_physical_factorial_summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
