"""Secondary radical, duplication, energy and correction-gain readouts from audited files."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.research.confirm_gaga_feedback import bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project
    folder=root/'runs/published_model_confirmation_v1';d=json.loads((folder/'audit.json').read_text());assert d['complete'] and sha(folder/'audit.npz')==d['arrays_sha256'];methods=d['methods']
    with np.load(folder/'audit.npz') as f:g=f['graph'].copy();ok=f['success'].copy();force=f['force'].copy()
    radicals=np.full(g.shape,-1);distinct=np.zeros(g.shape[:3]);energy=np.full(g.shape,np.nan)
    for si in [0,1]:
        folders=[root/f'runs/context_confirmation_v1/s{si}',folder/f's{si}']
        for run in folders:
            done=json.loads((run/'complete.json').read_text());assert done['complete'] and sha(run/'audit.json')==done['raw_audit_sha256'] and sha(run/'xtb/audit.json')==done['xtb_audit_sha256']
            records=json.loads((run/'audit.json').read_text())['rows']
            for method in {r['arm'] for r in records}:
                mi=methods.index(method);selected=[r for r in records if r['arm']==method];assert len(selected)==16
                for i,row in enumerate(selected):
                    distinct[si,mi,i]=row['distinct_connectivity']/16
                    for j,record in enumerate(row['records']):
                        if record['graph_supported']:radicals[si,mi,i,j]=record['radical_electrons']
            physical=json.loads((run/'xtb/audit.json').read_text());tasks={v['task']['task_id']:v for v in json.loads((run/'xtb/tasks.json').read_text())['tasks']}
            for row in physical['rows']:
                item=tasks[row['task_id']];t=item['task'];mi=methods.index(t['method']);i,j=t['condition_index'],t['sample_index']
                if row['success']:energy[si,mi,i,j]=row['energy_eV']/len(t['positions'])
    joint=g&ok&(force<=5);rng=np.random.default_rng(57131);summary={}
    for mi,method in enumerate(methods):
        summary[method]=dict(zero_radical_graph=float((g[:,mi]&(radicals[:,mi]==0)).mean()),zero_radical_joint5=float((joint[:,mi]&(radicals[:,mi]==0)).mean()),
            distinct_connectivity_yield=float(distinct[:,mi].mean()),graph_rate=float(g[:,mi].mean()))
    contrasts={}
    for method in ['pair_short','pair_long','pair_wide','context']:
        mi=methods.index(method);bi=methods.index('base')
        contrasts[method+'_minus_base_joint5']=bootstrap(joint[:,mi].mean(-1)-joint[:,bi].mean(-1),rng,20000)
        assert np.isfinite(energy[:,mi]).all() and np.isfinite(energy[:,bi]).all()
        contrasts[method+'_minus_base_all_output_energy_eV_atom']=bootstrap((energy[:,mi]-energy[:,bi]).mean(-1),rng,20000)
    values=np.stack([(joint[:,methods.index(m)].mean(-1)-joint[:,methods.index('base')].mean(-1)).mean(0) for m in ['pair_long','pair_wide','context']])
    means=values.mean(-1);indices=rng.integers(0,16,size=(20000,16));draws=values[:,indices].mean(-1);radius=float(np.quantile(np.abs(draws-means[:,None]).max(0),.95))
    simultaneous={m:dict(mean=float(v),ci95=[float(v-radius),float(v+radius)]) for m,v in zip(['pair_long','pair_wide','context'],means)}
    write(a.out,dict(complete=True,source_audit_sha256=sha(folder/'audit.json'),summary=summary,contrasts=contrasts,simultaneous_long_head_vs_parent=simultaneous,
        scope='Secondary diagnostics on the same audited panel, with no new generation or oracle calls. Radical counts concern inferred graph labels; duplication is measured only at16 draws per composition/run. Energy contrasts include all attempted outputs and are not equilibrium-distribution or per-isomer comparisons.'))


if __name__=='__main__':main()
