"""Reproduce released tables and key intervals using only NumPy and score CSVs."""
import argparse,csv,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
from cfm_mol.replication_statistics import paired_intervals

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--out',type=Path);a=p.parse_args();root=a.project.resolve();manifest=json.loads((root/'research/evidence/released_scores_v1.json').read_text());data={};seen=set()
    for name,digest in manifest['files'].items():
        f=root/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==digest
        with gzip.open(f,'rt') as stream:
            for row in csv.DictReader(stream):
                key=row['method'];index=(int(row['fit']),int(row['condition_index']),int(row['sample_index']));uid=(key,*index);assert uid not in seen;seen.add(uid)
                if key not in data:data[key]={v:np.full((5,64,16),np.nan) for v in ['graph','success','energy','force']}
                for metric,col in [('graph','graph_supported'),('success','physical_success'),('energy','energy_eV_per_atom'),('force','rms_force_eV_A')]:data[key][metric][index]=float(row[col])
    assert len(seen)==51200 and len(data)==10 and all(np.isfinite(x).all() for row in data.values() for x in row.values())
    for row in data.values():row['joint']=(row['graph']!=0)&(row['success']!=0)&(row['force']<=5)
    summary={name:dict(graph=float(v['graph'].mean()),joint=float(v['joint'].mean())) for name,v in data.items()}
    original=json.loads((root/'research/evidence/seed_replication_audit_v2.json').read_text());transfer=json.loads((root/'research/evidence/cross_generator_head_audit_v1.json').read_text())
    for name,v in summary.items():
        expected=transfer['summary'][name.split('_')[0]] if name.endswith('_cross') else original['summary'][name]
        np.testing.assert_allclose(v['graph'],expected.get('graph_rate',expected.get('graph')),atol=0,rtol=0);np.testing.assert_allclose(v['joint'],expected.get('joint_rate',expected.get('joint')),atol=0,rtol=0)
    strata=np.array([0 if c['n_atoms']<=28 else 1 for c in manifest['conditions']]);checks={}
    for name,left,right,metric,reference in [
        ('fm_physical_gain','fm_physical','fm_parent','joint',original['contrasts']['fm_physical_minus_parent']['new_three']['joint']),
        ('gaga_physical_gain','gaga_physical','gaga_parent','joint',original['contrasts']['gaga_physical_minus_parent']['new_three']['joint']),
        ('fm_head_on_gaga','gaga_cross','gaga_parent','joint',transfer['contrasts']['gaga_cross_minus_parent']['new_three']['joint']),
        ('gaga_head_on_fm','fm_cross','fm_parent','joint',transfer['contrasts']['fm_cross_minus_parent']['new_three']['joint']),
        ('learned_h_energy','fm_hydrogen','fm_radial','energy',original['contrasts']['fm_hydrogen_minus_radial']['new_three']['energy'])]:
        delta=(data[left][metric][2:].astype(float)-data[right][metric][2:].astype(float)).mean(-1);value=paired_intervals(delta,strata=strata)
        for key in ['mean','by_fit','composition_ci95','crossed_fit_composition_ci95','equal_stratum_mean','equal_stratum_crossed_ci95']:np.testing.assert_allclose(value[key],reference[key],atol=1e-10,rtol=1e-10)
        checks[name]=value
    assert 'torch' not in sys.modules
    result=dict(complete=True,scored_output_rows=len(seen),torch_required=False,summary=summary,intervals_reproduced=list(checks))
    if a.out:
        if a.out.exists():raise FileExistsError(a.out)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
