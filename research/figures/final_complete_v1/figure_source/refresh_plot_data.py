"""Rebuild Fig2 plot values from the supplied per-attempt archives."""
from pathlib import Path
import csv,gzip,json,numpy as np
o=Path(__file__).resolve().parent
# Derive only plotting values from the archived per-attempt CSV and arrays.
final=dict(np.load(o/'data/geometry_candidate_five_fit_audit_v1.npz'));other=dict(np.load(o/'data/other_baseline_transfer_audit_v1.npz'));additional=dict(np.load(o/'data/baseline_chemical_geometry_audit_v1.npz'))
shape=(5,64,16);gg=np.zeros(shape,bool);gs=gg.copy();gf=np.full(shape,np.nan);geom=gg.copy()
for row in csv.DictReader(gzip.open(o/'data/seed_replication_audit_v2_scores.csv.gz','rt')):
 if row['method']=='gaga_parent':
  key=tuple(int(row[x]) for x in ['fit','condition_index','sample_index']);gg[key]=bool(int(row['graph_supported']));gs[key]=bool(int(row['physical_success']));gf[key]=float(row['rms_force_eV_A'])
for line in gzip.open(o/'data/round2_chemical_geometry_records_v1.jsonl.gz','rt'):
 row=json.loads(line)
 if row['method']=='gaga_parent':geom[row['fit'],row['condition'],row['sample']]=row['closed_shell_geometry_pass']
assert np.isfinite(gf).all()
s=[]
for name,key in [('Gaussian FM','gaussian_fm'),('EDM','edm')]:s.append(dict(name=name,graph=other[key+'_graph'][:,0],success=other[key+'_success'][:,0],force=other[key+'_force'][:,0],geometry=additional[key+'_geometry'][:,0]))
s += [dict(name='GAGA',graph=gg,success=gs,force=gf,geometry=geom),dict(name='LatentSpring',graph=final['graph'][:,1],success=final['success'][:,1],force=final['force'][:,1],geometry=final['geometry'][:,1])]
plots=[]
for t in s:
 valid=t['geometry']&t['success'];f=t['force'];threshold=np.unique(np.r_[0,5,f[valid&(f<=10)],10]);curves=np.array([100*np.mean(valid&(f<=x)) for x in threshold]);plots.append(dict(method=t['name'],fits=len(f),attempts=f.size,graph=100*t['graph'].mean(),graph_force=100*(t['graph']&t['success']&(f<=5)).mean(),geometry_force=100*(valid&(f<=5)).mean(),threshold=threshold.tolist(),curve=curves.tolist(),graph_by_fit=(100*t['graph'].mean((1,2))).tolist(),graph_force_by_fit=(100*(t['graph']&t['success']&(f<=5)).mean((1,2))).tolist()))
(o/'data/figure2_plot_data.json').write_text(json.dumps(plots,indent=2))
print([{k:v for k,v in x.items() if k not in ['threshold','curve','graph_by_fit','graph_force_by_fit']} for x in plots])
