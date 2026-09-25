"""Check the incoming manuscript bundle against frozen scientific evidence."""
import argparse,gzip,hashlib,json,re
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ['project','package','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();package=a.package.resolve()
    manifest=json.loads((package/'SHA256.json').read_text())
    for name,value in manifest.items():assert digest(package/name)==value,name
    templates=['iclr2027_conference.sty','iclr2027_conference.bst','fancyhdr.sty','natbib.sty']
    for name in templates:assert digest(package/'manuscript'/name)==digest(root/'paper'/name)
    bib='\n'.join((root/'paper'/name).read_text() for name in ['refs.bib','tree_refs.bib'])
    assert bib==(package/'manuscript/refs.bib').read_text()
    counts={env:0 for env in ['tabular','equation','align']}
    for section in sorted((package/'manuscript/sections').glob('*.tex')):
        old=(root/'paper/sections'/section.name).read_text();new=section.read_text()
        for env in counts:
            pattern=rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}'
            blocks=lambda t:[re.sub(r'\s+','',s) for s in re.findall(pattern,t,re.S)]
            assert blocks(old)==blocks(new),(section.name,env)
            counts[env]+=len(blocks(new))
    data=package/'figure_source/data';copied_arrays=['geometry_candidate_five_fit_audit_v1.npz','other_baseline_transfer_audit_v1.npz','baseline_chemical_geometry_audit_v1.npz']
    for name in copied_arrays:assert digest(data/name)==digest(root/'research/evidence'/name)
    final=dict(np.load(root/'research/evidence'/copied_arrays[0]));other=dict(np.load(root/'research/evidence'/copied_arrays[1]));additional=dict(np.load(root/'research/evidence'/copied_arrays[2]))
    seed=dict(np.load(root/'research/evidence/seed_replication_audit_v2.npz'));meta=json.loads((root/'research/evidence/seed_replication_audit_v2.json').read_text());index=meta['methods'].index('gaga_parent')
    geom=np.zeros((5,64,16),bool)
    for line in gzip.decompress((root/'research/evidence/round2_chemical_geometry_records_v1.jsonl.gz').read_bytes()).decode().splitlines():
        row=json.loads(line)
        if row['method']=='gaga_parent':geom[row['fit'],row['condition'],row['sample']]=row['closed_shell_geometry_pass']
    series=[]
    for name,key in [('Gaussian FM','gaussian_fm'),('EDM','edm')]:series.append(dict(name=name,graph=other[key+'_graph'][:,0],force=other[key+'_force'][:,0],success=other[key+'_success'][:,0],geometry=additional[key+'_geometry'][:,0]))
    series.extend([dict(name='GAGA',graph=seed['graph'][:,index],force=seed['force'][:,index],success=seed['success'][:,index],geometry=geom),dict(name='LatentSpring',graph=final['graph'][:,1],force=final['force'][:,1],success=final['success'][:,1],geometry=final['geometry'][:,1])])
    plots=json.loads((data/'figure2_plot_data.json').read_text());summaries=[]
    for s,row in zip(series,plots):
        assert s['name']==row['method'];assert row['attempts']==s['graph'].size
        valid=s['geometry']&s['success'];force=s['force']
        expected=dict(graph=100*s['graph'].mean(),graph_force=100*(s['graph']&s['success']&(force<=5)).mean(),geometry_force=100*(valid&(force<=5)).mean())
        for key,value in expected.items():assert row[key]==value,(s['name'],key)
        thresholds=np.unique(np.r_[0,5,force[valid&(force<=10)],10]);curve=np.array([100*np.mean(valid&(force<=x)) for x in thresholds])
        np.testing.assert_array_equal(thresholds,row['threshold']);np.testing.assert_array_equal(curve,row['curve'])
        summaries.append(dict(method=s['name'],attempts=row['attempts'],**expected))
    scene=root/'research/figures/geometric_method_v1/scenes.json';assert digest(data/'scenes.json')==digest(scene)
    out=dict(complete=True,package_sha256_manifest_verified=True,verified_files=len(manifest),conference_styles_unchanged=True,bibliography_unchanged=True,unchanged_appendix_blocks=counts,figure2_means_and_every_cdf_point_match=True,figure2_maximum_numeric_difference=0,summary=summaries,molecular_scene_unchanged=True,scene_sha256=digest(scene),new_training_or_physical_evaluations=0,scope='Independent package-content check against local archived raw arrays; manuscript edits and rendering fixes are recorded separately.')
    a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
