#!/usr/bin/env python3
"""Replay every stored physical readout and expose within-composition energy tradeoffs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['protocol','audit','run','fixed','learned','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());rp=a.run/'results.json';report=json.loads(rp.read_text())
    assert report['complete'] and report['protocol_sha256']==sha(a.protocol)
    structural=json.loads(a.audit.read_text())
    assert spec['frozen'] and structural['complete']
    assert sha(a.audit)==spec['structural_audit_sha256']==report['structural_audit_sha256']
    assert structural['protocol_sha256']==spec['generator_protocol_sha256']
    assert report['oracle_sha256']==spec['oracle_sha256']
    assert report['new_molecular_oracle_calls']==report['requested_raw_queries']==spec['raw_oracle_calls']
    if a.out.exists():raise FileExistsError(a.out)
    checked=0;rows=[];arrays={};seen=set()
    for row in report['rows']:
        method,index=row['method'],row['condition_index'];group=a.fixed if method in ['warm','gaussian','fixed'] else a.learned
        assert (method,index) not in seen;seen.add((method,index))
        result_path=group/f'evaluation/{method}_results.json'
        assert sha(result_path)==structural['sources'][method]['results_sha256']
        generated=json.loads(result_path.read_text())
        source_row=next(r for r in generated['rows'] if r['condition_index']==index)
        source=group/f'evaluation/{method}_c{index}.pt';assert sha(source)==row['source_sample_sha256']
        original=torch.load(source,map_location='cpu',weights_only=False);parts=[];eplus=[];potentials=[]
        assert row['condition']==source_row['condition']==original['condition']
        assert len(original['positions'])==spec['samples_per_condition']
        assert sha(source)==source_row['sample_sha256']
        assert row['geometric_support']==[r['geometrically_supported'] for r in source_row['records']]
        assert row['graph_support']==[r['graph_supported'] for r in source_row['records']]
        for chunk in row['chunks']:
            path=a.run/chunk['path'];assert sha(path)==chunk['sha256'];d=torch.load(path,map_location='cpu',weights_only=False)
            x=d['positions'];e=np.asarray(d['raw_energy_eV']);inverse=np.asarray(d['inverted_energy_eV'])
            assert np.isfinite(e).all() and np.isfinite(inverse).all()
            for name in ['raw_force_eV_A','inverted_force_eV_A']:
                assert d[name].shape==x.shape and torch.isfinite(d[name]).all()
            even=.5*(e+inverse);u=even+.5*spec['restraint_eV_A2']*np.asarray(x.square().sum((1,2)))
            np.testing.assert_allclose(even,np.asarray(d['even_energy_eV']),atol=1e-9,rtol=0)
            np.testing.assert_allclose(u,np.asarray(d['unrestricted_potential_eV']),atol=1e-9,rtol=0)
            parts.append(x);eplus.extend(even);potentials.extend(u);checked+=2*len(x)
        torch.testing.assert_close(torch.cat(parts),original['positions'],atol=0,rtol=0)
        eplus=np.array(eplus);u=np.array(potentials);geom=np.array(row['geometric_support']);graph=np.array(row['graph_support'])
        np.testing.assert_allclose(eplus,row['even_energy_eV'],atol=1e-9,rtol=0)
        np.testing.assert_allclose(u,row['unrestricted_potential_eV'],atol=1e-9,rtol=0)
        assert abs(eplus.mean()-row['mean_even_energy_eV'])<2e-8
        assert abs(u.mean()-row['mean_unrestricted_potential_eV'])<2e-8
        assert int(geom.sum())==row['geometric_support_count'] and int(graph.sum())==row['graph_support_count']
        if geom.any():assert abs(u[geom].mean()-row['mean_potential_on_geometric_support_eV'])<2e-8
        if graph.any():assert abs(u[graph].mean()-row['mean_potential_on_graph_support_eV'])<2e-8
        arrays[method,index]=(eplus,u,geom,graph)
        rows.append(dict(method=method,condition_index=index,mean_energy_eV=float(eplus.mean()),median_energy_eV=float(np.median(eplus)),
            mean_potential_eV=float(u.mean()),geometric_support_count=int(geom.sum()),graph_support_count=int(graph.sum()),
            potential_quantiles_on_geometric_support=np.quantile(u[geom],[.1,.5,.9]).tolist() if geom.any() else None))
    assert checked==spec['raw_oracle_calls'] and len(rows)==len(spec['conditions'])*len(spec['methods'])
    assert seen=={(m,i) for m in spec['methods'] for i in spec['conditions']}
    comparisons={}
    for left,right in [('fixed','gaussian'),('node','gaussian'),('pair','gaussian'),('node','fixed'),('pair','fixed')]:
        de=[];du=[];common=[]
        for index in spec['conditions']:
            le,lu,lg,lgraph=arrays[left,index];re,ru,rg,rgraph=arrays[right,index]
            de.append(le-re);du.append(lu-ru);both=lg&rg
            common.append(dict(condition=index,n_common_geometric=int(both.sum()),
                mean_potential_difference_common_geometric_eV=float((lu-ru)[both].mean()) if both.any() else None,
                mean_energy_difference_all_eV=float((le-re).mean())))
        de=np.stack(de);du=np.stack(du);g=np.random.default_rng(88701);ids=g.integers(de.shape[1],size=(10000,*de.shape))
        comparisons[left+' minus '+right]=dict(mean_energy_difference_eV=float(de.mean()),
            energy_conditional_paired_bootstrap95=np.quantile(np.take_along_axis(de[None],ids,axis=2).mean((1,2)),[.025,.975]).tolist(),
            mean_potential_difference_eV=float(du.mean()),potential_conditional_paired_bootstrap95=np.quantile(np.take_along_axis(du[None],ids,axis=2).mean((1,2)),[.025,.975]).tolist(),per_condition=common)
    write(a.out,dict(complete=True,source_results_sha256=sha(rp),protocol_sha256=sha(a.protocol),structural_audit_sha256=sha(a.audit),oracle_sha256=report['oracle_sha256'],raw_energy_force_rows_checked=checked,
        rows=rows,comparisons=comparisons,new_oracle_calls_for_audit=0,producer_raw_oracle_calls=report['new_molecular_oracle_calls'],
        scientific_submission_ready=False,scope='All stored E_plus/restraint calculations, source positions, masks and readouts replay. No independent oracle re-query or equilibrium reference.',
        limits=['Energy differences are paired within composition; overall means weight the fixed8 cases equally.','Common-geometric-support subsets depend on both methods; they are descriptive, not unbiased equilibrium comparisons.','Energy measured only for the first connected-data training seed.','The energy model and graph perception are not quantum-validity certificates; lower energy does not demonstrate Boltzmann-distributed outputs.']))
    print(json.dumps(comparisons),flush=True)


if __name__=='__main__':main()
