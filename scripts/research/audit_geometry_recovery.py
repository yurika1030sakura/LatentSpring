"""Recompute all-output geometry, force and diversity contrasts for the pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def contrast(candidate, control, seed, repeats):
    difference = (candidate.astype(float) - control.astype(float)).mean(-1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(difference.shape[1],size=(repeats,difference.shape[1]))
    composition = difference[:,indices].mean((0,2))
    fits = rng.integers(len(difference),size=(repeats,len(difference)))
    crossed = difference[fits[:,:,None],indices[:,None,:]].mean((1,2))
    return dict(mean=float(difference.mean()),by_fit=difference.mean(1).tolist(),
        composition_ci95=np.quantile(composition,[.025,.975]).tolist(),
        crossed_fit_composition_ci95=np.quantile(crossed,[.025,.975]).tolist(),
        composition_ci97_5_two_candidates=np.quantile(composition,[.0125,.9875]).tolist())


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','run']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    methods=['frozen']+protocol['variants'];fits=protocol['fits'];count=protocol['samples_per_condition']
    shape=(len(fits),len(methods),len(protocol['conditions']),count)
    arrays={key:np.zeros(shape,dtype=bool) for key in ['graph','geometry','geometry_force','graph_force','heavy_connected','unique_graph','unique_geometry']}
    arrays['force']=np.full(shape,np.nan)
    missing=[];provenance={};summaries={}
    for fi,fit in enumerate(fits):
        initializations=[]
        for mi,method in enumerate(methods):
            folder=a.run/f's{fit}'/method
            evaluation=folder if method=='frozen' else folder/'evaluation'
            complete=evaluation/'complete.json'
            if not complete.exists():missing.append(str(complete));continue
            done=json.loads(complete.read_text());assert done['complete'] and done['protocol_sha256']==ph
            assert done['attempted']==len(protocol['conditions'])*count
            for name,key in [('geometry.json','geometry_sha256'),('xtb/results.json','physical_sha256'),('generation/generation.json','generation_sha256')]:
                assert sha(evaluation/name)==done[key]
            generation=json.loads((evaluation/'generation/generation.json').read_text())
            for row in generation['rows']:assert sha(evaluation/'generation'/row['file'])==row['sha256']
            geometry=json.loads((evaluation/'geometry.json').read_text())['rows']
            scores=json.loads((evaluation/'xtb/results.json').read_text())['rows']
            physical={(r['condition_index'],r['sample_index']):r for r in scores}
            assert len(physical)==len(geometry)==done['attempted']
            seen_graph={i:set() for i in range(shape[2])};seen_geometry={i:set() for i in range(shape[2])}
            for row in sorted(geometry,key=lambda r:(r['condition'],r['sample'])):
                ci,j=row['condition'],row['sample'];score=physical[ci,j];slot=(fi,mi,ci,j)
                assert score['graph']==row['graph']
                arrays['graph'][slot]=row['graph'];arrays['geometry'][slot]=row['closed_shell_geometry_pass']
                arrays['heavy_connected'][slot]=row['heavy_components']==1
                force=score['rms_force'] if score['success'] and score['rms_force'] is not None else np.nan
                arrays['force'][slot]=force
                low_force=bool(np.isfinite(force) and force<=5)
                arrays['geometry_force'][slot]=row['closed_shell_geometry_pass'] and low_force
                arrays['graph_force'][slot]=row['graph'] and low_force
                smiles=row.get('smiles')
                if row['graph'] and smiles:
                    arrays['unique_graph'][slot]=smiles not in seen_graph[ci];seen_graph[ci].add(smiles)
                if row['closed_shell_geometry_pass'] and smiles:
                    arrays['unique_geometry'][slot]=smiles not in seen_geometry[ci];seen_geometry[ci].add(smiles)
            provenance[str(complete)]=sha(complete)
            if method!='frozen':
                initializations.append(json.loads((folder/'initialization.json').read_text()))
                training=json.loads((folder/'training.json').read_text());assert training['complete'] and training['protocol_sha256']==ph
                assert training['steps']==protocol['steps'] and sha(folder/'last.ckpt')==training['checkpoint_sha256']
        if len(initializations)==len(protocol['variants']):
            assert len({r['initial_state_sha256'] for r in initializations})==1
            assert len({r['batch_schedule_sha256'] for r in initializations})==1
    if missing:
        write(a.run/'audit_pending.json',dict(complete=False,protocol_sha256=ph,missing=missing))
        print(json.dumps(dict(complete=False,missing=missing)),flush=True);return
    metrics=[k for k in arrays if k!='force']
    for mi,method in enumerate(methods):
        summaries[method]={metric:dict(count=int(arrays[metric][:,mi].sum()),
            rate=float(arrays[metric][:,mi].mean()),by_fit=arrays[metric][:,mi].mean((1,2)).tolist()) for metric in metrics}
        summaries[method]['attempted']=len(fits)*shape[2]*count
    stats=protocol['statistics'];contrasts={};advance={}
    for method in ['recovery','recovery_local']:
        mi=methods.index(method);contrasts[method]={}
        for control in ['frozen','replay']:
            ci=methods.index(control)
            contrasts[method][control]={metric:contrast(arrays[metric][:,mi],arrays[metric][:,ci],
                stats['bootstrap_seed'],stats['bootstrap_repetitions']) for metric in metrics}
        advance[method]=all(
            min(contrasts[method][control]['geometry']['by_fit'])>0 and
            min(contrasts[method][control]['geometry_force']['by_fit'])>0 and
            contrasts[method][control]['geometry']['composition_ci97_5_two_candidates'][0]>0 and
            contrasts[method][control]['unique_graph']['mean']>=-.02
            for control in ['frozen','replay'])
    eligible=[method for method,value in advance.items() if value]
    selected=max(eligible,key=lambda method:summaries[method]['geometry_force']['rate']) if eligible else None
    np.savez_compressed(a.run/'audit.npz',**arrays)
    result=dict(complete=True,protocol_sha256=ph,methods=methods,summary=summaries,contrasts=contrasts,
        development_advance=advance,selected=selected,arrays_sha256=sha(a.run/'audit.npz'),
        provenance=provenance,new_optimizer_steps=protocol['budget']['new_backbone_updates'],
        new_backbone_training_example_forwards=protocol['budget']['backbone_training_example_forwards'],
        new_generation_outputs=int(np.prod(shape)),new_gfn2_attempts=int(np.prod(shape)),new_esen_queries=0,
        all_attempts_retained=True,geometry_optimized=False,scope=protocol['scope'])
    write(a.run/'audit.json',result)
    print(json.dumps(dict(complete=True,summary=summaries,advance=advance,selected=selected),indent=2),flush=True)


if __name__=='__main__':main()
