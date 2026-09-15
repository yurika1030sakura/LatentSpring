#!/usr/bin/env python3
"""Separate audits of local teachers, paired FM students, and independent GFN2."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from cfm_mol.curvature_escort import centered_basis
from cfm_mol.escorted_thermal_teacher import graph_key
from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.audit_expanded_generators import check_rows,totals
from scripts.research.audit_source_utility import flags,intervals
from scripts.research.audit_source_sc_energy import masked_intervals
from scripts.research.evaluate_generator_quality import write
from scripts.research.train_electronic_fm import sha


def read(path):return json.loads(path.read_text())


def teachers(args):
    records=[];artifacts=[];queries=0;error=0.
    for seed in [0,1]:
        protocol=args.project/f'research/evidence/curvature_full_teacher_s{seed}_v1.json';spec=read(protocol)
        folder=args.run/f'teacher/s{seed}';resultfile=folder/'results.json';result=read(resultfile)
        assert result['complete'] and result['protocol_sha256']==sha(protocol)
        assert result['new_raw_queries']==spec['maximum_new_esen_queries']
        lookup={(r['method'],r['condition_index'],r['anchor_index']):r for r in result['rows']}
        checked=0
        for ref in spec['teacher_files']:
            path=args.project/ref['path'];assert sha(path)==ref['sha256']
            old=torch.load(path,weights_only=False,map_location='cpu');c=old['condition'];basis=centered_basis(c['n_atoms']);d=basis.shape[1]
            assert ref['anchor_indices']==torch.where(old['record']['eligible'])[0].tolist()
            for j in ref['anchor_indices']:
                anchor=old['record']['anchor'][j];sigma=old['record']['sigma'][j]
                key=graph_key(infer_chemical_graph(anchor,c['numbers'],c['charge']))
                rng=torch.Generator().manual_seed(spec['production_seed']*1000003+ref['condition_index']*100003+j)
                noise=torch.randn((24,c['n_atoms'],3),dtype=torch.float64,generator=rng);noise-=noise.mean(1,keepdim=True)
                for method in spec['methods']:
                    row=lookup[method,ref['condition_index'],j];file=folder/row['artifact'];assert sha(file)==row['artifact_sha256']
                    saved=torch.load(file,weights_only=False,map_location='cpu')
                    assert saved['protocol_sha256']==sha(protocol) and saved['source_teacher_sha256']==ref['sha256']
                    n=spec['production_particles'][method];y=saved['proposal']
                    torch.testing.assert_close(saved['source'],anchor[None]+sigma*noise[:n],atol=0,rtol=0)
                    assert torch.equal(anchor,saved['anchor']) and torch.equal(sigma,saved['sigma'])
                    sx=(saved['source']-anchor).reshape(n,-1)@basis;py=(y-anchor).reshape(n,-1)@basis
                    torch.testing.assert_close(py,sx@saved['matrix'].T+saved['shift'],atol=1e-12,rtol=1e-12)
                    proposal=torch.distributions.MultivariateNormal(saved['shift'],covariance_matrix=sigma**2*saved['matrix']@saved['matrix'].T)
                    reference=torch.distributions.MultivariateNormal(torch.zeros(d,dtype=basis.dtype),covariance_matrix=sigma**2*torch.eye(d,dtype=basis.dtype))
                    e=(saved['raw_energy_eV'][:n]+saved['raw_energy_eV'][n:])/2
                    assert torch.equal(e,saved['even_energy_eV'])
                    logw=reference.log_prob(py)-(e-saved['anchor_energy_eV'])/spec['kT']-proposal.log_prob(py)
                    valid=[]
                    for x in y:
                        try:valid.append(graph_key(infer_chemical_graph(x,c['numbers'],c['charge']))==key)
                        except (ValueError,RuntimeError,IndexError):valid.append(False)
                    valid=torch.tensor(valid);assert torch.equal(valid,saved['valid']) and valid.any()
                    error=max(error,float((saved['logw'][valid]-logw[valid]).abs().max()))
                    torch.testing.assert_close(saved['logw'][valid],logw[valid],atol=1e-10,rtol=1e-10)
                    weights=torch.where(valid,logw,torch.full_like(logw,-torch.inf)).softmax(0)
                    torch.testing.assert_close(weights,saved['weights'],atol=1e-10,rtol=1e-10)
                    ess=float(1/weights.square().sum());assert abs(ess-row['ess'])<1e-8
                    assert row['production_raw_queries']+row['pilot_raw_queries']<=48
                    records.append(dict(seed=seed,condition=ref['condition_index'],anchor=j,method=method,ess=ess,
                        same_graph=int(valid.sum()),particles=n));checked+=1
        assert checked==len(result['rows']);queries+=result['new_raw_queries']
        artifacts.append(dict(seed=seed,protocol_sha256=sha(protocol),result_sha256=sha(resultfile)))
    summary={s:{m:float(np.mean([r['ess'] for r in records if r['seed']==s and r['method']==m])) for m in ['translation','secant']} for s in [0,1]}
    write(args.out,dict(complete=True,rows=records,summary=summary,artifacts=artifacts,new_raw_queries=queries,
        maximum_independent_density_error=error,all_original_anchors_retained=True,neural_improvement_established=False))
    print(json.dumps(dict(teacher_summary=summary,raw_queries=queries)))


def compare(arrays,methods):
    graph=arrays['graph'];energy=arrays['energy_per_atom'];force=arrays['force_rms'];success=arrays.get('success',np.ones_like(graph,bool))
    result={};rng=np.random.default_rng(41791)
    for left,right in [('curvature_work16','work24'),('curvature_work16','force24'),('work24','force24'),
        ('curvature_work16','harmonic_tree'),('curvature_work16','escort_delta'),('force24','harmonic_tree')]:
        li,ri=methods.index(left),methods.index(right);mask=success[:,li]&success[:,ri]&graph[:,li]&graph[:,ri]
        both=success[:,li]&success[:,ri];row={}
        for name,value in [('energy_per_atom_eV',energy),('force_rms_eV_A',force)]:
            delta=value[:,li]-value[:,ri]
            row[name]=dict(jointly_valid=masked_intervals(delta,mask),all_successful=masked_intervals(delta,both),
                by_seed=[masked_intervals(delta[s:s+1],mask[s:s+1]) for s in [0,1]])
        row['graph_validity']=intervals(graph[:,li].astype(float)-graph[:,ri].astype(float),[i//8 for i in range(24)],rng)
        result[left+' minus '+right]=row
    return result


def generation(args):
    from flowmol.model_utils.load import model_from_config,read_config_file
    from cfm_mol.radial_reference import prepare_research_backbone
    teacher_audit=args.project/'research/evidence/curvature_full_teacher_audit_v1.json';ta=read(teacher_audit)
    assert ta['complete'] and ta['all_original_anchors_retained']
    oldpath=args.project/'research/evidence/fresh_physics_esen_audit_v1.json';old=read(oldpath)
    oldarrays=oldpath.with_suffix('.npz');assert sha(oldarrays)==old['arrays_sha256'];prior=np.load(oldarrays)
    methods=['harmonic_tree','escort_delta','force24','work24','curvature_work16'];shape=(2,5,24,32)
    arrays={k:np.zeros(shape,dtype=bool if k=='graph' else float) for k in ['graph','energy_per_atom','force_rms']}
    for k in arrays:arrays[k][:,:2]=prior[k][:,[1,2]]
    summary={};artifacts=[];errors=[]
    for seed in [0,1]:
        protocol=args.project/f'research/evidence/curvature_distillation_s{seed}_v1.json';spec=read(protocol);ph=sha(protocol)
        folder=args.run/f'student/s{seed}';done=read(folder/'complete.json')
        assert done['complete'] and done['protocol_sha256']==ph and done['evaluation_raw_queries']==4608
        for key in ['warm_checkpoint','replay_checkpoint','selection_log','teacher_protocol','data','config','condition_manifest']:
            assert sha(args.project/spec[key])==spec[key+'_sha256']
        warm=torch.load(args.project/spec['warm_checkpoint'],weights_only=False,map_location='cpu')
        replay=torch.load(args.project/spec['replay_checkpoint'],weights_only=False,map_location='cpu')
        cfg=read_config_file(args.project/spec['config']);cfg['mol_fm'].pop('bgfm',None)
        model=model_from_config(cfg);prepare_research_backbone(model,warm['research_protocol']);names={n for n,_ in model.named_parameters()};del model
        source=prior_from_checkpoint(warm);panel=read(args.project/spec['condition_manifest'])['rows']
        original=[json.loads(line) for line in (args.project/spec['selection_log']).read_text().splitlines()]
        ts=read(args.project/spec['teacher_protocol']);tr=read(args.project/spec['teacher_run']/'results.json')
        assert any(a['seed']==seed and a['result_sha256']==sha(args.project/spec['teacher_run']/'results.json') for a in ta['artifacts'])
        lookup={(r['method'],r['condition_index'],r['anchor_index']):r for r in tr['rows']};cache={}
        quality=read(folder/'physical_eval/results.json');assert quality['complete'] and quality['raw_queries']==4608 and quality['protocol_sha256']==ph
        qr={(r['method'],r['condition_index']):r for r in quality['rows']};summary[seed]={}
        for method in spec['methods']:
            mi=methods.index(method);directory=folder/method;training=read(directory/'training.json')
            metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            assert len(metrics)==len(original)==training['steps']==1000
            for new,oldrow in zip(metrics,original):
                assert new['step']==oldrow['step'] and new['composition']==oldrow['composition']
                label,oldlabel=new['selection'],oldrow['selection']
                assert {k:v for k,v in label.items() if k!='particle'}=={k:v for k,v in oldlabel.items() if k!='particle'}
                assert np.isfinite(new['loss']) and np.isfinite(new['gradient_norm'])
                if label['kind']=='reference':assert label==oldlabel
                else:
                    ref=ts['teacher_files'][label['pool']];j=ref['anchor_indices'][label['row']]
                    m='secant' if method=='curvature_work16' else 'translation';key=(m,ref['condition_index'],j)
                    if key not in cache:
                        row=lookup[key];file=args.project/spec['teacher_run']/row['artifact'];assert sha(file)==row['artifact_sha256']
                        cache[key]=torch.load(file,weights_only=False,map_location='cpu')
                    saved=cache[key];weights=saved['valid'].double()/saved['valid'].sum() if method=='force24' else saved['weights']
                    selected=int(torch.searchsorted(weights.cumsum(0),label['uniform_draw'],right=True).clamp_max(len(weights)-1))
                    assert selected==label['particle'] and weights[selected]>0
            student=torch.load(directory/'student.ckpt',weights_only=False,map_location='cpu')
            final=torch.load(directory/'last.ckpt',weights_only=False,map_location='cpu')
            assert sha(directory/'student.ckpt')==training['student_sha256'] and sha(directory/'last.ckpt')==training['checkpoint_sha256']
            assert student['protocol_sha256']==ph
            for name,value in warm['state_dict'].items():
                expected=value+(student['state_dict'][name]-replay['state_dict'][name]) if name in names else value
                torch.testing.assert_close(final['state_dict'][name],expected,atol=0,rtol=0)
            del student,final
            reportfile=folder/'evaluation'/f'{method}_results.json';report=read(reportfile)
            assert report['protocol_sha256']==ph and report['checkpoint_sha256']==training['checkpoint_sha256']
            rows=check_rows(reportfile.parent,method,report,panel,32,source,spec['evaluation_seed'],source_atol=1e-12,source_errors=errors)
            summary[seed][method]=totals(rows)
            for i,row in enumerate(rows):
                sourcefile=reportfile.parent/f'{method}_c{i}.pt';sample=torch.load(sourcefile,weights_only=False,map_location='cpu')
                file=folder/'physical_eval'/f'{method}_c{i}.pt';assert sha(file)==qr[method,i]['artifact_sha256']
                saved=torch.load(file,weights_only=False,map_location='cpu');assert saved['source_sample_sha256']==sha(sourcefile)
                assert torch.equal(saved['positions'],sample['positions'])
                e,f=saved['raw_energy_eV'],saved['raw_force_eV_A'];assert e.shape==(64,) and f.shape==(64,panel[i]['n_atoms'],3)
                even,force=(e[:32]+e[32:])/2,(f[:32]-f[32:])/2
                assert torch.isfinite(e).all() and torch.isfinite(f).all()
                assert torch.equal(even,saved['even_energy_eV']) and torch.equal(force,saved['even_force_eV_A'])
                arrays['graph'][seed,mi,i]=flags(row,'graph_supported').astype(bool)
                arrays['energy_per_atom'][seed,mi,i]=(even/panel[i]['n_atoms']).numpy()
                arrays['force_rms'][seed,mi,i]=force.square().sum(-1).mean(-1).sqrt().numpy()
            artifacts.append(dict(seed=seed,method=method,report_sha256=sha(reportfile),checkpoint_sha256=training['checkpoint_sha256'],protocol_sha256=ph))
    arrayfile=args.out.with_suffix('.npz');np.savez_compressed(arrayfile,**arrays)
    comparisons=compare(arrays,methods)
    write(args.out,dict(complete=True,methods=methods,summary=summary,comparisons=comparisons,artifacts=artifacts,
        arrays_sha256=sha(arrayfile),source_replay_max_error_A=max(errors),teacher_audit_sha256=sha(teacher_audit),
        reused_baseline_audit_sha256=sha(oldpath),new_neural_outputs=4608,new_optimizer_steps=6000,new_raw_esen_queries=9216,
        exact_selection_and_parameter_replay=True,scope='Paired raw neural outputs on a reused unfitted24-composition panel. Conditional energy comparisons retain joint validity and all-successful estimands separately.'))
    print(json.dumps(dict(summary=summary,primary=comparisons['curvature_work16 minus work24']),indent=2))


def xtb(args):
    from ase.data import chemical_symbols
    parentfile=args.project/'research/evidence/curvature_distillation_esen_audit_v1.json';parent=read(parentfile)
    assert parent['complete'] and sha(parentfile.with_suffix('.npz'))==parent['arrays_sha256']
    methods=parent['methods'];graph=np.load(parentfile.with_suffix('.npz'))['graph']
    arrays=dict(graph=graph,energy_per_atom=np.full(graph.shape,np.nan),force_rms=np.full(graph.shape,np.nan),success=np.zeros(graph.shape,bool))
    oldfile=args.project/'research/evidence/fresh_physics_xtb_audit_v1.json';old=read(oldfile)
    assert sha(oldfile.with_suffix('.npz'))==old['arrays_sha256'];prior=np.load(oldfile.with_suffix('.npz'))
    for key in ['energy_per_atom','force_rms','success']:arrays[key][:,:2]=prior[key][:,[1,2]]
    checked=0;inversions=[];artifacts=[]
    for seed in [0,1]:
        protocol=args.project/f'research/evidence/curvature_distillation_s{seed}_v1.json';spec=read(protocol);ph=sha(protocol)
        folder=args.run/f'xtb/s{seed}';file=folder/'results.json';result=read(file)
        assert result['complete'] and result['protocol_sha256']==ph and result['attempted']==2352
        tasksfile=folder/'tasks.json';assert sha(tasksfile)==result['tasks_sha256'];tasks=read(tasksfile)
        lookup={r['task']['task_id']:r for r in tasks['tasks']};assert len(lookup)==2352
        cache={};by_id={}
        panel=read(args.project/spec['condition_manifest']);pool=read(args.project/panel['source_pool'])['rows']
        for row in result['rows']:
            item=lookup[row['task_id']];task=item['task'];c=item['condition'];i,m,j=task['condition_index'],task['method'],task['sample_index']
            assert c['composition_hex']==panel['rows'][i]['composition_hex']
            if m=='reference':expected=np.asarray(pool[c['pool_index']]['reference_positions']);expected=-expected if task['inversion_check'] else expected
            else:
                if (m,i) not in cache:
                    path=args.run/f'student/s{seed}/evaluation/{m}_c{i}.pt'
                    ref=next(r for r in tasks['sources'] if r['method']==m and r['condition_index']==i)
                    assert sha(path)==ref['sample_sha256'];cache[m,i]=torch.load(path,weights_only=False,map_location='cpu')['positions'].numpy()
                expected=cache[m,i][j]
            np.testing.assert_array_equal(task['positions'],expected)
            directory=folder/'details'/row['task_id']
            for name,key in [('input.xyz','input_xyz_sha256'),('stdout.txt','stdout_sha256'),('stderr.txt','stderr_sha256')]:assert sha(directory/name)==row[key]
            lines=(directory/'input.xyz').read_text().splitlines();assert int(lines[0])==c['n_atoms'] and len(lines)==c['n_atoms']+2
            assert [s.split()[0] for s in lines[2:]]==[chemical_symbols[z] for z in c['numbers']]
            np.testing.assert_allclose([[float(v) for v in s.split()[1:]] for s in lines[2:]],expected,atol=1e-12,rtol=0)
            if row['returncode'] is None:assert not row['success'] and row['failure']=='single_point_timeout'
            else:
                gradient=(directory/'gradient').read_text() if (directory/'gradient').exists() else ''
                if row['gradient_sha256'] is not None:assert sha(directory/'gradient')==row['gradient_sha256']
                parsed=parse_singlepoint((directory/'stdout.txt').read_text(),(directory/'stderr.txt').read_text(),row['returncode'],gradient,c['n_atoms'])
                assert parsed['success']==row['success']
                if row['success']:
                    assert parsed['energy_eV']==row['energy_eV'];np.testing.assert_array_equal(parsed['force_eV_A'],row['force_eV_A'])
                else:assert parsed['failure']==row['failure']
            assert '--opt' not in row['command'] and row['command'][-1]=='--grad' and row['original_charge']==0 and row['original_spin_multiplicity']==1
            checked+=1;by_id[row['task_id']]=row
            if m!='reference':
                mi=methods.index(m);arrays['success'][seed,mi,i,j]=row['success']
                if row['success']:
                    arrays['energy_per_atom'][seed,mi,i,j]=row['energy_eV']/c['n_atoms']
                    f=np.asarray(row['force_eV_A']);arrays['force_rms'][seed,mi,i,j]=np.sqrt(np.mean(np.sum(f*f,axis=-1)))
        for i in range(24):
            plus,minus=[by_id[f'reference_c{i}_s0_{s}'] for s in ['plus','minus']]
            passed=plus['success'] and minus['success']
            if passed:passed=abs(plus['energy_eV']-minus['energy_eV'])<=spec['xtb_energy_inversion_tolerance_eV'] and np.max(np.abs(np.asarray(plus['force_eV_A'])+np.asarray(minus['force_eV_A'])))<=spec['xtb_force_inversion_tolerance_eV_A']
            inversions.append(dict(seed=seed,condition=i,passed=bool(passed)))
        artifacts.append(dict(seed=seed,results_sha256=sha(file),tasks_sha256=sha(tasksfile),protocol_sha256=ph))
    assert checked==4704;comparisons=compare(arrays,methods)
    def passed(comp):
        energy=comp['energy_per_atom_eV'];pooled=energy['jointly_valid']
        return bool(pooled['mean'] is not None and pooled['paired_draw95'][1]<0 and all(x['mean'] is not None and x['mean']<0 for x in energy['by_seed']))
    key='curvature_work16 minus work24'
    validity=float((graph[:,4].astype(float)-graph[:,3].astype(float)).mean())
    gate=passed(comparisons[key]) and passed(parent['comparisons'][key]) and validity>=-.02 and all(r['passed'] for r in inversions)
    arrayfile=args.out.with_suffix('.npz');np.savez_compressed(arrayfile,**arrays)
    write(args.out,dict(complete=True,methods=methods,comparisons=comparisons,artifacts=artifacts,arrays_sha256=sha(arrayfile),
        raw_xtb_attempts_reparsed=checked,reference_inversions=inversions,reference_inversion_gate=all(r['passed'] for r in inversions),
        esen_parent_audit_sha256=sha(parentfile),reused_baseline_audit_sha256=sha(oldfile),cross_potential_primary_gate=gate,
        physical_success_by_method={m:int(arrays['success'][:,i].sum()) for i,m in enumerate(methods)},
        scope='All raw neural outputs independently scored by GFN2 without geometry optimization. Jointly-valid and all-successful contrasts reported separately; no global-equilibrium claim.'))
    print(json.dumps(dict(primary=comparisons[key],cross_potential_primary_gate=gate),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','run','out']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--stage',choices=['teachers','generation','xtb'],required=True)
    args=parser.parse_args();torch.set_num_threads(1);assert not args.out.exists()
    dict(teachers=teachers,generation=generation,xtb=xtb)[args.stage](args)


if __name__=='__main__':main()
