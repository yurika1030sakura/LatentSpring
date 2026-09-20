"""Fit frozen physical heads and score all final-study arms on one new panel."""
import argparse,copy,gc,json,subprocess,sys,time
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from scripts.research.run_matched_connection import load_parent,make_context,teacher,fit,generate,score
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=range(5),required=True);a=p.parse_args();root=a.project.resolve();a.out=a.out.resolve();si=a.fit
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    campaign=json.loads(a.protocol.read_text());assert campaign['frozen'];cp=sha(a.protocol);tick=time.perf_counter()
    panel_file=root/campaign['panel'];audit_file=root/campaign['panel_audit'];panel=json.loads(panel_file.read_text());audit=json.loads(audit_file.read_text())
    assert panel['complete'] and audit['complete'] and panel['qualification_audit_sha256']==sha(audit_file) and len(panel['rows'])==64
    if audit['protocol_sha256']!=campaign['pool_protocol_sha256']:
        resolution=json.loads((root/'research/evidence/seed_replication_panel_resolution_v1.json').read_text())
        assert resolution['complete'] and resolution['campaign_sha256']==cp and resolution['panel_sha256']==sha(panel_file)
        assert resolution['audit_sha256']==sha(audit_file) and resolution['qualification_protocol_sha256']==audit['protocol_sha256']
        assert resolution['before_any_new_generation']
    datafile=root/campaign['data'];assert sha(datafile)==campaign['data_sha256'];data=torch.load(datafile,map_location='cpu',weights_only=False)
    tests=[dict(c,numbers=c['atomic_numbers']) for c in panel['rows']];keys={c['composition_hex'] for c in tests}
    assert len(keys)==64 and not keys&{r['condition']['composition_hex'] for r in data['training']+data['validation']}
    assert all(not corpus['matches'][h] for h in keys for corpus in audit['corpora'].values())
    spec=json.loads((root/campaign['old_connection_protocol']).read_text());assert sha(root/campaign['old_connection_protocol'])==campaign['old_connection_protocol_sha256']
    parents=[]
    for fit_info in campaign['fits']:
        arms=copy.deepcopy(fit_info['parents'])
        for family,arm in arms.items():
            path=root/arm['checkpoint']
            if arm['checkpoint_sha256'] is None:
                if fit_info['index']!=si:continue
                done=json.loads((path.parent/'complete.json').read_text());assert done['complete'] and done['campaign_sha256']==cp and done['protocol_sha256']==arm['protocol_sha256']
                arm['checkpoint_sha256']=sha(path);assert arm['checkpoint_sha256']==done['checkpoint_sha256']
        parents.append(arms)
    if si>=2:
        init=[json.loads((root/parents[si][family]['checkpoint']).with_name('initialization.json').read_text()) for family in ['fm','gaga']]
        assert init[0]['state_sha256']==init[1]['state_sha256']
    spec.update(format='seed_replication_resolved_v1',campaign_sha256=cp,parents=parents,trajectory_seeds=campaign['trajectory_seeds'],head_seeds=campaign['head_seeds'],
        evaluation_seeds=campaign['evaluation_seeds'],test_rows=tests,test_samples=16,strength_limit=4.,strengths=[0.,4.],
        panel_sha256=sha(panel_file),panel_audit_sha256=sha(audit_file))
    a.out.mkdir(parents=True,exist_ok=False);protocol=a.out/'resolved_protocol.json';write(protocol,spec);ph=sha(protocol)
    old_h=json.loads((root/campaign['old_hydrogen_protocol']).read_text());source=base.HarmonicSource();reports=[];heads={}
    for family,arm in spec['parents'][si].items():
        model=load_parent(root,arm);context=make_context(arm,source)
        if si<2:
            info=old_h['physical_heads'][str(si)][family];head,_=restore_head(root,info)
        else:
            bank=teacher(root,spec,ph,si,family,model,source,context,data['training'],a.out/family/'teacher')
            directory=a.out/family/'head';head=fit(spec,ph,si,bank,directory);info=dict(path=str((directory/'step_20000.pt').relative_to(root)),sha256=sha(directory/'step_20000.pt'))
        heads[family]=info;directory=a.out/'parents'/family
        report=generate(spec,ph,si,family,model,source,context,head,tests,campaign['evaluation_seeds'][si],16,[0.,4.],directory)
        reports.append((directory,report));del model,context,head;gc.collect();torch.cuda.empty_cache()
    physical=a.out/'parent_xtb';score(spec,ph,si,reports,physical)
    # Resolve per-fit readout provenance while preserving the original training
    # protocol for reused fits. No inference input contains reference coordinates.
    hs=copy.deepcopy(old_h);h_training=root/'runs/seed_replication_v1/hydrogen_training'
    training_file=root/('research/evidence/hydrogen_completion_v1.json' if si<2 else 'research/evidence/seed_replication_hydrogen_v1.json')
    training_spec=json.loads(training_file.read_text())
    hs.update(format='seed_replication_readout_v1',campaign_sha256=cp,test_rows=tests,methods=['base','radial','molecule_start0'],
        seeds=[f['hydrogen_seed'] for f in campaign['fits']],training_protocol_sha256=sha(training_file),
        network_spec=training_spec['network_spec'],batch_seeds=campaign['h_batch_seeds'],noise_seeds=campaign['h_noise_seeds'])
    hp=a.out/'readout_protocol.json';write(hp,hs);hph=sha(hp);sources={}
    for family in ['fm','gaga']:
        gf=a.out/'parents'/family/'generation.json';sources[family]=dict(generation=str(gf.relative_to(root)),generation_sha256=sha(gf),method=family+'_a1',
            condition_count=64,samples_per_condition=16,physical=str((physical/'results.json').relative_to(root)),physical_sha256=sha(physical/'results.json'))
    manifest=a.out/'parents.json';write(manifest,dict(complete=True,protocol_sha256=hph,seed_index=si,sources=sources))
    subprocess.run([sys.executable,'-s','-u','-m','scripts.research.evaluate_hydrogen_completion','--project',str(root),'--protocol',str(hp),
        '--training',str(h_training),'--out',str(a.out/'readouts'),'--seed-index',str(si),'--parents',str(manifest)],check=True)
    done=json.loads((a.out/'readouts/complete.json').read_text());assert done['complete']
    write(a.out/'complete.json',dict(complete=True,campaign_sha256=cp,fit=si,resolved_protocol_sha256=ph,readout_protocol_sha256=hph,
        heads=heads,parents_manifest_sha256=sha(manifest),parent_results_sha256=sha(physical/'results.json'),readout_complete_sha256=sha(a.out/'readouts/complete.json'),
        new_fit_trajectories=512 if si>=2 else 0,new_evaluation_parent_trajectories=4096,new_derived_outputs=4096,
        new_esen_queries=2048 if si>=2 else 0,new_gfn2_attempts=4096+done['new_gfn2_attempts'],new_head_updates=40000 if si>=2 else 0,
        decoder_network_example_calls=done['decoder_network_example_calls'],seconds=time.perf_counter()-tick))

if __name__=='__main__':main()
