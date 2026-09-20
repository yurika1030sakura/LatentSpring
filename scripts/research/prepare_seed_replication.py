"""Freeze independent fits and a new metadata-only panel for the final comparison."""
import copy,datetime,json
from pathlib import Path
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write

ROOT=Path(__file__).resolve().parents[2]
def main():
    root=ROOT;out=root/'research/evidence/seed_replication_v1.json';assert not out.exists()
    original=root/'research/evidence/matched_connection_v1.json';connection=json.loads(original.read_text())
    hfile=root/'research/evidence/hydrogen_physical_confirmation_v1.json';hydrogen=json.loads(hfile.read_text())
    fits=[]
    for si in range(5):
        entry=dict(index=si,old=si<2,parents={})
        for family in ['fm','gaga']:
            old=connection['parents'][si if si<2 else 0][family]
            spec=copy.deepcopy(old['spec'])
            if si>=2:
                spec.update(initialization_seed=64101+si,batch_seed=64201+si,noise_seed=64301+si,
                    format='independent_seed_replication_parent_v1',statistical_scope='Three additional prespecified initializations, no checkpoint or hyperparameter selection.',
                    checkpoint_every=1000,validation_steps=[])
                for key in ['baselines','panel','panel_sha256','campaign']:spec.pop(key,None)
                f=root/f'research/evidence/seed_replication_{family}_s{si}_v1.json';assert not f.exists();write(f,spec)
                arm=dict(spec=spec,protocol=str(f.relative_to(root)),protocol_sha256=sha(f),
                    checkpoint=f'runs/seed_replication_v1/training/s{si}/{family}/last.ckpt',checkpoint_sha256=None)
            else:arm=copy.deepcopy(old)
            entry['parents'][family]=arm
        entry['hydrogen_seed']=63011+si if si<2 else 64401+si
        fits.append(entry)
    poolfile=root/'research/evidence/seed_replication_pool_v1.json';pool=json.loads((root/'research/evidence/generalization_pool_protocol_v2.json').read_text())
    pool.update(format='seed_replication_pool_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        size_bins=[[17,28],[29,40]],pool_per_bin=512,panel_per_bin=32,selection_seed=64501,rank_seed=64502,
        amendment='A new pool and new independent training fits; all existing pool_v2 compositions excluded before selecting by metadata. No model outcomes or energy ranks used.')
    pool['exclusion_manifests']['runs/benchmark_expansion_v1/pool_v2.json']=sha(root/'runs/benchmark_expansion_v1/pool_v2.json')
    write(poolfile,pool)
    spec=dict(format='seed_replication_v1',frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        fits=fits,old_connection_protocol=str(original.relative_to(root)),old_connection_protocol_sha256=sha(original),
        old_hydrogen_protocol=str(hfile.relative_to(root)),old_hydrogen_protocol_sha256=sha(hfile),
        data=connection['data'],data_sha256=connection['data_sha256'],pool_protocol=str(poolfile.relative_to(root)),pool_protocol_sha256=sha(poolfile),
        panel='runs/seed_replication_v1/panel/panel.json',panel_audit='runs/seed_replication_v1/panel/audit.json',
        condition_count=64,samples_per_condition=16,evaluation_batch=8,backbone_calls=128,
        strengths=[0.,4.],readouts=['base','radial','molecule_start0'],
        evaluation_seeds=[64601+s for s in range(5)],trajectory_seeds=[60001,60002,64703,64704,64705],
        head_seeds=[60011,60012,64803,64804,64805],h_batch_seeds=[63021,63022,64903,64904,64905],h_noise_seeds=[63031,63032,65003,65004,65005],
        decoder_steps=4,velocity_cap_A=2.,parent_strength=4.,radial_contact=1.25,radial_target=1.,
        parent_training='Each FM fit15000x32x2 and GAGA fit30000x32x1 =960000 backbone example passes. Same initialization within each fit pair, data and batch-prefix schedule. All three new fits reported; no model selection.',
        physical_training='Unchanged7106-parameter global pair head,128 identical TRAIN compositions,256 native trajectories,1024 eSEN queries and20000 head updates per parent. Reuse old fits0/1; collect new labels for fits2/3/4.',
        hydrogen_training='Unchanged153790-total-parameter H EGNN,10000 updates x32, final EMA, same20k references. One decoder shared between FM/GAGA per fit. Reuse old0/1 and train2/3/4.',
        primary='Replicate within-FM physical-head improvement in all-attempt graph-valid GFN2 RMSforce<=5 yield at fixed strength4 versus0. Report each new fit separately, three-new-fit mean and five-fit mean. Confirm H-flow energy benefit versus the fixed radial rule as the second planned endpoint.',
        secondary='Equally augmented FM versus GAGA in graph validity and joint yield; report intervals and every fit, without mandatory superiority or retuning after outcomes.',
        statistics=dict(bootstrap_repetitions=20000,bootstrap_seed=65101,primary_unit='composition with all fitted models retained',
            training_uncertainty='Also report crossed bootstrap resampling both fit indices and composition indices; five fits remain a limited sample of training randomness.',
            new_fit_replication='Primary independent-training conclusion uses fits2/3/4; pooled five-fit statistics are supplementary because old fits informed method development.'),
        planned_new_parent_training_updates=135000,planned_new_parent_training_forward_examples=5760000,
        planned_new_head_updates=120000,planned_new_h_updates=30000,planned_new_h_training_forward_examples=960000,
        planned_new_fit_trajectories=1536,planned_new_esen_queries=6144,planned_new_evaluation_parent_trajectories=20480,
        planned_derived_readout_outputs=20480,maximum_new_gfn2_attempts=40960,
        no_new_settings_after_outcomes=True,reserved_outcomes_allowed=False,
        scope='Frozen method replication, not a new architecture search. All attempts remain in denominators. Generation uses no energy queries, ranking or quantum optimization. The new panel is excluded from both processed corpora and20k model-training references before any sampling.')
    write(out,spec)
    print(json.dumps(dict(protocol=str(out),new_parent_fits=6,new_hydrogen_fits=3,new_panel=64)))

if __name__=='__main__':main()
