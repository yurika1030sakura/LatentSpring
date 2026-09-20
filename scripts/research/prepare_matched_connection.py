"""Freeze a shared-EGNN physical-head transfer study before new sampling."""
import argparse,datetime,hashlib,json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def order(tag,value):return hashlib.sha256((tag+value).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.project;assert not a.out.exists();parents=[]
    for si in [0,1]:
        arms={}
        for name,proto,checkpoint in [('fm',f'gaga_feedback_distance_s{si}_v1',f'runs/gaga_feedback_v1/training/s{si}/distance/last.ckpt'),
                ('gaga',f'matched_generators_gaga_s{si}_v1',f'runs/matched_generators_v1/training/s{si}/gaga/last.ckpt')]:
            path=root/f'research/evidence/{proto}.json';spec=json.loads(path.read_text())
            saved=torch.load(root/checkpoint,map_location='cpu',weights_only=False);assert 'ema_state_dict' in saved
            arms[name]=dict(spec=spec,protocol=str(path.relative_to(root)),protocol_sha256=sha(path),checkpoint=checkpoint,checkpoint_sha256=sha(root/checkpoint))
        parents.append(arms)
    ds=parents[0]['fm']['spec'];datafile=root/ds['data'];assert sha(datafile)==ds['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False);training=data['training']
    assert len(training)==20000
    seen={r['condition']['composition_hex'] for r in training+data['validation']}
    selected=[];validation_slots=[]
    for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:
        unique={}
        for i,row in enumerate(training):
            c=row['condition']
            if lo<=c['n_atoms']<=hi and c['charge']==0 and c['spin_multiplicity']==1:unique.setdefault(c['composition_hex'],i)
        candidates=sorted(unique,key=lambda h:order('matched-connection-train-v1:',h));assert len(candidates)>=32
        start=len(selected);selected.extend(unique[h] for h in candidates[:32]);validation_slots.extend(range(start+24,start+32))
    exclusions={}
    for name,keys in [('wide_generalization_panel_v1',['rows']),('gaga_feedback_panel_v1',['validation_rows','test_rows']),
            ('physical_strength_calibration_v1',['validation_rows','test_rows']),('broad_physical_endpoints_v1',['test_rows']),
            ('physical_connection_v1',['test_rows']),('trajectory_connection_v1',['test_rows']),('context_confirmation_v1',['test_rows'])]:
        path=root/f'research/evidence/{name}.json';s=json.loads(path.read_text());exclusions[str(path.relative_to(root))]=sha(path)
        for key in keys:seen.update(c['composition_hex'] for c in s[key])
    prior=json.loads((root/'research/evidence/context_confirmation_v1.json').read_text())
    poolfile=root/prior['source_pool'];auditfile=root/prior['qualification_audit']
    assert sha(poolfile)==prior['source_pool_sha256'] and sha(auditfile)==prior['qualification_audit_sha256']
    pool=json.loads(poolfile.read_text());audit=json.loads(auditfile.read_text());eligible=[]
    for d in audit['decisions']:
        if not d['qualified']:continue
        c=pool['rows'][d['pool_index']]['condition'];h=c['composition_hex']
        if h in seen or c['charge']!=0 or c['spin_multiplicity']!=1 or not set(c['atomic_numbers'])<=set(ds['atomic_numbers']):continue
        assert all(not v['matches'][h] for v in audit['corpora'].values())
        seen.add(h);eligible.append(dict(c,numbers=c['atomic_numbers'],pool_index=d['pool_index']))
    validation=[];test=[]
    for lo,hi in [(17,28),(29,40)]:
        candidates=sorted([c for c in eligible if lo<=c['n_atoms']<=hi],key=lambda c:order('matched-connection-eval-v1:',c['composition_hex']))
        assert len(candidates)>=12,(lo,hi,len(candidates))
        validation.extend(candidates[:4]);test.extend(candidates[4:12])
    write(a.out,dict(format='matched_connection_v1',frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parents=parents,data=ds['data'],data_sha256=ds['data_sha256'],training_rows=selected,validation_teacher_slots=validation_slots,
        validation_rows=validation,test_rows=test,excluded_manifests=exclusions,
        source_pool=prior['source_pool'],source_pool_sha256=sha(poolfile),qualification_audit=prior['qualification_audit'],qualification_audit_sha256=sha(auditfile),
        oracle=prior['oracle'],xtb_binary=prior['xtb_binary'],xtb_binary_sha256=prior['xtb_binary_sha256'],
        head_configuration=dict(atomic_numbers=ds['atomic_numbers'],embedding_dim=16,hidden_dim=64,velocity_scale=2.,gate_power=2),
        trajectory_seeds=[60001,60002],head_seeds=[60011,60012],validation_seeds=[60021,60022],evaluation_seeds=[60031,60032],
        training_draws=2,backbone_calls=128,capture_calls=dict(fm=[47,59],gaga=[93,117]),
        target=dict(kT=.0258519998,max_sigma=.03,max_shift=.1),steps=20000,learning_rate=.0003,diagnostic_steps=[2000,10000,20000],
        strengths=[0.,.25,.5,1.],validation_samples=16,test_samples=16,evaluation_batch=8,
        selection='Select one shared strength per algorithm across both fits on eight separate validation compositions: maximum all-attempt graph-valid GFN2 force<=5 yield, ties choose smaller strength. Test outputs are not generated until selection is frozen.',
        primary='On the fresh16 compositions, FM selected correction minus its parent in joint graph-valid GFN2 RMS-force<=5 yield. Also report GAGA improvement, FM minus GAGA before/after equal physical supervision, and the difference in improvements, with all seed results and paired composition-bootstrap intervals.',
        gate='Primary improvement interval lower bound >0 and positive gain in both independently trained parent/head fits. An FM-over-GAGA claim additionally requires its corresponding corrected-system interval lower bound >0.',
        physical_budget='Each parent receives128 identical TRAIN compositions,256 native trajectories,512 provisional endpoints and1024 inversion-symmetrized eSEN single-point evaluations. No validity rejection. Heads train on96 TRAIN compositions;32 TRAIN compositions are used only for target-prediction diagnostics. Exactly20000 head updates per fit.',
        semantics='Head predicts A with2*progress^2 per-atom bound. Endpoint displacement=(1-progress)*A. FM adds A to velocity; GAGA adds-alpha*(noise_time/max_time)*A/sigma to epsilon. FM progress=time; GAGA progress=1-noise_time/max_time. GAGA native observation noise and fixed gamma schedule are preserved.',
        scope='Same2.38M EGNN capacity and original equal960000 backbone training-example passes; parent objectives, source laws and sampling algorithms differ. Identical physical architecture/data/query/update budgets; FM has64 head calls versus GAGA128 per trajectory at128 backbone calls, with no equal FLOP/wall-time claim. No oracle, selection or geometry optimization during test generation. No global Boltzmann claim.'))


if __name__=='__main__':main()
