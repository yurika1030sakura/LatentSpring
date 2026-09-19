"""Freeze broader TRAIN targets and a previously ungenerated confirmation panel."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project;assert not a.out.exists()
    original=json.loads((root/'research/evidence/source_sc_s0_v1.json').read_text())
    datafile=root/original['data'];assert sha(datafile)==original['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False);unique={}
    for i,row in enumerate(data['training']):unique.setdefault(row['condition']['composition_hex'],i)
    train_rows=[]
    for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:
        candidates=[i for i in unique.values() if lo<=data['training'][i]['condition']['n_atoms']<=hi]
        candidates.sort(key=lambda i:hashlib.sha256(('broad-physical-target-train-v1:'+data['training'][i]['condition']['composition_hex']).encode()).hexdigest())
        assert len(candidates)>=32;train_rows.extend(candidates[:32])
    inherited=json.loads((root/'research/evidence/gaga_feedback_panel_v1.json').read_text())
    poolfile=root/inherited['source_pool'];pool=json.loads(poolfile.read_text());auditfile=root/inherited['qualification_audit'];audit=json.loads(auditfile.read_text())
    assert sha(poolfile)==inherited['source_pool_sha256'] and sha(auditfile)==inherited['qualification_audit_sha256']
    assert pool['complete'] and audit['complete'] and not pool['reserved_outcomes_allowed']
    exclusions={r['condition']['composition_hex'] for r in data['training']+data['prior_validation']}
    excluded_panels={'research/evidence/wide_generalization_panel_v1.json':['rows'],
        'research/evidence/gaga_feedback_panel_v1.json':['validation_rows','test_rows'],
        'research/evidence/physical_strength_calibration_v1.json':['validation_rows','test_rows']}
    for file,keys in excluded_panels.items():
        d=json.loads((root/file).read_text())
        for key in keys:exclusions|={r['composition_hex'] for r in d[key]}
    eligible=[]
    for decision in audit['decisions']:
        if not decision['qualified']:continue
        c=pool['rows'][decision['pool_index']]['condition']
        if c['composition_hex'] in exclusions:continue
        assert all(not corpus['matches'][c['composition_hex']] for corpus in audit['corpora'].values())
        eligible.append(dict(c,pool_index=decision['pool_index']))
    test=[]
    for lo,hi in [(17,28),(29,40)]:
        candidates=[c for c in eligible if lo<=c['n_atoms']<=hi]
        candidates.sort(key=lambda c:hashlib.sha256(('broad-physical-target-test-v1:'+c['composition_hex']).encode()).hexdigest())
        assert len(candidates)>=8;test.extend(candidates[:8])
    previous=json.loads((root/'research/evidence/matched_physical_s0_v2.json').read_text())
    checkpoint='runs/source_sc_v1/s0/study/harmonic_tree/last.ckpt'
    write(a.out,dict(frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        data=original['data'],data_sha256=sha(datafile),training_rows=train_rows,
        training_selection='One original TRAIN reference per composition;32 compositions in each8-16/17-24/25-32/33-40 bin by fixed salted hash. No energy or model outcome selection.',
        config=original['config'],config_sha256=sha(root/original['config']),checkpoint=checkpoint,checkpoint_sha256=sha(root/checkpoint),
        test_rows=test,source_pool=inherited['source_pool'],source_pool_sha256=sha(poolfile),qualification_audit=inherited['qualification_audit'],qualification_audit_sha256=sha(auditfile),
        excluded_panels={f:sha(root/f) for f in excluded_panels},
        test_selection='Previously qualified but previously ungenerated pool compositions, disjoint from all original processed corpora and prior64/32/16 model panels. Eight in each17-28 and29-40 bin. Reserved722 outcomes excluded by pool construction.',
        optimizer=dict(max_steps=96,max_evaluations=160,force_tolerance_eV_A=.1,max_atom_step_A=.05,
            initial_inverse_hessian_A2_eV=.02,history_size=10,backtracking_steps=10,armijo=1e-4,energy_tolerance_eV=2e-5),
        oracle={k:previous[k] for k in ['oracle_checkpoint','oracle_interpreter','oracle_sha256','oracle_worker_sha256']},
        training_seeds=[52001,52002],evaluation_seeds=[52051,52052],training_steps=2000,
        learning_rate=2e-5,feedback_learning_rate=3e-4,endpoint_smoothing_A=.005,
        methods=['base','reference_ft','relaxed_ft'],samples_per_condition=16,midpoint_steps=32,backbone_calls=128,
        xtb_binary=previous['xtb_binary'],xtb_binary_sha256=previous['xtb_binary_sha256'],xtb=previous['xtb'],thresholds=[.1,.5,1.,2.,5.,10.,20.],
        primary='Relaxed-reference FT minus original-reference FT in all-attempt graph-valid GFN2 force<=5 yield; both FT methods also compared with the parent. Two fitted seeds and composition-bootstrap intervals reported.',
        training_budget='Two2000-step students per seed, batch1, two supervised network passes. Same reference IDs, source/augmentation seeds, smoothing, optimizer and parent; only target coordinates differ.',
        inference='128 backbone calls, zero terminal noise, no online energy evaluations or geometry optimization.',
        target='Empirical uniform distribution over128 TRAIN compositions and the specified capped graph-preserving relaxation map. Retain every composition including unchanged/stalled targets. Report stationarity separately. No thermal/Boltzmann or work-weighting benefit is claimed by this geometry-only control.',
        scope='Broader physical endpoint learning, not a new optimizer, full-corpus training, GAGA superiority claim, or a pure training-coverage ablation against the earlier500-step pilot.'))
    print(json.dumps(dict(training_compositions=len(train_rows),fresh_test_compositions=len(test))),flush=True)


if __name__=='__main__':main()
