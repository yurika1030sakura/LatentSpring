#!/usr/bin/env python3
"""Freeze matched physical adaptation for the validated FM and GAGA settings."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);a=p.parse_args();root=a.project
    selection_path=root/'runs/gaga_feedback_v1/selection.json';selection=json.loads(selection_path.read_text());assert selection['complete']
    audit_path=root/'runs/gaga_feedback_v1/audit.json';audit=json.loads(audit_path.read_text());assert audit['complete']
    assert selection['selected_challenger']=='distance' and selection['selected_gaga_max_t']==650
    template=json.loads((root/'research/evidence/gaga_feedback_distance_s0_v1.json').read_text())
    data=torch.load(root/template['data'],map_location='cpu',weights_only=False);assert sha(root/template['data'])==template['data_sha256']
    counts=Counter(r['condition']['composition_hex'] for r in data['training']);first={}
    for index,row in enumerate(data['training']):
        c=row['condition']
        if 17<=c['n_atoms']<=28:first.setdefault(c['composition_hex'],index)
    ordered=sorted(first,key=lambda key:(-counts[key],hashlib.sha256(('matched-physics-fit-v1:'+key).encode()).hexdigest()))
    chosen=[first[key] for key in ordered[:8]];assert len(chosen)==8
    physics=json.loads((root/'research/evidence/source_physical_factorial_s0_v1.json').read_text())
    for seed in [0,1]:
        arms={}
        for name in ['distance','gaga']:
            path=root/(f'research/evidence/gaga_feedback_distance_s{seed}_v1.json' if name=='distance' else f'research/evidence/matched_generators_gaga_s{seed}_v1.json')
            spec=json.loads(path.read_text());checkpoint=(f'runs/gaga_feedback_v1/training/s{seed}/distance/last.ckpt' if name=='distance' else f'runs/matched_generators_v1/training/s{seed}/gaga/last.ckpt')
            base_label='distance' if name=='distance' else 'gaga_650';report=f'runs/gaga_feedback_v1/confirmation/s{seed}/{base_label}_results.json'
            arms[name]=dict(spec=spec,spec_path=str(path.relative_to(root)),spec_sha256=sha(path),checkpoint=checkpoint,checkpoint_sha256=sha(root/checkpoint),
                base_label=base_label,base_report=report,base_report_sha256=sha(root/report),
                student_steps=1000 if name=='distance' else 2000,passes_per_example=2 if name=='distance' else 1)
        record=dict(format='matched_physical_v1',frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_seed=seed,
            data=template['data'],data_sha256=template['data_sha256'],panel=template['panel'],panel_sha256=template['panel_sha256'],
            selection=str(selection_path.relative_to(root)),selection_sha256=sha(selection_path),parent_audit=str(audit_path.relative_to(root)),parent_audit_sha256=sha(audit_path),arms=arms,
            teacher_rows=chosen,teacher_selection='Eight most frequent TRAIN compositions with17-28 atoms, ties by fixed metadata hash; first row supplies atom order only. No generated outcome or energy chooses conditions.',
            teacher_composition_counts=[counts[data['training'][i]['condition']['composition_hex']] for i in chosen],teacher_draws=64,minimum_teacher_compositions=4,
            teacher_seed=46101+seed,training_seed=46201+seed,thermal_teacher=physics['thermal_teacher'],learning_rate=1e-4,
            inference_calls=128,evaluation_seed=43101+seed,evaluation_batch=8,samples_per_condition=32,
            oracle_checkpoint=physics['oracle_checkpoint'],oracle_sha256=physics['oracle_sha256'],oracle_interpreter=physics['oracle_interpreter'],oracle_worker_sha256=physics['oracle_worker_sha256'],
            xtb_binary=physics['xtb_binary'],xtb_binary_sha256=physics['xtb_binary_sha256'],xtb=physics['xtb'],
            force_thresholds=[1.,2.,5.,10.,20.,50.,100.],primary_force_threshold=5.,
            primary='Paired-physical harmonic FM with distance self-conditioning versus paired-physical GAGA, on the same2381566-parameter backbone and20000 original training rows.',
            primary_gate='Positive joint graph/force<=5 yield difference in each seed and positive lower bound of a paired composition-bootstrap95 interval under BOTH eSEN and GFN2. All outcomes reported regardless of gate.',
            controls='Compare physical adaptation versus each own frozen base; base outputs reused from the audited32-composition confirmation.',
            budget='Each student gets2000 supervised backbone example passes:1000 two-pass FM updates or2000 one-pass GAGA updates. Replay and physical students are paired within each parent. Original960000 backbone example training passes per parent and128 inference calls are matched; optimizer steps, data presentations, matrix overhead and actual wall time are reported separately.',
            scope='Fixed follow-up on an already evaluated32-composition panel. Selection of the two parent settings uses the earlier validation choice; no new untouched-test or global Boltzmann claim.',
            no_replenishment=True,reserved_outcomes_allowed=False,paired_coefficient=1.,
            expected_new_fit_outputs_per_seed=1024,expected_new_evaluation_outputs_per_seed=2048,expected_student_updates_per_seed=6000,
            expected_esen_evaluation_queries_per_seed=8256,maximum_teacher_queries_per_seed=18432,expected_gfn2_attempts_per_seed=4160,
            reference_inversion_energy_tolerance_eV=1e-4,reference_inversion_force_tolerance_eV_A=1e-3)
        target=root/f'research/evidence/matched_physical_s{seed}_v1.json';assert not target.exists();target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(teacher_rows=chosen,counts=[counts[data['training'][i]['condition']['composition_hex']] for i in chosen])))


if __name__=='__main__':main()
