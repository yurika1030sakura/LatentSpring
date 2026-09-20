"""Record completed follow-ups without counting coordinate readouts as trajectories."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def save(p,d):
    file=ROOT/p
    if file.exists():raise FileExistsError(file)
    file.write_text(json.dumps(d,indent=2)+'\n')

def main():
    studies=[
        (32,'atomwise_validation','atomwise_connection_selection_v1',6656,6656,240000,0,0),
        (33,'atomwise_confirmation','atomwise_confirmation_audit_v1',8192,8192,0,0,0),
        (34,'hydrogen_training_and_validation','hydrogen_completion_selection_v1',0,155,20000,2560,992),
        (35,'hydrogen_physical_confirmation','hydrogen_physical_confirmation_audit_v1',1920,2460,0,5760,2880)]
    for version,name,audit,parents,gfn,steps,derived,calls in studies:
        previous=f'research/evidence/generator_reference_registry_v{version-1}.json';old=read(previous)
        file=f'research/evidence/{audit}.json';d=read(file);assert d['complete'] and d['new_gfn2_attempts']==gfn and d['new_optimizer_steps']==steps
        assert d.get('new_parent_trajectories',d.get('new_neural_outputs'))==parents
        record=dict(previous_registry=previous,previous_registry_sha256=sha(ROOT/previous),at_utc=datetime.now(timezone.utc).isoformat(),
            studies={name:dict(new_eval_neural_trajectories=parents,new_fit=0,optimizer_steps=steps,esen=0,xtb_generator=gfn,
                derived_evaluation_outputs=derived,decoder_network_example_calls=calls)},
            new_neural_generation_outputs=parents,new_evaluation_generator_outputs=parents,new_fit_generator_outputs=0,
            cumulative_tree_architecture_evaluation_outputs=old['cumulative_tree_architecture_evaluation_outputs']+parents,
            cumulative_fit_generator_outputs=old['cumulative_fit_generator_outputs'],
            cumulative_neural_generation_records=old['cumulative_neural_generation_records']+parents,
            new_raw_esen_queries=0,cumulative_source_teacher_raw_physical_rows=old['cumulative_source_teacher_raw_physical_rows'],
            new_gfn2_attempts=gfn,cumulative_gfn2_confirmation_and_followup_attempts=old['cumulative_gfn2_confirmation_and_followup_attempts']+gfn,
            new_optimizer_steps=steps,new_derived_evaluation_outputs=derived,
            cumulative_derived_evaluation_outputs_since_v34=old.get('cumulative_derived_evaluation_outputs_since_v34',0)+derived,
            decoder_network_example_calls=calls,ongoing_studies_not_yet_in_completed_totals=[],reserved_outcomes_unqueried=722,
            evidence={file:sha(ROOT/file)})
        if version==34:
            fits=[read(f'runs/hydrogen_completion_v1/training/s{s}/complete.json') for s in [0,1]]
            record.update(new_training_forward_examples=sum(x['new_training_forward_examples'] for x in fits),
                new_diagnostic_forward_examples=sum(x['diagnostic_forward_examples'] for x in fits),new_derived_train_probe_outputs=1024)
            record['scope']='Two hydrogen models shared by both parents. TRAIN radial diagnostic reuses1024 cached outputs. Development yields fail the original selection gate. Derived readouts, training forwards, and reused physical scores are not new parent trajectories. Only155 changed-coordinate GFN2 calculations are new.'
        elif version==35:record['scope']='Separate physical hypothesis frozen before30 fresh compositions. All1920 parent outputs and540 unique changed-coordinate readouts scored.5760 derived outputs are separate from1920 trajectories. Physical gate passes; original development yield screen remains false. No globally equilibrated molecular distribution claim.'
        else:record['scope']='All factorial variants and failures retained. Atomwise development gate and independent confirmation gates fail; these models are not promoted.'
        save(f'research/evidence/generator_reference_registry_v{version}.json',record)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
