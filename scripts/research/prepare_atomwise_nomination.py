"""Freeze a separate confirmation after a near-threshold development nomination."""
import argparse,datetime,json
from pathlib import Path
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);a=p.parse_args();root=a.project.resolve()
    pilot=root/'research/evidence/atomwise_connection_v1.json';result=root/'research/evidence/atomwise_connection_selection_v1.json'
    spec=json.loads(pilot.read_text());selected=json.loads(result.read_text());assert selected['complete'] and selected['protocol_sha256']==sha(pilot) and not selected['advance']
    assert not (root/'runs/atomwise_connection_v1/test').exists() and not (root/'runs/atomwise_confirmation_v1/test').exists()
    for file,digest in selected['validation_provenance'].items():assert sha(root/file)==digest
    # The nomination is documented as an exception to the DEVELOPMENT resource
    # screen. It does not change the subsequent statistical success criterion.
    fm,control=selected['selected']['fm'],selected['controls']['fm']
    assert fm['variant']=='atom_balanced' and selected['validation_fm_gain']==5/256 and all(v>0 for v in selected['validation_fm_gain_by_seed'])
    assert all(l>r for l,r in zip(fm['graph_by_seed'],control['graph_by_seed']))
    protocol=root/'research/evidence/atomwise_confirmation_v1.json';selection=root/'research/evidence/atomwise_confirmation_selection_v1.json'
    assert not protocol.exists() and not selection.exists();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    spec.update(format='atomwise_confirmation_v1',at_utc=now,pilot_protocol=str(pilot.relative_to(root)),pilot_protocol_sha256=sha(pilot),
        pilot_selection=str(result.relative_to(root)),pilot_selection_sha256=sha(result),
        advance_gate='The original minimum2pp development screen is FALSE: the gain is5/256=1.953125pp. A separate confirmation is nominated because graph and joint yields improve in both fits and the combined variant has the highest validation joint yield. Do not relabel the original screen a pass. Fresh-test success still requires positive paired95% intervals and improvements in both fitted models.',
        confirmation='No setting, test composition, parent, head, or control changes after nomination. Run all four factorial variants for both generators at their validation-selected strengths, plus each independently selected original control strength if different. Primary comparison is selected FM versus its original control; GAGA comparison and component contrasts are also reported.',
        nomination_scope='Nomination was made after viewing only the development result and before any fresh output. The exact preselected32-composition panel is retained. No subsequent tuning or panel replenishment on its outcomes.',
        expected_new_validation_outputs=0,expected_new_head_updates=0)
    write(protocol,spec)
    nominated=dict(selected,protocol_sha256=sha(protocol),at_utc=now,advance=True,original_development_gate_passed=False,
        nomination_reason='Positive graph/joint changes in both fits; joint gain1.953125pp missed the development2pp screen by less than one output. Independent confirmation, not declaration of success.',
        pilot_selection_sha256=sha(result),new_neural_outputs=0,new_gfn2_attempts=0,new_optimizer_steps=0,new_esen_queries=0)
    write(selection,nominated)


if __name__=='__main__':main()
