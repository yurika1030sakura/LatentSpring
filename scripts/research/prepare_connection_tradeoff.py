"""Declare a separate joint-yield objective after the strict graph guard failed."""
import argparse,datetime,json
from pathlib import Path
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);a=p.parse_args();root=a.project
    parentfile=root/'research/evidence/extended_connection_v1.json';resultfile=root/'research/evidence/extended_connection_selection_v1.json'
    spec=json.loads(parentfile.read_text());result=json.loads(resultfile.read_text());assert result['complete'] and result['protocol_sha256']==sha(parentfile) and not result['advance']
    assert not (root/'runs/extended_connection_v1/test').exists() and not (root/'runs/connection_tradeoff_v1/test').exists()
    for file,digest in result['validation_provenance'].items():assert sha(root/file)==digest
    choices={name:max(curve,key=lambda k:(curve[k]['joint_rate'],-float(k))) for name,curve in result['curves'].items()}
    protocol=root/'research/evidence/connection_tradeoff_v1.json';selection=root/'research/evidence/connection_tradeoff_selection_v1.json';assert not protocol.exists() and not selection.exists()
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    spec.update(format='connection_tradeoff_v1',at_utc=now,calibration_protocol=str(parentfile.relative_to(root)),calibration_protocol_sha256=sha(parentfile),
        calibration_audit=str(resultfile.relative_to(root)),calibration_audit_sha256=sha(resultfile),
        selection='Select the maximum pooled validation joint yield for each algorithm over the same six strengths, tie toward smaller strength. This separate follow-up removes the within-model graph noninferiority guard; both graph rates and the original strength1 controls must be reported on fresh test.',
        advance_gate='The strict guard in extended_connection_v1 failed and remains failed. This distinct confirmation tests the validation joint-yield maximizers despite their measured tradeoff against their own strength1 graph rates. No result from the reserved32-composition panel is used for this decision.',
        primary='Selected FM minus equally selected GAGA on all-attempt graph-valid GFN2-force<=5 yield. Report graph validity as a co-primary readout, both parent fits, full threshold curves, physical energy diagnostics, diversity, and measured sampling time. Retain strength1 controls for both algorithms to expose the tradeoff.',
        provenance_scope='New objective declared after reading only the eight-composition extended validation results and before any fresh confirmation outputs. This is not a successful advance under the earlier strict graph guard. The exact same32 metadata-selected untouched conditions are retained; no panel replenishment.',
        expected_new_validation_outputs=0,scope=spec['scope']+' This fresh experiment explicitly evaluates the graph/force tradeoff. A comprehensive win requires supported advantages on both joint yield and graph validity; mean validation ordering alone is insufficient.')
    write(protocol,spec)
    write(selection,dict(complete=True,protocol_sha256=sha(protocol),at_utc=now,advance=True,
        policy='Separate joint-yield tradeoff confirmation; original strict guard failed',original_guard_passed=False,
        strengths={name:float(k) for name,k in choices.items()},validation_provenance=result['validation_provenance'],
        source_calibration_sha256=sha(resultfile),fresh_test_outputs_used=False,new_neural_outputs=0,new_gfn2_attempts=0,
        selected_validation={name:result['curves'][name][k] for name,k in choices.items()}))


if __name__=='__main__':main()
