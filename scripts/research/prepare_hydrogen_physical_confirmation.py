"""Freeze the physical-readout follow-up without changing the failed yield gate."""
import argparse,datetime,json
from pathlib import Path
import numpy as np
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from scripts.research.confirm_gaga_feedback import bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);a=p.parse_args();root=a.project.resolve()
    prior=root/'research/evidence/hydrogen_completion_v1.json';audit=root/'research/evidence/hydrogen_completion_selection_v1.json'
    spec=json.loads(prior.read_text());selected=json.loads(audit.read_text());assert selected['complete'] and selected['protocol_sha256']==sha(prior) and not selected['advance']
    assert selected['selected']['fm']['method']=='molecule_start0' and not (root/'runs/hydrogen_completion_v1/test').exists() and not (root/'runs/hydrogen_physical_confirmation_v1').exists()
    for path,digest in selected['provenance'].items():assert sha(root/path)==digest
    methods=['base','radial','molecule_start0'];energy=np.full((2,3,16,8),np.nan)
    for si in [0,1]:
        records=json.loads((root/f'runs/hydrogen_completion_v1/validation/s{si}/physical.json').read_text())
        for row in records['rows']:
            if not row['method'].startswith('fm_'):continue
            m=row['method'][3:]
            if m not in methods:continue
            r=row['physical']['result'];assert r['success'];energy[si,methods.index(m),row['condition_index'],row['sample_index']]=r['energy_eV']/len(r['force_eV_A'])
    assert np.isfinite(energy).all();rng=np.random.default_rng(63071);contrasts={}
    for j in [0,1]:contrasts['learned_minus_'+methods[j]]=bootstrap((energy[:,2]-energy[:,j]).mean(-1),rng,20000)
    assert all(v['ci95'][1]<0 and max(v['by_seed'])<0 for v in contrasts.values())
    diagnostic=root/'research/evidence/hydrogen_energy_nomination_v1.json';write(diagnostic,dict(complete=True,source_audit_sha256=sha(audit),contrasts=contrasts,
        scope='Secondary development energy diagnostic after the original yield gate failed. The model was already selected by joint yield; no new architecture, checkpoint, or sampler setting is chosen using energy. Not a fresh superiority result.'))
    protocol=root/'research/evidence/hydrogen_physical_confirmation_v1.json';assert not protocol.exists()
    spec.update(format='hydrogen_physical_confirmation_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_protocol=str(prior.relative_to(root)),training_protocol_sha256=sha(prior),pilot_audit=str(audit.relative_to(root)),pilot_audit_sha256=sha(audit),
        energy_nomination=str(diagnostic.relative_to(root)),energy_nomination_sha256=sha(diagnostic),methods=['base','radial','atom_start0','molecule_start0'],
        selection='All settings are frozen: EMA10000, four midpoint steps, start0, original physical parents at strength4. Evaluate identity, fixed radial projection, detached-H flow, and concerted-H flow for both families on the same preselected30 untouched compositions.',
        primary='Within FM, concerted-H flow minus fixed radial readout in all-output GFN2 energy per atom. A useful physical gain requires upper95% paired-composition interval<0, reductions in both fits, and no pooled graph/joint yield loss versus the radial control. Also compare parent and detached-H ablation, and report both GAGA readouts.',
        advance_gate='The original yield-only development gate remains FALSE. This separate physical-readout hypothesis was declared after the secondary energy diagnostic and before any new test generation. Do not call it a pass under the earlier criterion.',
        scope='Conditional learned H readout against an explicit geometric rule and unchanged parents. All four methods and failures retained. Existing successful outputs are protected and checked. No test-time energy queries or quantum geometry optimization. The changed objective and development-stage energy nomination are disclosed; no global Boltzmann or new hydrogen-decoration task claim.')
    write(protocol,spec)


if __name__=='__main__':main()
