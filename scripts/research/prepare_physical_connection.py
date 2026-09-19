"""Freeze a small geometry-correction network and fresh matched controls."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from ase.data import atomic_numbers
from flowmol.model_utils.load import read_config_file
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project;assert not a.out.exists()
    previous=root/'research/evidence/broad_physical_endpoints_v1.json';spec=json.loads(previous.read_text())
    poolpath=root/spec['source_pool'];pool=json.loads(poolpath.read_text());auditpath=root/spec['qualification_audit'];audit=json.loads(auditpath.read_text())
    assert sha(poolpath)==spec['source_pool_sha256'] and sha(auditpath)==spec['qualification_audit_sha256']
    excluded={c['composition_hex'] for c in spec['test_rows']}
    for file,keys in [('research/evidence/wide_generalization_panel_v1.json',['rows']),('research/evidence/gaga_feedback_panel_v1.json',['validation_rows','test_rows']),('research/evidence/physical_strength_calibration_v1.json',['validation_rows','test_rows'])]:
        data=json.loads((root/file).read_text())
        for k in keys:excluded|={c['composition_hex'] for c in data[k]}
    eligible=[]
    for decision in audit['decisions']:
        if not decision['qualified']:continue
        c=pool['rows'][decision['pool_index']]['condition']
        if c['composition_hex'] in excluded:continue
        assert all(not corpus['matches'][c['composition_hex']] for corpus in audit['corpora'].values())
        eligible.append(dict(c,pool_index=decision['pool_index']))
    test=[]
    for lo,hi in [(17,28),(29,40)]:
        candidates=[c for c in eligible if lo<=c['n_atoms']<=hi]
        candidates.sort(key=lambda c:hashlib.sha256(('physical-connection-test-v1:'+c['composition_hex']).encode()).hexdigest());assert len(candidates)>=8
        test.extend(candidates[:8])
    config=read_config_file(root/spec['config'])
    spec.update(format='physical_connection_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        teacher_protocol='research/evidence/broad_physical_endpoints_v1.json',teacher_protocol_sha256=sha(previous),
        teacher_bank='runs/broad_physical_endpoints_v1/teacher/bank.pt',teacher_bank_sha256=sha(root/'runs/broad_physical_endpoints_v1/teacher/bank.pt'),
        training_seeds=[53001,53002],evaluation_seeds=[53051,53052],test_rows=test,
        methods=['base','relaxed_ft','connection_ref','connection_physical'],
        physical_connection=dict(atomic_numbers=[atomic_numbers[s] for s in config['dataset']['atom_map']],embedding_dim=16,hidden_dim=64,velocity_scale=1.,gate_power=2),
        primary='Physical-target correction network minus zero-correction parent on all-attempt GFN2 graph-valid force<=5 yield; compare with matched original-reference correction and full fine-tuning. All methods and both seeds reported.',
        training_budget='Each trained method uses2000 updates,batch1 and4000 backbone forward examples. Full FT updates the backbone with its original two-pass loss. Correction methods freeze every parent tensor and supervise the corrected final velocity, adding2000 small-head forward examples and head-only backward passes. Same target IDs/noise within a seed; measured costs reported, not equal wall time.',
        inference='All methods use128 backbone calls plus64 small-head calls. Parent and full-FT controls receive the same zero-initialized head for matching sampling operations. No physical oracle or optimization in generation.',
        architecture='An8178-parameter symmetric pair MLP forms antisymmetric velocity corrections from current and parent-predicted geometry, atomic identities and time. Soft contacts are geometric weights, not supplied bonds or bond probabilities. A t^2 gate and degree normalization bound each instantaneous correction norm by t^2 Angstrom per unit flow time.',
        target=spec['target'],scope='Candidate frozen-parent physical geometry correction. The pairwise equivariant construction is established practice; utility and novelty depend on task-specific evidence. No global-equilibrium, exact-graph-preservation, exact endpoint-displacement-bound, or GAGA-superiority claim.')
    write(a.out,spec);print('Frozen correction protocol with16 fresh compositions',flush=True)


if __name__=='__main__':main()
