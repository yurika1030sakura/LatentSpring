"""Freeze complete-model confirmation, including the existing GAGA baselines."""
import argparse,datetime,hashlib,json
from pathlib import Path
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','out','wide-protocol']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();a.out=a.out.resolve();a.wide_protocol=a.wide_protocol.resolve()
    assert not a.out.exists() and not a.wide_protocol.exists()
    capfile=root/'research/evidence/connection_capacity_v1.json';cap=json.loads(capfile.read_text())
    wide=dict(cap,format='connection_capacity_wide_v1',variants=['wide'],wide_config=dict(cap['pair_config'],hidden_dim=170),basis_probe_states=[],
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=cap['scope']+' Width170 gives37540 parameters, versus37586 in the contextual head; this controls approximate parameter count.')
    write(a.wide_protocol,wide)
    s=json.loads((root/'research/evidence/trajectory_connection_v1.json').read_text());excluded=set()
    for name,keys in [('wide_generalization_panel_v1',['rows']),('gaga_feedback_panel_v1',['validation_rows','test_rows']),('physical_strength_calibration_v1',['validation_rows','test_rows']),('broad_physical_endpoints_v1',['test_rows']),('physical_connection_v1',['test_rows']),('trajectory_connection_v1',['test_rows'])]:
        d=json.loads((root/f'research/evidence/{name}.json').read_text())
        for key in keys:excluded|={c['composition_hex'] for c in d[key]}
    poolfile=root/s['source_pool'];auditfile=root/s['qualification_audit'];assert sha(poolfile)==s['source_pool_sha256'] and sha(auditfile)==s['qualification_audit_sha256']
    pool=json.loads(poolfile.read_text());audit=json.loads(auditfile.read_text());eligible=[]
    vocab=set(json.loads((root/'research/evidence/matched_generators_gaga_s0_v1.json').read_text())['atomic_numbers'])
    for row in audit['decisions']:
        if not row['qualified']:continue
        c=pool['rows'][row['pool_index']]['condition']
        if c['composition_hex'] in excluded or c['charge']!=0 or c['spin_multiplicity']!=1 or not set(c['atomic_numbers'])<=vocab:continue
        assert all(not d['matches'][c['composition_hex']] for d in audit['corpora'].values())
        eligible.append(dict(c,pool_index=row['pool_index']))
    test=[]
    for lo,hi in [(17,28),(29,40)]:
        candidates=[c for c in eligible if lo<=c['n_atoms']<=hi]
        candidates.sort(key=lambda c:hashlib.sha256(('context-confirmation-v1:'+c['composition_hex']).encode()).hexdigest());assert len(candidates)>=8
        test.extend(candidates[:8])
    sources=[]
    for si in [0,1]:
        heads={}
        for name,kind in [('pair_long','pair'),('context','context')]:
            path=root/f'runs/connection_capacity_v1/s{si}/{kind}/step_20000.pt'
            heads[name]=dict(path=str(path.relative_to(root)),sha256=sha(path),protocol_sha256=sha(capfile))
        proto=root/f'research/evidence/matched_generators_gaga_s{si}_v1.json';info=json.loads((root/f'research/evidence/gaga_feedback_distance_s{si}_v1.json').read_text())['baselines']['gaga'];assert sha(root/info['path'])==info['sha256']
        sources.append(dict(heads=heads,gaga=dict(protocol=str(proto.relative_to(root)),protocol_sha256=sha(proto),checkpoint=info['path'],checkpoint_sha256=info['sha256'])))
    s.update(format='context_confirmation_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_rows=test,
        training_seeds=cap['seeds'],evaluation_seeds=[57051,57052],methods=['base','pair_long','pair_wide','context','gaga','gaga_full'],
        wide_protocol=str(a.wide_protocol.relative_to(root)),wide_protocol_sha256=sha(a.wide_protocol),sources=sources,
        training_steps=20000,physical_connection=cap['context_config'],
        teacher_audit='research/evidence/trajectory_teacher_audit_v1.json',teacher_audit_sha256=sha(root/'research/evidence/trajectory_teacher_audit_v1.json'),
        primary='Contextual head minus unadapted parent on all-attempt graph-valid GFN2 force<=5 yield; pair-long and nearly parameter-matched wide-pair controls at20000 updates. Both seeds and all arms reported.',
        comparison_scope='GAGA is the existing position-conditional OMol25 adaptation, at128 and its complete651 calls. Full FlowMol and EGNN systems differ in architecture, capacity, training data/history and cost. This is a common-task full-model comparison, not a matched-backbone ablation or native published GAGA reproduction. Earlier shared-EGNN comparisons remain separately reported.',
        inference='FlowMol arms use128 backbone+64 head calls. GAGA uses128 or651 backbone calls and no head. No equal FLOP, total-call or wall-time claim. All outputs unoptimized and no physical oracle during sampling.',
        scope='Fresh16-composition full-model confirmation and physical-head ablations. Separate physical-target prediction from actual generation gains. No global Boltzmann or GAGA-family superiority claim.')
    write(a.out,s)


if __name__=='__main__':main()
