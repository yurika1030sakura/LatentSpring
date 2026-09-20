"""Freeze normalization/mobility ablations on TRAIN forces, before new validation."""
import argparse,datetime,hashlib,json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project.resolve();assert not a.out.exists()
    original=root/'research/evidence/matched_connection_v1.json';selection=root/'research/evidence/matched_connection_selection_v1.json'
    spec=json.loads(original.read_text());old=json.loads(selection.read_text());assert old['complete'] and old['protocol_sha256']==sha(original)
    datafile=root/spec['data'];assert sha(datafile)==spec['data_sha256'];data=torch.load(datafile,map_location='cpu',weights_only=False)
    train={r['condition']['composition_hex'] for r in data['training']};validation={r['condition']['composition_hex'] for r in data['validation']}
    panel=root/'research/evidence/gaga_feedback_panel_v1.json';pinfo=json.loads(panel.read_text())
    val=[dict(c,numbers=c['atomic_numbers']) for c in pinfo['validation_rows'] if 17<=c['n_atoms']<=40]
    assert len(val)==16 and {c['composition_hex'] for c in val}<=validation and not train&{c['composition_hex'] for c in val}
    excluded=train|validation;exclusions={}
    for name,keys in [('wide_generalization_panel_v1',['rows']),('gaga_feedback_panel_v1',['validation_rows','test_rows']),
            ('physical_strength_calibration_v1',['validation_rows','test_rows']),('broad_physical_endpoints_v1',['test_rows']),
            ('physical_connection_v1',['test_rows']),('trajectory_connection_v1',['test_rows']),('context_confirmation_v1',['test_rows']),
            ('matched_connection_v1',['validation_rows','test_rows']),('connection_tradeoff_v1',['validation_rows','test_rows'])]:
        f=root/f'research/evidence/{name}.json';s=json.loads(f.read_text());exclusions[str(f.relative_to(root))]=sha(f)
        for key in keys:excluded.update(c['composition_hex'] for c in s[key])
    poolfile=root/spec['source_pool'];auditfile=root/spec['qualification_audit'];assert sha(poolfile)==spec['source_pool_sha256'] and sha(auditfile)==spec['qualification_audit_sha256']
    pool=json.loads(poolfile.read_text());audit=json.loads(auditfile.read_text());eligible=[]
    for row in audit['decisions']:
        if not row['qualified']:continue
        c=pool['rows'][row['pool_index']]['condition'];h=c['composition_hex']
        if h in excluded or c['charge']!=0 or c['spin_multiplicity']!=1 or not set(c['atomic_numbers'])<=set(spec['head_configuration']['atomic_numbers']):continue
        if any(v['matches'][h] for v in audit['corpora'].values()):continue
        excluded.add(h);eligible.append(dict(c,numbers=c['atomic_numbers'],pool_index=row['pool_index']))
    strata=[]
    for lo,hi in [(17,28),(29,40)]:
        cases=[c for c in eligible if lo<=c['n_atoms']<=hi];cases.sort(key=lambda c:hashlib.sha256(('atomwise-connection-v1:'+c['composition_hex']).encode()).hexdigest());strata.append(cases)
    perbin=min(16,*[len(c) for c in strata]);assert perbin>=8,[(len(c)) for c in strata]
    test=sum([cases[:perbin] for cases in strata],[]);banks=[]
    for si in [0,1]:
        byarm={}
        for name in ['fm','gaga']:
            folder=root/f'runs/matched_connection_v1/validation/s{si}/{name}/teacher';f=folder/'bank.pt';done=json.loads((folder/'complete.json').read_text())
            assert done['complete'] and done['protocol_sha256']==sha(original) and sha(f)==done['bank_sha256']
            byarm[name]=dict(path=str(f.relative_to(root)),sha256=sha(f),teacher_folder=str(folder.relative_to(root)))
        banks.append(byarm)
    config=spec['head_configuration'];variants={
        'pair_global':dict(configuration=config,target='global',reuse=True),
        'atom_global':dict(configuration=dict(config,normalization='atomwise'),target='global',reuse=False),
        'pair_balanced':dict(configuration=config,target='balanced',reuse=False),
        'atom_balanced':dict(configuration=dict(config,normalization='atomwise'),target='balanced',reuse=False)}
    spec.update(format='atomwise_connection_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_protocol=str(original.relative_to(root)),original_protocol_sha256=sha(original),
        original_selection=str(selection.relative_to(root)),original_selection_sha256=sha(selection),reused_heads=old['heads'],banks=banks,variants=variants,
        force_floor_eV_A=1.,strengths=[0.,1.,2.,4.],strength_limit=4.,validation_rows=val,validation_samples=8,
        validation_parent_panel=str(panel.relative_to(root)),validation_parent_panel_sha256=sha(panel),test_rows=test,excluded_manifests=exclusions,
        validation_seeds=[62021,62022],evaluation_seeds=[62031,62032],
        target='Balanced mobility is positive per-atom1/sqrt(norm(force_i)^2+1^2), followed by the same zero-COM projection. Each balanced displacement has exactly the original global-force displacement norm; only its direction changes. Teacher labels are reconstructed from existing TRAIN force records, with no oracle calls or evaluation geometries.',
        selection='On16 development compositions select variant and strength separately for FM/GAGA, pooled across both fits: maximum all-attempt graph+GFN2 force<=5 yield, then graph yield, then smaller strength, then simpler variant in the declared order. Zero strength is generated once with pair_global and represents all zero-head models.',
        advance_gate='Advance only if selected FM is a new variant and beats its own validation-selected pair_global control by at least2pp joint yield with gains in both fits and no decrease in pooled graph rate. Otherwise stop this route without sampling its fresh panel.',
        confirmation='Before test generation freeze selected variants/strengths and each pair_global control. Test all four variants at the selected strength for each algorithm, plus its independently selected pair_global strength if different. Reuse identical variant/strength outputs rather than generating duplicates. Primary contrasts: selected FM versus best original pair_global and versus equally selected GAGA; ablations distinguish architecture and target changes.',
        fitting='Original global/pair head and all physical labels are reused. Three new heads per parent, same7106 parameters, same zero-output initialization, same seeds, same384 training-state sequence,20000 updates each;128 teacher-validation states only diagnose prediction. Total new240000 head updates, zero new pretrained-backbone updates or eSEN queries.',
        expected_new_validation_outputs=6656,expected_new_head_updates=240000,
        scope='Candidate atomwise message normalization and balanced mobility, tested as a2x2 factorial on frozen FM/GAGA parents. Standard equivariant operations; no new symmetry theorem. No supplied bond labels, test-time energy calls, optimization, equilibrium guarantee, or GAGA-superiority claim before confirmation. Prior failed static-tree/context/constraint pilots remain unchanged.')
    write(a.out,spec);print(json.dumps(dict(validation_compositions=len(val),fresh_compositions=len(test),variants=list(variants),new_validation_outputs=spec['expected_new_validation_outputs'])))


if __name__=='__main__':main()
