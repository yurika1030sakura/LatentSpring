"""Freeze equal-budget strength calibration; old test outcomes are not selection data."""
import argparse,datetime,hashlib,json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project
    assert not a.out.exists();oldfile=root/'research/evidence/matched_connection_v1.json';oldselection=root/'research/evidence/matched_connection_selection_v1.json'
    spec=json.loads(oldfile.read_text());selection=json.loads(oldselection.read_text());assert selection['complete'] and selection['protocol_sha256']==sha(oldfile)
    for file,digest in selection['validation_provenance'].items():assert sha(root/file)==digest
    datafile=root/spec['data'];assert sha(datafile)==spec['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False)
    excluded={r['condition']['composition_hex'] for r in data['training']+data['validation']};exclusions={}
    for name,keys in [('wide_generalization_panel_v1',['rows']),('gaga_feedback_panel_v1',['validation_rows','test_rows']),
            ('physical_strength_calibration_v1',['validation_rows','test_rows']),('broad_physical_endpoints_v1',['test_rows']),
            ('physical_connection_v1',['test_rows']),('trajectory_connection_v1',['test_rows']),('context_confirmation_v1',['test_rows']),
            ('matched_connection_v1',['validation_rows','test_rows'])]:
        file=root/f'research/evidence/{name}.json';s=json.loads(file.read_text());exclusions[str(file.relative_to(root))]=sha(file)
        for key in keys:excluded.update(c['composition_hex'] for c in s[key])
    poolfile=root/spec['source_pool'];auditfile=root/spec['qualification_audit'];assert sha(poolfile)==spec['source_pool_sha256'] and sha(auditfile)==spec['qualification_audit_sha256']
    pool=json.loads(poolfile.read_text());audit=json.loads(auditfile.read_text());eligible=[]
    for row in audit['decisions']:
        if not row['qualified']:continue
        c=pool['rows'][row['pool_index']]['condition'];h=c['composition_hex']
        if h in excluded or c['charge']!=0 or c['spin_multiplicity']!=1 or not set(c['atomic_numbers'])<=set(spec['head_configuration']['atomic_numbers']):continue
        if any(corpus['matches'][h] for corpus in audit['corpora'].values()):continue
        excluded.add(h);eligible.append(dict(c,numbers=c['atomic_numbers'],pool_index=row['pool_index']))
    fresh=[]
    for lo,hi in [(17,28),(29,40)]:
        candidates=[c for c in eligible if lo<=c['n_atoms']<=hi]
        candidates.sort(key=lambda c:hashlib.sha256(('extended-connection-v1:'+c['composition_hex']).encode()).hexdigest())
        assert len(candidates)>=16,(lo,hi,len(candidates));fresh.extend(candidates[:16])
    spec.update(format='extended_connection_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_protocol=str(oldfile.relative_to(root)),original_protocol_sha256=sha(oldfile),
        original_selection=str(oldselection.relative_to(root)),original_selection_sha256=sha(oldselection),
        heads=selection['heads'],strength_limit=4.,new_strengths=[2.,4.],all_strengths=[0.,.25,.5,1.,2.,4.],test_rows=fresh,
        excluded_manifests=exclusions,evaluation_seeds=[61031,61032],
        selection='Reuse the audited eight-composition validation panel and all four previous strengths; generate only strengths2 and4 with identical original validation RNG seeds. For each algorithm maximize pooled all-attempt graph+GFN2 force<=5 yield among strengths whose pooled graph rate is no more than1pp below strength1 and each-fit graph rate no more than3pp below strength1. Ties prefer smaller strength.',
        advance_gate='Advance to fresh confirmation only if selected FM gains at least2pp joint yield over strength1, with positive changes in both fits and the graph guard satisfied. Apply the same candidate strengths and selection rules to GAGA. Failure ends this calibration route without querying the fresh panel.',
        primary='On32 fresh compositions compare selected FM versus its strength1 model, and selected FM versus equally calibrated GAGA. Report both fits, graph rates, force-threshold curves, diversity, and generation timers with sampling-call counts. Every generated attempt remains in the denominator.',
        provenance_scope='The previous16-composition test has been read and is excluded. This is a follow-up motivated by the validation optimum hitting the previous upper bound; it is not a prespecified part of the original confirmation.',
        new_fit_outputs=0,new_esen_queries=0,new_optimizer_steps=0,expected_new_validation_outputs=1024,expected_fresh_outputs_if_advanced=4096,
        scope='Identical frozen EGNN parents, trained heads and physical preparation as matched_connection_v1. This is bounded validation-only inference calibration, not an additional AI architecture contribution. No energy query or geometry optimizer in the sampler. GAGA native noise schedule and observation noise remain intact. The per-atom velocity bound scales to2*strength*progress^2; no unchanged unit-strength or final-geometry guarantee.')
    write(a.out,spec)


if __name__=='__main__':main()
