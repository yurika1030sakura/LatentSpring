"""Freeze the parent-trajectory force/work correction experiment before fitting."""
import argparse,datetime,hashlib,json
from pathlib import Path
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();root=a.project
    previous=root/'research/evidence/physical_connection_v1.json';s=json.loads(previous.read_text())
    excluded=set()
    for name,keys in [('wide_generalization_panel_v1',['rows']),('gaga_feedback_panel_v1',['validation_rows','test_rows']),('physical_strength_calibration_v1',['validation_rows','test_rows']),('broad_physical_endpoints_v1',['test_rows']),('physical_connection_v1',['test_rows'])]:
        d=json.loads((root/f'research/evidence/{name}.json').read_text())
        for key in keys:excluded|={c['composition_hex'] for c in d[key]}
    poolpath=root/s['source_pool'];auditpath=root/s['qualification_audit']
    assert sha(poolpath)==s['source_pool_sha256'] and sha(auditpath)==s['qualification_audit_sha256']
    pool=json.loads(poolpath.read_text());audit=json.loads(auditpath.read_text());eligible=[]
    for decision in audit['decisions']:
        if not decision['qualified']:continue
        c=pool['rows'][decision['pool_index']]['condition']
        if c['composition_hex'] in excluded:continue
        assert all(not corpus['matches'][c['composition_hex']] for corpus in audit['corpora'].values())
        eligible.append(dict(c,pool_index=decision['pool_index']))
    test=[]
    for lo,hi in [(17,28),(29,40)]:
        candidates=[c for c in eligible if lo<=c['n_atoms']<=hi]
        candidates.sort(key=lambda c:hashlib.sha256(('trajectory-connection-v1:'+c['composition_hex']).encode()).hexdigest());assert len(candidates)>=8
        test.extend(candidates[:8])
    s['physical_connection']['velocity_scale']=2.
    s.update(format='trajectory_connection_v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_protocol=str(previous.relative_to(root)),previous_protocol_sha256=sha(previous),
        training_seeds=[54001,54002],evaluation_seeds=[54051,54052],test_rows=test,
        methods=['base','connection_physical','connection_force','connection_work'],connection_init_seed_offset=800000,
        trajectory_bank='runs/trajectory_connection_v1/teacher/bank.pt',
        trajectory=dict(seed=54101,draws_per_composition=2,capture_steps=[23,29],
            kT=0.0258519998,particles=8,max_sigma=.03,max_shift=.1,radius=1.),
        primary='Trajectory-force head minus parent on all-attempt graph-valid GFN2 force<=5 yield. The work-minus-force contrast separately tests the work-weight contribution. Report both seeds and all controls.',
        training_budget='Each head:2000 updates and8178 trainable parameters. FM control uses4000 backbone forward examples. Cached-trajectory heads use2000 head-only forwards each and share256 new TRAIN parent rollouts plus9216 eSEN queries. Preparation is counted; no matched total-compute or wall-time claim.',
        architecture='Same symmetric8178-parameter pair head in all arms, with2*t^2 per-atom instantaneous velocity bound. Every parent tensor is frozen.',
        target='FM arm uses the reused128 empirical physically corrected references. Trajectory-force/work arms use512 provisional endpoints from256 frozen-parent TRAIN trajectories, with no chemical validity rejection. The work target is the self-normalized finite-particle mean of a local Gaussian-restrained eSEN energy tilt inside a1-Angstrom Frobenius ball; zero-support states fall back to force and are retained. This is not global Boltzmann learning.',
        scope='Frozen-parent physical residual candidate. Actual parent-generated states, physical supervision and complete local work weights are tested separately. No proven global thermodynamic law for the finite trained generator, no test-time energy optimization, no GAGA-superiority claim.')
    write(a.out,s);print(sha(a.out))


if __name__=='__main__':main()
