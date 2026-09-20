"""Freeze a conditional H-flow candidate and its deterministic readout control."""
import argparse,copy,datetime,hashlib,json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project.resolve();assert not a.out.exists()
    origin=root/'research/evidence/atomwise_connection_v1.json';prior=json.loads(origin.read_text());datafile=root/prior['data'];assert sha(datafile)==prior['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False);assert len(data['training'])==20000
    assert all(1 in r['condition']['numbers'] and any(z!=1 for z in r['condition']['numbers']) for r in data['training']+data['validation'])
    network=copy.deepcopy(prior['parents'][0]['gaga']['spec']);network['upstream_args'].update(nf=64,n_layers=4)
    network['initialization_seed']=63011;model=base.initialize(network,'cpu');parameters=sum(p.numel() for p in model.parameters())
    excluded={r['condition']['composition_hex'] for r in data['training']+data['validation']};exclusions={}
    names=[('wide_generalization_panel_v1',['rows']),('gaga_feedback_panel_v1',['validation_rows','test_rows']),
        ('physical_strength_calibration_v1',['validation_rows','test_rows']),('broad_physical_endpoints_v1',['test_rows']),
        ('physical_connection_v1',['test_rows']),('trajectory_connection_v1',['test_rows']),('context_confirmation_v1',['test_rows']),
        ('matched_connection_v1',['validation_rows','test_rows']),('connection_tradeoff_v1',['validation_rows','test_rows']),
        ('atomwise_confirmation_v1',['validation_rows','test_rows'])]
    for name,keys in names:
        file=root/f'research/evidence/{name}.json';s=json.loads(file.read_text());exclusions[str(file.relative_to(root))]=sha(file)
        for key in keys:excluded.update(c['composition_hex'] for c in s[key])
    poolfile=root/prior['source_pool'];auditfile=root/prior['qualification_audit'];assert sha(poolfile)==prior['source_pool_sha256'] and sha(auditfile)==prior['qualification_audit_sha256']
    pool=json.loads(poolfile.read_text());audit=json.loads(auditfile.read_text());eligible=[]
    for row in audit['decisions']:
        if not row['qualified']:continue
        c=pool['rows'][row['pool_index']]['condition'];h=c['composition_hex']
        if h in excluded or c['charge']!=0 or c['spin_multiplicity']!=1 or not set(c['atomic_numbers'])<=set(network['atomic_numbers']):continue
        if any(v['matches'][h] for v in audit['corpora'].values()):continue
        excluded.add(h);eligible.append(dict(c,numbers=c['atomic_numbers'],pool_index=row['pool_index']))
    strata=[]
    for lo,hi in [(17,28),(29,40)]:
        values=[c for c in eligible if lo<=c['n_atoms']<=hi];values.sort(key=lambda c:hashlib.sha256(('hydrogen-completion-v1:'+c['composition_hex']).encode()).hexdigest());strata.append(values)
    size=min(16,*[len(v) for v in strata]);assert size>=8,[len(v) for v in strata];fresh=sum([v[:size] for v in strata],[])
    sources=[];auditpath=root/'research/evidence/atomwise_connection_selection_v1.json';parentaudit=json.loads(auditpath.read_text());assert parentaudit['complete']
    for si in [0,1]:
        pair={}
        for name in ['fm','gaga']:
            folder=root/f'runs/atomwise_connection_v1/validation/s{si}/{name}/pair_global/generation';file=folder/'generation.json';assert sha(file)==parentaudit['validation_provenance'][str(file.relative_to(root))]
            physical=root/f'runs/atomwise_connection_v1/validation/s{si}/xtb/results.json';assert sha(physical)==parentaudit['validation_provenance'][str(physical.relative_to(root))]
            pair[name]=dict(generation=str(file.relative_to(root)),generation_sha256=sha(file),method=name+'_pair_global_a3',
                physical=str(physical.relative_to(root)),physical_sha256=sha(physical),condition_count=16,samples_per_condition=8)
        sources.append(pair)
    diagnostic=sorted(range(len(data['validation'])),key=lambda i:hashlib.sha256(('hydrogen-loss-v1:'+str(i)).encode()).hexdigest())[:128]
    spec=dict(format='hydrogen_completion_v1',frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        data=prior['data'],data_sha256=prior['data_sha256'],network_spec=network,parameter_count=parameters,model_max_atoms=200,
        seeds=[63011,63012],batch_seeds=[63021,63022],noise_seeds=[63031,63032],diagnostic_seed=63041,
        steps=10000,batch_size=32,learning_rate=.0003,ema_decay=.999,checkpoint_every=2000,diagnostic_steps=[2000,10000],
        reference_validation_rows=diagnostic,noise_std_A=.4,coordinate_objective='Hydrogen-only conditional flow matching, with identical-H assignment and a mixture of full/partial H corruptions. Heavy relative coordinates are conditioned and held fixed. No supplied or inferred bond targets.',
        validation_rows=prior['validation_rows'],validation_sources=sources,validation_parent_audit=str(auditpath.relative_to(root)),validation_parent_audit_sha256=sha(auditpath),
        test_rows=fresh,excluded_manifests=exclusions,source_pool=prior['source_pool'],source_pool_sha256=prior['source_pool_sha256'],
        qualification_audit=prior['qualification_audit'],qualification_audit_sha256=prior['qualification_audit_sha256'],
        parents=prior['parents'],physical_heads=prior['reused_heads'],parent_strength=4.,backbone_calls=128,evaluation_batch=8,test_samples=16,evaluation_seeds=[63051,63052],
        decoder_steps=4,velocity_cap_A=2.,methods=['base','radial','atom_start0','atom_start_half','molecule_start0','molecule_start_half'],
        radial_contact=1.25,radial_target=1.,xtb_binary=prior['xtb_binary'],xtb_binary_sha256=prior['xtb_binary_sha256'],
        selection='Each family selects one readout pooled across two fits by joint graph+GFN2-force<=5 yield, then graph yield, then simpler readout in the declared method order. Same two H decoders and all choices are available to FM and GAGA. Fixed EMA10000 checkpoints; no best-checkpoint selection.',
        advance_gate='A learned FM readout must improve joint yield in both fits over both its unmodified parent and the fixed radial control, without a pooled graph-rate decrease versus either. Otherwise stop the learned-decoder route and preserve the deterministic baseline as a diagnostic. Fresh-test statistical claims still require paired confidence intervals.',
        primary='Selected FM readout versus parent and fixed geometric control, and versus equally selected GAGA. Report all readouts, physical energy/force, coordinate changes, decoder calls and time. Do not count recycled parent trajectories or unchanged physical measurements twice.',
        literature=dict(quetzal='https://arxiv.org/abs/2505.13791',note='Hydrogen decoration and conditional molecular generation are established tasks. This candidate claims neither first hydrogen completion nor an exact target distribution for arbitrary parent errors.'),
        scope='Conditional coordinate completion candidate, applied only to initially detached H or all H in molecules containing one. Already accepted parent coordinates are preserved and checked. A learned readout must beat the explicit fixed geometric map. No energy queries or force-field optimization in generation. No evidence of utility or additional AI novelty before held-out confirmation.')
    write(a.out,spec);print(json.dumps(dict(parameters=parameters,training_rows=20000,validation_compositions=16,fresh_compositions=len(fresh))))


if __name__=='__main__':main()
