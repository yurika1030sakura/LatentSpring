"""Select update strength on validation, then compare on a new frozen panel."""
import argparse
import copy
import datetime
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.run_matched_physical import make_model,make_context
from scripts.research.run_gaga_feedback import evaluate,atomic_save
from scripts.research.confirm_gaga_feedback import audit_report,bootstrap
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def freeze(root,path):
    assert not path.exists()
    panelpath=root/'research/evidence/gaga_feedback_panel_v1.json'
    panel=json.loads(panelpath.read_text())
    pool=json.loads((root/panel['source_pool']).read_text())
    audit=json.loads((root/panel['qualification_audit']).read_text())
    assert sha(root/panel['source_pool'])==panel['source_pool_sha256']
    assert sha(root/panel['qualification_audit'])==panel['qualification_audit_sha256']
    original=json.loads((root/'research/evidence/matched_physical_s0_v2.json').read_text())
    data=torch.load(root/original['data'],map_location='cpu',weights_only=False)
    excluded={r['condition']['composition_hex'] for key in ['training','validation'] for r in data[key]}
    excluded|={c['composition_hex'] for key in ['validation_rows','test_rows'] for c in panel[key]}
    excluded|={r['composition_hex'] for r in json.loads((root/panel['excluded_old_panel']).read_text())['rows']}
    eligible=[]
    for d in audit['decisions']:
        if not d['qualified']:continue
        c=pool['rows'][d['pool_index']]['condition']
        if c['composition_hex'] in excluded:continue
        assert all(not corpus['matches'][c['composition_hex']] for corpus in audit['corpora'].values())
        eligible.append(dict(c,pool_index=d['pool_index']))
    test=[]
    for lo,hi in [(17,28),(29,40),(41,52),(53,64)]:
        choices=[c for c in eligible if lo<=c['n_atoms']<=hi]
        choices.sort(key=lambda c:hashlib.sha256(('physical-strength-confirmation-v1:'+c['composition_hex']).encode()).hexdigest())
        assert len(choices)>=4,(lo,hi,len(choices))
        test.extend(choices[:4])
    inputs={};arms={}
    for seed in [0,1]:
        p=root/f'research/evidence/matched_physical_s{seed}_v2.json';s=json.loads(p.read_text())
        inputs[str(p.relative_to(root))]=sha(p);arms[str(seed)]=copy.deepcopy(s['arms'])
        for method,arm in arms[str(seed)].items():
            for role in ['physical','replay']:
                name=f'runs/matched_physical_v2/s{seed}/study/{method}/{role}/last.ckpt'
                arm[role+'_checkpoint']=name;inputs[name]=sha(root/name)
            inputs[arm['checkpoint']]=arm['checkpoint_sha256']
    write(path,dict(frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validation_rows=panel['validation_rows'],test_rows=test,arms=arms,inputs=inputs,
        alpha=[0.,.0625,.125,.25,.5,1.],seeds=[0,1],validation_samples=16,test_samples=64,
        validation_seed=43001,test_seed=49101,calls_per_sample=128,training_steps=0,
        selection='Per family, one alpha shared by both seeds. Maximize pooled eSEN graph-valid force<=5 yield among candidates whose graph rate in each seed is within2pp of alpha0 or better. Ties favor smaller alpha. alpha0 is always eligible.',
        primary_gate='Selected FM minus selected GAGA: positive joint-yield difference in each seed and positive lower composition-bootstrap95 bound. eSEN is provisional; independent GFN2 needed for a two-potential claim.',
        oracle={k:original[k] for k in ['oracle_interpreter','oracle_checkpoint','oracle_sha256','oracle_worker_sha256']},
        parent_panel_sha256=sha(panelpath),source_pool=panel['source_pool'],source_pool_sha256=panel['source_pool_sha256'],
        new_panel_rule='Four eligible compositions per fixed size bin by salted hash, disjoint from training, internal validation, previous64 and previous32 model panels; no model outcomes used.',
        scope='Validation-based calibration following a failed fixed-strength transfer. New confirmation outcomes are not used to select alpha. Same six choices for each family. Raw coordinates, no optimization or selection of outputs.'))


def score(report,directory,label,oracle):
    destination=directory/(label+'_quality.json')
    if destination.exists():
        result=json.loads(destination.read_text());assert result['report_sha256']==sha(directory/(label+'_results.json'))
        assert sha(directory/result['arrays'])==result['arrays_sha256'];return result
    graph=[];joint=[];artifacts=[];queries=oracle.evaluated
    for i,row in enumerate(report['rows']):
        file=directory/f'{label}_c{i}.pt';assert sha(file)==row['sample_sha256']
        raw=torch.load(file,map_location='cpu',weights_only=False)
        x=raw['positions'];valid=torch.tensor([r['graph_supported'] for r in row['records']])
        c=row['condition'];oracle.condition=dict(numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
        force=torch.full((len(x),),float('inf'),dtype=torch.float64)
        idx=torch.where(valid)[0];artifact=directory/f'{label}_physical_c{i}.pt'
        if len(idx):
            selected=x[idx];n=len(selected)
            e,f=oracle.evaluate_chunked(torch.cat([selected,-selected]),max_request=16)
            force[idx]=((f[:n]-f[n:])/2).square().sum(-1).mean(-1).sqrt()
            atomic_save(dict(raw_indices=idx,raw_energy_eV=e,raw_force_eV_A=f,
                source_sha256=sha(file),force_rms=force),artifact)
        else:atomic_save(dict(raw_indices=idx,source_sha256=sha(file),force_rms=force),artifact)
        graph.append(valid.numpy());joint.append((valid & (force<=5.)).numpy());artifacts.append(dict(path=artifact.name,sha256=sha(artifact)))
    arrays=directory/(label+'_quality.npz');np.savez_compressed(arrays,graph=np.stack(graph),joint=np.stack(joint))
    result=dict(complete=True,report_sha256=sha(directory/(label+'_results.json')),
        arrays=arrays.name,arrays_sha256=sha(arrays),artifacts=artifacts,
        graph=float(np.mean(graph)),joint=float(np.mean(joint)),queries=oracle.evaluated-queries,
        all_attempt_denominator=True,physical_queries_on_graph_valid_only=True)
    write(destination,result);return result


def merged_model(arm,alpha):
    parent=torch.load(arm['checkpoint'],map_location='cpu',weights_only=False)['ema_state_dict']
    model=make_model(arm,parent);names={n for n,p in model.named_parameters() if p.requires_grad}
    phys=torch.load(arm['physical_checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    replay=torch.load(arm['replay_checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    for n in parent:
        if n in names:parent[n]=parent[n]+alpha*(phys[n]-replay[n])
        else:assert torch.equal(parent[n],phys[n]) and torch.equal(parent[n],replay[n]),n
    model.load_state_dict(parent,strict=True)
    return model


def run(root,path,out):
    spec=json.loads(path.read_text());assert spec['frozen'];torch.set_num_threads(2)
    for p,h in spec['inputs'].items():assert sha(root/p)==h,p
    oracle_spec=spec['oracle'];assert sha(Path(oracle_spec['oracle_checkpoint']))==oracle_spec['oracle_sha256']
    worker=Path(__file__).resolve().parent/'oracle_worker.py';assert sha(worker)==oracle_spec['oracle_worker_sha256']
    out.mkdir(parents=True,exist_ok=True);source=base.HarmonicSource();all_scores={}
    for stage in ['validation','test']:
        if stage=='test':
            selection={}
            for method in ['distance','gaga']:
                eligible=[]
                for alpha in spec['alpha']:
                    scores=[all_scores[f'{seed}/{method}/{alpha}'] for seed in spec['seeds']]
                    if all(scores[j]['graph']>=all_scores[f'{seed}/{method}/0.0']['graph']-.02-1e-12 for j,seed in enumerate(spec['seeds'])):
                        eligible.append((np.mean([s['joint'] for s in scores]),-alpha))
                selection[method]=-max(eligible)[1]
            receipt=dict(complete=True,alpha=selection,validation=all_scores,protocol_sha256=sha(path),test_outcomes_used=False)
            if (out/'selection.json').exists():assert json.loads((out/'selection.json').read_text())==receipt
            else:write(out/'selection.json',receipt)
            print(json.dumps(dict(selection=selection)),flush=True)
        rows=spec[stage+'_rows'];c=rows[0]
        with EnergyOracle(oracle_spec['oracle_interpreter'],worker,oracle_spec['oracle_checkpoint'],
                numbers=c['atomic_numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],
                device='cuda',batch_size=8,timeout_seconds=180.) as oracle:
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
            for seed in spec['seeds']:
                for method,arm0 in spec['arms'][str(seed)].items():
                    arm=copy.deepcopy(arm0)
                    for key in ['checkpoint','physical_checkpoint','replay_checkpoint']:arm[key]=root/arm[key]
                    for alpha in (spec['alpha'] if stage=='validation' else sorted(set([0.,selection[method]]))):
                        label=method+'_a'+str(alpha).replace('.','p');directory=out/stage/f's{seed}'
                        model=merged_model(arm,alpha);context=make_context(arm,source)
                        report=evaluate(model,source,arm['spec'],context,rows,spec[stage+'_seed']+seed,spec[stage+'_samples'],directory,label)
                        del model,context;gc.collect();torch.cuda.empty_cache()
                        # Replay every saved graph assay before using it for selection.
                        audit_report(directory/(label+'_results.json'),rows,spec[stage+'_samples'])
                        quality=score(report,directory,label,oracle)
                        if stage=='validation':all_scores[f'{seed}/{method}/{alpha}']=quality
    effects={};matrices={}
    for method in ['distance','gaga']:
        for role,alpha in [('selected',selection[method]),('base',0.)]:
            values=[]
            for seed in spec['seeds']:
                label=method+'_a'+str(float(alpha)).replace('.','p')
                with np.load(out/'test'/f's{seed}'/(label+'_quality.npz')) as a:values.append(a['joint'].mean(-1))
            matrices[method+'_'+role]=np.array(values)
    rng=np.random.default_rng(49123)
    for label,left,right in [('selected_fm_minus_selected_gaga','distance_selected','gaga_selected'),
            ('physical_in_fm','distance_selected','distance_base'),('physical_in_gaga','gaga_selected','gaga_base')]:
        effects[label]=bootstrap(matrices[left]-matrices[right],rng,20000)
    primary=effects['selected_fm_minus_selected_gaga']
    write(out/'audit.json',dict(complete=True,protocol_sha256=sha(path),selection_sha256=sha(out/'selection.json'),
        primary_esen_gate=bool(min(primary['by_seed'])>0 and primary['ci95'][0]>0),contrasts=effects,
        selected_alpha=selection,confirmation_joint_yields={k:dict(mean=float(v.mean()),by_seed=v.mean(-1).tolist()) for k,v in matrices.items()},
        independent_xtb_complete=False,scope=spec['scope'],reserved_outcomes_queried=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--out',type=Path);parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze(args.project,args.protocol)
    else:
        assert args.out is not None
        run(args.project,args.protocol,args.out)
