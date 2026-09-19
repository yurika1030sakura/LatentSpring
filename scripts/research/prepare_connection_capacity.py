"""Freeze a composition-separated, TRAIN-only correction-capacity diagnosis."""
import argparse,datetime,hashlib,json
from pathlib import Path
import torch
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();prior=a.project/'research/evidence/trajectory_connection_v1.json';old=json.loads(prior.read_text())
    bank=a.project/old['trajectory_bank'];data=torch.load(bank,map_location='cpu',weights_only=False);assert data['protocol_sha256']==sha(prior)
    rows=data['rows'];compositions={r['composition_slot']:r['condition'] for r in rows};validation=set();probes=[]
    for lo,hi in [(8,16),(17,24),(25,32),(33,40)]:
        eligible=[i for i,c in compositions.items() if lo<=c['n_atoms']<=hi];assert len(eligible)==32
        eligible.sort(key=lambda i:hashlib.sha256(('capacity-holdout-v1:'+compositions[i]['composition_hex']).encode()).hexdigest())
        validation.update(eligible[:8]);probes.extend(4*i+2 for i in eligible[8:10])
    pair=old['physical_connection'];context=dict(pair,context_layers=2,context_width=32)
    spec=dict(format='connection_capacity_v1',frozen=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bank=old['trajectory_bank'],bank_sha256=sha(bank),
        teacher_protocol_sha256=sha(prior),pair_config=pair,context_config=context,seeds=[56001,56002],steps=20000,diagnostic_steps=[2000,10000,20000],learning_rate=.0003,
        fit_states=[i for i,r in enumerate(rows) if r['composition_slot'] not in validation],validation_states=[i for i,r in enumerate(rows) if r['composition_slot'] in validation],basis_probe_states=probes,
        decision='Compare final20000-step models on internal held-out-composition force-target MSE, with both seeds retained. Earlier checkpoints diagnose optimization; no final generator advantage is inferred. The next model must meaningfully improve this diagnostic before spending on another fresh generator evaluation.',
        scope='Capacity diagnosis within the existing TRAIN teacher bank:96 compositions fit and32 held out from head training, four states each. The pretrained parent has seen TRAIN compositions; this is not external generator confirmation. Direct edge-coefficient fits use each target and cannot be deployed. No new oracle calls, no new generated outputs, no GAGA superiority follows.')
    write(a.out,spec);print(sha(a.out))


if __name__=='__main__':main()
