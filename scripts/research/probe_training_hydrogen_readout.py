"""TRAIN-only diagnostic of a simple detached-H geometric output map."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from scripts.research.run_gaga_feedback import atomic_save


def project_detached_hydrogens(x,numbers,contact=1.25,target=1.):
    z=torch.tensor(numbers,device=x.device);hydrogen=torch.where(z==1)[0];heavy=torch.where(z!=1)[0]
    y=x.clone()
    if not len(hydrogen) or not len(heavy):return y,torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device)
    r=covalent_radii(numbers,device=x.device,dtype=x.dtype);length=r[hydrogen,None]+r[heavy][None]
    offset=x[:,hydrogen,None]-x[:,None,heavy];distance=offset.norm(dim=-1);normalized=distance/length[None]
    nearest=normalized.argmin(-1);minimum=normalized.gather(-1,nearest[...,None])[...,0];repair=minimum>contact
    batch=torch.arange(len(x),device=x.device)[:,None];anchors=heavy[nearest];d=offset[batch,torch.arange(len(hydrogen),device=x.device)[None],nearest]
    proposed=x[batch,anchors]+target*length[None].expand(len(x),-1,-1).gather(-1,nearest[...,None])*d/d.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    y[:,hydrogen]=torch.where(repair[...,None],proposed,y[:,hydrogen]);y-=y.mean(1,keepdim=True)
    mask=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device);mask[:,hydrogen]=repair
    return y,mask


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project.resolve();torch.set_num_threads(2)
    assert not a.out.exists();a.out.mkdir(parents=True);proto=root/'research/evidence/matched_connection_v1.json';spec=json.loads(proto.read_text());ph=sha(proto);rows=[]
    for si in [0,1]:
        for name in ['fm','gaga']:
            folder=root/f'runs/matched_connection_v1/validation/s{si}/{name}/teacher';bank=torch.load(folder/'bank.pt',map_location='cpu',weights_only=False)
            assert bank['protocol_sha256']==ph
            for slot,index in enumerate(spec['training_rows']):
                file=folder/'records'/f'c{slot}.pt';r=torch.load(file,map_location='cpu',weights_only=False);digest=sha(file)
                assert r['training_row']==index and r['protocol_sha256']==ph and all(v['record_sha256']==digest for v in bank['rows'] if v['composition_slot']==slot)
                c=r['condition'];x=r['final_positions'].double();y,mask=project_detached_hydrogens(x,c['numbers'])
                before=assess(x,c,list(range(2)));after=assess(y,c,list(range(2)))
                # This is a diagnostic output map, not a trained model or quantum
                # relaxation. Original raw records and geometry tests are intact.
                for j,(old,new) in enumerate(zip(before['records'],after['records'])):
                    if old['graph_supported']:
                        assert not mask[j].any() and new['graph_supported']
                    if not mask[j].any():torch.testing.assert_close(y[j]-y[j].mean(0),x[j]-x[j].mean(0),atol=1e-12,rtol=0)
                saved=a.out/f'{name}_s{si}_c{slot}.pt';atomic_save(dict(condition=c,source_sha256=digest,original=x,projected=y,moved_hydrogen_mask=mask),saved)
                rows.append(dict(seed=si,family=name,composition_slot=slot,training_row=index,source_sha256=digest,file=saved.name,sha256=sha(saved),
                    before=before,after=after,moved_hydrogens=int(mask.sum()),changed_structures=int(mask.any(1).sum())))
    summary={}
    for name in ['fm','gaga']:
        v=[r for r in rows if r['family']==name]
        summary[name]=dict(attempted=sum(r['before']['attempted'] for r in v),graph_before=sum(r['before']['graph_supported'] for r in v),graph_after=sum(r['after']['graph_supported'] for r in v),
            geometry_before=sum(r['before']['geometrically_supported'] for r in v),geometry_after=sum(r['after']['geometrically_supported'] for r in v),
            graph_by_seed={str(si):[sum(r[k]['graph_supported'] for r in v if r['seed']==si) for k in ['before','after']] for si in [0,1]},
            moved_hydrogens=sum(r['moved_hydrogens'] for r in v),changed_structures=sum(r['changed_structures'] for r in v))
    write(a.out/'audit.json',dict(complete=True,protocol_sha256=ph,summary=summary,rows=rows,new_parent_trajectories=0,new_oracle_queries=0,new_optimizer_steps=0,
        derived_train_outputs=1024,scope='Diagnostic on cached TRAIN native parent outputs. Only hydrogens beyond1.25 summed covalent radii from every heavy atom are moved to1.0 radii from the closest heavy atom. Original graph-valid outputs are retained. This is explicitly a geometric output map, not a learned contribution, new original raw samples, chemical guarantee, or physical-quality evidence.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
