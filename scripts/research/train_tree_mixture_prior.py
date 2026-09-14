#!/usr/bin/env python3
"""Fit label-free latent-tree priors on a fixed subset of actual OMol25 training data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from ase.data import atomic_numbers
from flowmol.model_utils.load import read_config_file
from flowmol.data_processing.dataset import MoleculeDataset
from cfm_mol.electronic_metadata import ElectronicMetadata
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def load_training(project,protocol):
    config=project/protocol['config'];assert sha(config)==protocol['config_sha256']
    cfg=read_config_file(config);cfg['mol_fm'].pop('bgfm',None)
    assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
    cfg['mol_fm']['prior_config']['x']['align']=False
    ds=MoleculeDataset('train',dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
        explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False)),prior_config=cfg['mol_fm']['prior_config'])
    metadata=ElectronicMetadata(project/protocol['metadata'],'train',[atomic_numbers[s] for s in cfg['dataset']['atom_map']])
    metadata.verify_processed_file(ds.processed_data_dir/'train_data_processed.pt')
    assert len(ds)==len(metadata.indices)
    order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(protocol['data_seed']))
    return cfg,ds,metadata,order


def row_from_dataset(ds,metadata,index):
    g=ds[index];metadata.attach(g,[index],1.)
    x=g.ndata['x_1_true'].double();x=x-x.mean(0)
    numbers=metadata.atomic_numbers[g.ndata['a_1_true'].argmax(-1)].tolist()
    state=g.ndata['electronic_state'][0]
    return dict(positions=x,numbers=numbers,charge=int(state[0]),spin=int(state[1]),processed_index=index)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());assert spec['frozen'];torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    cfg,ds,metadata,order=load_training(a.project,spec)
    train=[row_from_dataset(ds,metadata,int(i)) for i in order[:spec['prior_steps']]]
    held=[row_from_dataset(ds,metadata,int(i)) for i in order[spec['fm_steps']:spec['fm_steps']+spec['prior_validation_rows']]]
    del ds
    torch.save(dict(training=train,validation=held),a.out/'data.pt')
    report=dict(complete=False,protocol_sha256=sha(a.protocol),data_sha256=sha(a.out/'data.pt'),
        data_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),metadata_progress_sha256=metadata.progress_sha256,
        training_rows=len(train),validation_rows=len(held),new_molecular_oracle_calls=0,models={})
    write(a.out/'results.json',report)
    fixed=TreeMixturePrior(mode='fixed',width=spec['edge_log_width']).double()
    with torch.no_grad():
        report['fixed_validation_nll_per_dof']=[float(-fixed.log_prob(r['positions'],r['numbers'],r['charge'],r['spin'])/max(1,3*(len(r['numbers'])-1))) for r in held]
    for mode in ['node','pair']:
        torch.manual_seed(spec['prior_seed'])
        model=TreeMixturePrior(mode=mode,width=spec['edge_log_width']).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=spec['prior_lr'])
        metrics=[];start=time.perf_counter()
        for step,r in enumerate(train,1):
            loss=-model.log_prob(r['positions'],r['numbers'],r['charge'],r['spin'])/max(1,3*(len(r['numbers'])-1))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite prior likelihood')
            if loss.requires_grad:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            else:norm=loss.new_zeros(())
            row=dict(step=step,processed_index=r['processed_index'],n_atoms=len(r['numbers']),nll_per_dof=float(loss),gradient_norm=float(norm))
            metrics.append(row)
            with (a.out/f'{mode}_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%100==0 or step==1:print(json.dumps(dict(mode=mode,seconds=time.perf_counter()-start,**row)),flush=True)
        model.eval();model.requires_grad_(False)
        with torch.no_grad():
            values=[float(-model.log_prob(r['positions'],r['numbers'],r['charge'],r['spin'])/max(1,3*(len(r['numbers'])-1))) for r in held]
        checkpoint=a.out/f'{mode}.pt'
        torch.save(dict(configuration=model.configuration,state_dict=model.state_dict(),protocol_sha256=sha(a.protocol),data_sha256=report['data_sha256']),checkpoint)
        write(a.out/f'{mode}_metrics.json',metrics)
        report['models'][mode]=dict(checkpoint_sha256=sha(checkpoint),parameters=sum(p.numel() for p in model.parameters()),
            validation_nll_per_dof=values,seconds=time.perf_counter()-start,metrics_sha256=sha(a.out/f'{mode}_metrics.json'))
        write(a.out/'results.json',report)
    report['complete']=True;write(a.out/'results.json',report)


if __name__=='__main__':main()
