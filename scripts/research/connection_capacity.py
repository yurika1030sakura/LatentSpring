"""TRAIN-only capacity/optimization diagnosis; no fresh generation or oracle use."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import lsq_linear
from cfm_mol.physical_connection import make_physical_connection
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from scripts.research.run_gaga_feedback import atomic_save


def cache_rows(rows,config,device):
    lookup={z:i for i,z in enumerate(config['atomic_numbers'])};cache=[]
    for row in rows:
        n=len(row['x']);edges=torch.where(~torch.eye(n,dtype=torch.bool,device=device))
        cache.append((row['x'].to(device),row['endpoint'].to(device),torch.tensor([lookup[z] for z in row['condition']['numbers']],device=device),
            row['t'].reshape(1).to(device),torch.zeros(n,dtype=torch.long,device=device),*edges,row['force'].to(device)))
    return cache


def edge_basis(inputs,head):
    x,h,types,t,_,src,dst=inputs;keep=src<dst;i,j=src[keep],dst[keep]
    length=head.radii[types[i]]+head.radii[types[j]];r=x[i]-x[j];v=h[i]-h[j]
    rn=r/(r.square().sum(-1)+length.square()).sqrt()[:,None];hn=v/(v.square().sum(-1)+length.square()).sqrt()[:,None]
    weight=torch.sigmoid((1.25-(v.square().sum(-1)+1e-12).sqrt()/length)/.2)
    degree=x.new_zeros(len(x)).index_add(0,i,weight).index_add(0,j,weight)
    scale=head.velocity_scale*t.item()**head.gate_power/(2*degree.max().clamp_min(1.))
    rn=rn*weight[:,None]*scale;hn=hn*weight[:,None]*scale
    matrix=np.zeros((3*len(x),2*len(i)),dtype=np.float64)
    for k,(left,right) in enumerate(zip(i.tolist(),j.tolist())):
        block=torch.stack([rn[k],hn[k]],-1).cpu().double().numpy()
        matrix[3*left:3*left+3,2*k:2*k+2]=block;matrix[3*right:3*right+3,2*k:2*k+2]=-block
    return matrix


@torch.no_grad()
def evaluate(head,cache,indices):
    before=[];after=[];alignment=[]
    for index in indices:
        *inputs,target=cache[index];prediction=head(*inputs)
        before.append(float(target.square().sum(-1).mean()));after.append(float((prediction-target).square().sum(-1).mean()))
        alignment.append(float((prediction*target).sum()/(prediction.norm()*target.norm()).clamp_min(1e-12)))
    return dict(zero_mse=float(np.mean(before)),mse=float(np.mean(after)),fractional_reduction=float(1-np.mean(after)/np.mean(before)),mean_direction_cosine=float(np.mean(alignment)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);a=p.parse_args();torch.set_num_threads(2)
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen']
    bank=a.project/spec['bank'];assert sha(bank)==spec['bank_sha256'];data=torch.load(bank,map_location='cpu',weights_only=False)
    assert data['protocol_sha256']==spec['teacher_protocol_sha256'] and len(data['rows'])==512
    rows=data['rows'];train=spec['fit_states'];validation=spec['validation_states'];assert set(train).isdisjoint(validation) and sorted(train+validation)==list(range(512))
    assert not {rows[i]['condition']['composition_hex'] for i in train}&{rows[i]['condition']['composition_hex'] for i in validation}
    cache=cache_rows(rows,spec['pair_config'],'cuda');a.out.mkdir(parents=True,exist_ok=False);seed=spec['seeds'][a.seed_index]
    results={};checkpoints={};total_steps=0;diagnostic_forwards=0
    for kind in spec.get('variants',['pair','context']):
        torch.manual_seed(seed);config=spec[kind+'_config'];head=make_physical_connection(**config).cuda().float().train()
        optimizer=torch.optim.AdamW(head.parameters(),lr=spec['learning_rate'],weight_decay=1e-12);rng=np.random.default_rng(seed)
        tick=time.perf_counter();history=[];directory=a.out/kind;directory.mkdir()
        with (directory/'metrics.jsonl').open('w') as stream:
            for step in range(1,spec['steps']+1):
                index=train[int(rng.integers(len(train)))];*inputs,target=cache[index]
                prediction=head(*inputs);objective=(prediction-target).square().sum(-1).mean()
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite regression')
                optimizer.zero_grad(set_to_none=True);objective.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                stream.write(json.dumps(dict(step=step,bank_row=index,loss=float(objective.detach()),gradient_norm=float(norm)))+'\n')
                if step in spec['diagnostic_steps']:
                    record=dict(step=step,fit=evaluate(head,cache,train),validation=evaluate(head,cache,validation),seconds=time.perf_counter()-tick)
                    diagnostic_forwards+=len(cache);history.append(record);stream.flush();print(json.dumps(dict(seed=seed,kind=kind,**record)),flush=True)
                    checkpoint=directory/f'step_{step}.pt';atomic_save(dict(state_dict=head.state_dict(),configuration=config,protocol_sha256=ph,seed=seed,step=step),checkpoint)
                    checkpoints[f'{kind}/{step}']=sha(checkpoint)
        total_steps+=spec['steps'];results[kind]=dict(parameter_count=sum(p.numel() for p in head.parameters()),history=history,seconds=time.perf_counter()-tick)
    direct=[]
    if a.seed_index==0:
        # Independent per-state coefficients use the answer and are NOT a deployable model.
        head=make_physical_connection(**spec['pair_config']).double();cpu=cache_rows(rows,spec['pair_config'],'cpu')
        for index in spec['basis_probe_states']:
            *inputs,target=cpu[index];inputs=[v.double() if v.is_floating_point() else v for v in inputs];matrix=edge_basis(inputs,head)
            # Verify the explicit linear map against the production head.
            with torch.no_grad():head.pair_network[-1].weight.normal_(std=.1)
            logits=[];hook=head.pair_network.register_forward_hook(lambda module,args,output:logits.append(output.detach().clone()))
            with torch.no_grad():pred=head(*inputs)
            hook.remove();np.testing.assert_allclose(matrix@logits[0].tanh().numpy().ravel(),pred.numpy().ravel(),atol=1e-12,rtol=1e-10)
            y=target.double().numpy().ravel();sol=lsq_linear(matrix,y,bounds=(-1,1),method='trf',tol=1e-9,lsq_solver='exact',max_iter=300)
            after=float(np.mean((matrix@sol.x-y).reshape(-1,3)**2)*3);before=float(np.mean(y.reshape(-1,3)**2)*3)
            file=a.out/f'edge_coefficients_state{index}.npz';np.savez_compressed(file,coefficients=sol.x,target=y,prediction=matrix@sol.x)
            direct.append(dict(bank_row=index,n_atoms=len(target),zero_mse=before,mse=after,fractional_reduction=1-after/before,
                solver_status=int(sol.status),solver_optimality=float(sol.optimality),solver_success=bool(sol.success),iterations=sol.nit,file=file.name,sha256=sha(file)))
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,seed=seed,results=results,checkpoints=checkpoints,
        direct_edge_fits=direct,optimizer_steps=total_steps,diagnostic_head_forwards=diagnostic_forwards,new_neural_generation_outputs=0,
        new_oracle_queries=0,scope=spec['scope']))


if __name__=='__main__':main()
