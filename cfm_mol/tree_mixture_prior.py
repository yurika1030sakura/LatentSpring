"""Normalized COM coordinate priors from latent spanning-tree mixtures.

Trees are auxiliary dependence structures, never observed molecular bonds.
Each tree edge has a normalized isotropic lognormal radial displacement law.
The marginal coordinate density sums all trees by the matrix-tree theorem.
Tree ensembles and harmonic priors have prior art; molecular utility is unproven.
"""
import math
import numpy as np
import torch
from torch import nn
from rdkit import Chem


def log_tree_partition(log_weights):
    """Log spanning-tree partition via positive star-mesh elimination.

    Equivalent to a grounded Laplacian determinant, without cancellation or
    exponentiating large negative edge log weights. No ODE/trace estimator.
    """
    if log_weights.ndim!=2 or log_weights.shape[0]!=log_weights.shape[1]:
        raise ValueError('Square symmetric log-edge matrix required')
    n=len(log_weights)
    if n<2:return log_weights.new_zeros(())
    if not torch.allclose(log_weights,log_weights.T,atol=1e-8,rtol=1e-8):
        raise ValueError('Undirected tree weights must be symmetric')
    w=log_weights.masked_fill(torch.eye(n,dtype=torch.bool,device=log_weights.device),-torch.inf)
    total=w.new_zeros(())
    # Leave the final vertex as root. Each eliminated degree is a Schur pivot.
    for remaining in range(n,1,-1):
        edge=w[0,1:];degree=torch.logsumexp(edge,0)
        if not torch.isfinite(degree):raise ValueError('Tree partition requires connected finite support')
        total=total+degree
        w=torch.logaddexp(w[1:,1:],edge[:,None]+edge[None,:]-degree)
        w=w.masked_fill(torch.eye(remaining-1,dtype=torch.bool,device=w.device),-torch.inf)
    return total


def radial_log_density(distance, length, width=.2):
    """R^3 density: lognormal radius with mean length, uniform solid angle."""
    if not 0<width<2 or (distance<=0).any():
        raise ValueError('Positive noncoincident distances and width in(0,2) required')
    log_r=distance.log();mu=length.log()-.5*width**2
    return -.5*((log_r-mu)/width).square()-3*log_r-math.log(4*math.pi*width*math.sqrt(2*math.pi))


def tree_mixture_log_prob(x, log_weights, edge_lengths, width=.2):
    """Normalized density against intrinsic unweighted COM Lebesgue measure.

    q(x)=N^(3/2) tau(a_ij h_ij(x_i-x_j))/tau(a_ij). Exact collisions
    are outside this implementation's differentiable evaluation domain.
    """
    n=len(x)
    if x.shape!=(n,3) or log_weights.shape!=(n,n) or edge_lengths.shape!=(n,n):
        raise ValueError('Coordinate/edge shape mismatch')
    if not torch.isfinite(x).all() or float(x.mean(0).abs().max())>1e-5:
        raise ValueError('Finite COM-centered coordinates required')
    if n<2:return x.new_zeros(())
    i,j=torch.triu_indices(n,n,1,device=x.device)
    distance=(x[i]-x[j]).square().sum(-1).sqrt()
    edges=radial_log_density(distance,edge_lengths[i,j],width)
    log_h=x.new_full((n,n),-torch.inf)
    log_h[i,j]=edges;log_h[j,i]=edges
    return 1.5*math.log(n)+log_tree_partition(log_weights+log_h)-log_tree_partition(log_weights)


def weighted_tree(log_weights, rng):
    """Wilson loop-erased random walk for an undirected weighted tree."""
    logw=np.asarray(log_weights,dtype=np.float64).copy();n=len(logw)
    np.fill_diagonal(logw,-np.inf)
    if n==1:return []
    weights=np.exp(logw-np.max(logw,axis=1,keepdims=True))
    transition=weights/weights.sum(axis=1,keepdims=True)
    # Root choice does not change the undirected tree law. A high-degree root
    # avoids long initial walks to a very low-propensity leaf.
    root=int(np.argmax(np.logaddexp.reduce(logw,axis=1)))
    in_tree={root};edges=[]
    for start in range(n):
        if start in in_tree:continue
        path=[start];locations={start:0}
        while path[-1] not in in_tree:
            nxt=int(rng.choice(n,p=transition[path[-1]]))
            if nxt in locations:
                cut=locations[nxt]
                for old in path[cut+1:]:locations.pop(old)
                path=path[:cut+1]
            else:
                locations[nxt]=len(path);path.append(nxt)
        for child,parent in zip(path[:-1],path[1:]):
            edges.append((child,parent));in_tree.add(child)
    if len(edges)!=n-1:raise AssertionError('Tree sampler lost an edge')
    return edges


@torch.no_grad()
def sample_tree_coordinates(log_weights, edge_lengths, *, rng, generator, width=.2):
    n=len(log_weights)
    if n==1:return edge_lengths.new_zeros((1,3)),[]
    edges=weighted_tree(log_weights.detach().cpu().numpy(),rng)
    direction=torch.randn((n-1,3),dtype=edge_lengths.dtype,device=edge_lengths.device,generator=generator)
    direction=direction/direction.norm(dim=1,keepdim=True)
    innovation=torch.randn((n-1,),dtype=edge_lengths.dtype,device=edge_lengths.device,generator=generator)
    length=torch.stack([edge_lengths[i,j] for i,j in edges])
    radius=length*torch.exp(width*innovation-.5*width**2)
    displacement=direction*radius[:,None]
    adjacency=[[] for _ in range(n)]
    for e,(child,parent) in enumerate(edges):
        adjacency[parent].append((child,e,1));adjacency[child].append((parent,e,-1))
    positions=edge_lengths.new_zeros((n,3));seen={0};stack=[0]
    while stack:
        parent=stack.pop()
        for child,e,sign in adjacency[parent]:
            if child in seen:continue
            positions[child]=positions[parent]+sign*displacement[e]
            seen.add(child);stack.append(child)
    return positions-positions.mean(0,keepdim=True),edges


class TreeMixturePrior(nn.Module):
    """Composition/electronic-state conditioned tree weights; no geometry input."""
    def __init__(self,mode='pair',embedding=8,hidden=32,width=.2):
        super().__init__()
        if mode not in ['fixed','node','pair']:raise ValueError('Unknown tree-weight model')
        self.mode=mode;self.width=width
        self.configuration=dict(mode=mode,embedding=embedding,hidden=hidden,width=width)
        table=Chem.GetPeriodicTable()
        radii=[1.]+[table.GetRcovalent(z) for z in range(1,119)]
        propensity=[1.]+[.05 if z in [1,9,17,35,53] else float(max(1,table.GetDefaultValence(z)-1)) for z in range(1,119)]
        self.register_buffer('radii',torch.tensor(radii))
        self.register_buffer('base_log_propensity',torch.tensor(propensity).log())
        self.embedding=nn.Embedding(119,embedding)
        dimension=(2 if mode=='node' else 3)*embedding+3
        self.head=nn.Sequential(nn.Linear(dimension,hidden),nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        if mode=='fixed':self.requires_grad_(False)

    def parameters_for(self,numbers,charge,spin):
        numbers=torch.as_tensor(numbers,dtype=torch.long,device=self.radii.device)
        n=len(numbers)
        if not 1<=n<=200 or (numbers<1).any() or (numbers>118).any() or float(spin)<1:
            raise ValueError('Require1--200 supported atoms and positive spin multiplicity')
        electrons=int(numbers.sum())-float(charge)
        if float(charge)!=round(float(charge)) or float(spin)!=round(float(spin)) or electrons<1 or electrons<float(spin)-1 or (electrons-float(spin)+1)%2:
            raise ValueError('Electronic condition violates integer/electron-count constraints')
        base=self.base_log_propensity[numbers]
        loga=base[:,None]+base[None,:]
        length=self.radii[numbers,None]+self.radii[numbers][None,:]
        if self.mode!='fixed':
            h=self.embedding(numbers);pool=h.mean(0)
            context=h.new_tensor([math.log(n),float(charge)/n,math.log(float(spin))])
            if self.mode=='node':
                inputs=torch.cat([h,pool.expand(n,-1),context.expand(n,-1)],-1)
                u=2*torch.tanh(self.head(inputs).squeeze(-1))
                loga=loga+u[:,None]+u[None,:]
            else:
                inputs=torch.cat([h[:,None]+h[None,:],h[:,None]*h[None,:],pool.expand(n,n,-1),context.expand(n,n,-1)],-1)
                loga=loga+2*torch.tanh(self.head(inputs).squeeze(-1))
        return loga,length

    def log_prob(self,x,numbers,charge,spin):
        a,length=self.parameters_for(numbers,charge,spin)
        return tree_mixture_log_prob(x.to(a),a,length,self.width)

    @torch.no_grad()
    def sample(self,numbers,charge,spin,*,rng,generator):
        a,length=self.parameters_for(numbers,charge,spin)
        return sample_tree_coordinates(a,length,rng=rng,generator=generator,width=self.width)
