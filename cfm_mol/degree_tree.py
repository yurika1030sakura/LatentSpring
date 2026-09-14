"""Normalized degree-sensitive latent trees, using classical Pruefer enumeration.

The degree cap constrains an auxiliary spatial tree, not inferred bond orders.
Nonlinear degree potentials can learn coordination histograms; neither this
combinatorial identity nor degree constraints are claimed as new mathematics.
"""
import heapq
import math
import numpy as np
import torch
from torch import nn

# Conservative coordination ceilings for the declared neutral organic pilot.
# N/O/B ceilings allow common charged local centers within a neutral molecule.
ORGANIC_CAPS={1:1,5:4,6:4,7:4,8:3,9:1,14:4,15:6,16:6,17:1,35:1,53:1}


def count_prefixes(potentials,max_counts,total):
    """Reachable log coefficients; suitable for core degrees and leaf counts."""
    if total<0 or min(max_counts,default=0)<0 or sum(max_counts)<total:raise ValueError('Infeasible count budget')
    if len(potentials)!=len(max_counts) or max(max_counts,default=0)>=potentials.shape[1]:raise ValueError('Missing count potentials')
    if not torch.isfinite(potentials).all():raise ValueError('Finite count potentials required')
    prefixes=[potentials.new_zeros(1)]
    for i,cap in enumerate(max_counts):
        previous=prefixes[-1];length=min(total+1,len(previous)+cap)
        terms=[]
        for k in range(min(cap,length-1)+1):
            values=previous[:length-k]+potentials[i,k]-math.lgamma(k+1)
            terms.append(torch.nn.functional.pad(values,(k,length-k-len(values)),value=-torch.inf))
        prefixes.append(torch.logsumexp(torch.stack(terms),0))
    return prefixes


def degree_prefixes(potentials,caps):
    """Log coefficients of product_i sum_k exp(psi_i(k))*z^k/k!."""
    n=len(caps);m=n-2
    if not 2<=n<=200 or potentials.ndim!=2 or len(potentials)!=n:
        raise ValueError('Require2--200 atoms and node/degree potential matrix')
    caps=[min(int(c),n-1) for c in caps]
    if min(caps)<1 or sum(c-1 for c in caps)<m:raise ValueError('No tree with these degree capacities')
    return count_prefixes(potentials,[c-1 for c in caps],m)


@torch.no_grad()
def sample_counts(potentials,max_counts,total,rng):
    prefixes=count_prefixes(potentials,max_counts,total);remaining=total;counts=[]
    for i in reversed(range(len(max_counts))):
        choices=[k for k in range(min(int(max_counts[i]),remaining)+1) if remaining-k<len(prefixes[i])]
        logp=torch.stack([potentials[i,k]-math.lgamma(k+1)+prefixes[i][remaining-k] for k in choices])
        k=int(rng.choice(choices,p=logp.softmax(0).cpu().numpy()));counts.append(k);remaining-=k
    assert remaining==0
    return list(reversed(counts))


def degree_log_partition(potentials,caps):
    return degree_prefixes(potentials,caps)[-1][-1]+math.lgamma(len(caps)-1)


def decode_pruefer(code,n):
    if len(code)!=n-2 or any(not 0<=int(i)<n for i in code):raise ValueError('Invalid Pruefer code')
    degree=np.ones(n,dtype=int)
    for v in code:degree[int(v)]+=1
    leaves=[i for i,d in enumerate(degree) if d==1];heapq.heapify(leaves);edges=[]
    for v in code:
        v=int(v);leaf=heapq.heappop(leaves);edges.append((min(leaf,v),max(leaf,v)))
        degree[leaf]-=1;degree[v]-=1
        if degree[v]==1:heapq.heappush(leaves,v)
    i,j=leaves;edges.append((min(i,j),max(i,j)))
    return sorted(edges)


def tree_degrees(edges,n):
    if len(edges)!=n-1:raise ValueError('A tree requires N-1 edges')
    degree=[0]*n;parent=list(range(n));seen=set()
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,j in edges:
        if not 0<=i<n or not 0<=j<n or i==j or (min(i,j),max(i,j)) in seen:raise ValueError('Invalid tree edge')
        ri,rj=root(i),root(j)
        if ri==rj:raise ValueError('Tree contains a cycle')
        parent[ri]=rj;degree[i]+=1;degree[j]+=1;seen.add((min(i,j),max(i,j)))
    return degree


@torch.no_grad()
def sample_degree_tree(potentials,caps,rng):
    prefixes=degree_prefixes(potentials,caps);remaining=len(caps)-2;counts=[]
    for i in reversed(range(len(caps))):
        choices=[k for k in range(min(int(caps[i])-1,remaining)+1) if remaining-k<len(prefixes[i])]
        logp=torch.stack([potentials[i,k]-math.lgamma(k+1)+prefixes[i][remaining-k] for k in choices])
        k=int(rng.choice(choices,p=logp.softmax(0).cpu().numpy()));counts.append(k);remaining-=k
    assert remaining==0
    counts=list(reversed(counts));code=np.repeat(np.arange(len(caps)),counts);rng.shuffle(code)
    edges=decode_pruefer(code,len(caps))
    assert tree_degrees(edges,len(caps))==[k+1 for k in counts]
    return edges


class DegreeTreePrior(nn.Module):
    """Composition/state-conditioned nonlinear degree potentials; exact tree Z."""
    def __init__(self,embedding=8,hidden=32):
        super().__init__();self.configuration=dict(embedding=embedding,hidden=hidden)
        self.embedding=nn.Embedding(119,embedding)
        self.degree_embedding=nn.Embedding(6,embedding)
        self.head=nn.Sequential(nn.Linear(3*embedding+3,hidden),nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def parameters_for(self,numbers,charge,spin):
        z=torch.as_tensor(numbers,dtype=torch.long,device=self.embedding.weight.device);n=len(z)
        if not 2<=n<=200 or any(int(v) not in ORGANIC_CAPS for v in z):raise ValueError('Require2--200 atoms in the declared organic scope')
        electrons=int(z.sum())-float(charge)
        if float(charge)!=round(float(charge)) or float(spin)!=round(float(spin)) or spin<1 or electrons<1 or electrons<spin-1 or (electrons-spin+1)%2:
            raise ValueError('Invalid electronic state')
        caps=[min(ORGANIC_CAPS[int(v)],n-1) for v in z]
        h=self.embedding(z);pool=h.mean(0);d=self.degree_embedding(torch.arange(6,device=z.device))
        context=h.new_tensor([math.log(n),float(charge)/n,math.log(float(spin))])
        inputs=torch.cat([h[:,None,:].expand(-1,6,-1),pool.expand(n,6,-1),d[None].expand(n,-1,-1),context.expand(n,6,-1)],-1)
        return 2*torch.tanh(self.head(inputs).squeeze(-1)),caps

    def log_prob(self,edges,numbers,charge,spin):
        psi,caps=self.parameters_for(numbers,charge,spin);degree=tree_degrees(edges,len(numbers))
        if any(d>c for d,c in zip(degree,caps)):return psi.new_full((),-torch.inf)
        k=torch.tensor(degree,device=psi.device)-1
        return psi[torch.arange(len(k),device=psi.device),k].sum()-degree_log_partition(psi,caps)

    @torch.no_grad()
    def sample(self,numbers,charge,spin,*,rng):
        psi,caps=self.parameters_for(numbers,charge,spin)
        return sample_degree_tree(psi,caps,rng)


class CoordinationTreePrior(DegreeTreePrior):
    """Degree-sensitive heavy core and capacity-preserving terminal allocations.

    Each terminal species is allocated by an exact count coefficient, followed
    by uniform assignment of its atom labels. No ordering of identical leaves
    changes the law. Source tree probability is explicit and normalized.
    """
    terminal_species={1,9,17,35,53}

    def __init__(self,embedding=8,hidden=32):
        super().__init__(embedding,hidden)
        self.allocation_degree_embedding=nn.Embedding(7,embedding)
        self.allocation_head=nn.Sequential(nn.Linear(4*embedding+5,hidden),nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.allocation_head[-1].weight);nn.init.zeros_(self.allocation_head[-1].bias)

    def decomposition(self,numbers,charge,spin):
        psi,caps=self.parameters_for(numbers,charge,spin)
        core=[i for i,z in enumerate(numbers) if int(z) not in self.terminal_species]
        if not core:raise ValueError('Declared organic prior requires a nonterminal core')
        groups={z:[i for i,value in enumerate(numbers) if int(value)==z] for z in sorted(self.terminal_species) if z in numbers}
        if sum(caps[i] for i in core)-2*(len(core)-1)<sum(map(len,groups.values())):
            raise ValueError('Insufficient coordination capacity for a connected tree')
        return psi,caps,core,groups

    def allocation_potentials(self,numbers,charge,spin,core,degree,remaining,leaf_z):
        z=torch.as_tensor(numbers,dtype=torch.long,device=self.embedding.weight.device)
        h=self.embedding(z);n=len(core);pool=h.mean(0)
        k=self.allocation_degree_embedding(torch.arange(7,device=z.device))
        leaf=self.embedding(z.new_tensor(leaf_z))
        context=h.new_tensor([[math.log(len(numbers)),float(charge)/len(numbers),math.log(float(spin)),d/6,r/6] for d,r in zip(degree,remaining)])
        inputs=torch.cat([h[core,None,:].expand(-1,7,-1),pool.expand(n,7,-1),leaf.expand(n,7,-1),k[None].expand(n,-1,-1),context[:,None,:].expand(-1,7,-1)],-1)
        return 2*torch.tanh(self.allocation_head(inputs).squeeze(-1))

    def log_prob(self,edges,numbers,charge,spin):
        psi,caps,core,groups=self.decomposition(numbers,charge,spin);degrees=tree_degrees(edges,len(numbers))
        if any(d>c for d,c in zip(degrees,caps)):return psi.new_full((),-torch.inf)
        if any(degrees[i]!=1 for members in groups.values() for i in members):return psi.new_full((),-torch.inf)
        lookup={node:i for i,node in enumerate(core)}
        core_edges=[(lookup[i],lookup[j]) for i,j in edges if i in lookup and j in lookup]
        dc=tree_degrees(core_edges,len(core)) if len(core)>1 else [0]
        if len(core)>1:
            k=torch.tensor(dc,device=psi.device)-1
            logp=psi[torch.tensor(core,device=psi.device),k].sum()-degree_log_partition(psi[core],[caps[i] for i in core])
        else:logp=psi.sum()*0
        remaining=[caps[i]-d for i,d in zip(core,dc)]
        parent={}
        for i,j in edges:
            if i not in lookup:parent[i]=j
            if j not in lookup:parent[j]=i
        for z,members in groups.items():
            counts=[0]*len(core)
            for node in members:
                if parent[node] not in lookup:return psi.new_full((),-torch.inf)
                counts[lookup[parent[node]]]+=1
            if any(k>r for k,r in zip(counts,remaining)):return psi.new_full((),-torch.inf)
            scores=self.allocation_potentials(numbers,charge,spin,core,dc,remaining,z)
            logp=logp+scores[torch.arange(len(core),device=psi.device),torch.tensor(counts,device=psi.device)].sum()
            logp=logp-math.lgamma(len(members)+1)-count_prefixes(scores,remaining,len(members))[-1][-1]
            remaining=[r-k for r,k in zip(remaining,counts)]
        return logp

    @torch.no_grad()
    def sample(self,numbers,charge,spin,*,rng):
        psi,caps,core,groups=self.decomposition(numbers,charge,spin)
        local=sample_degree_tree(psi[core],[caps[i] for i in core],rng) if len(core)>1 else []
        dc=tree_degrees(local,len(core)) if len(core)>1 else [0]
        edges=[(core[i],core[j]) for i,j in local];remaining=[caps[i]-d for i,d in zip(core,dc)]
        for z,members in groups.items():
            scores=self.allocation_potentials(numbers,charge,spin,core,dc,remaining,z)
            counts=sample_counts(scores,remaining,len(members),rng)
            parents=np.repeat(np.array(core),counts);rng.shuffle(parents)
            edges.extend((int(parent),node) for parent,node in zip(parents,members))
            remaining=[r-k for r,k in zip(remaining,counts)]
        assert all(d<=c for d,c in zip(tree_degrees(edges,len(numbers)),caps))
        return sorted((min(i,j),max(i,j)) for i,j in edges)
