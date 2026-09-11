"""Fixed graph guidance for proposal paths; never an added physical target term."""
import torch


def exchanged_bond_graph(bonds,action):
    i,j,k,l=action
    if i==j or k in (i,j) or l in (i,j):raise ValueError('Require passive anchors')
    if float(bonds[i,k])!=1 or float(bonds[j,l])!=1 or int((bonds[i]>0).sum())!=1 or int((bonds[j]>0).sum())!=1:
        raise ValueError('Single-bonded terminal roots required')
    result=bonds.clone()
    result[i,k]=result[k,i]=result[j,l]=result[l,j]=0
    result[i,l]=result[l,i]=result[j,k]=result[k,j]=1
    return result


def graph_guide_energy_force(x,bonds,radii,*,bond_kappa,nonbond_kappa,nonbond_factor):
    """Covalent-radius bond springs and a soft nonbond contact exclusion.

    Applies to each labelled graph independently and is invariant under rigid
    motions and joint atom permutations. This pilot uses single-bond graphs;
    these radii/springs are a proposal guide, not a force field or valence model.
    """
    if (x.ndim!=3 or x.shape[-1]!=3 or bonds.shape!=x.shape[:2]+(x.shape[1],)
            or radii.shape!=(x.shape[1],) or bond_kappa<=0 or nonbond_kappa<=0 or nonbond_factor<=1):
        raise ValueError('Invalid graph-guide inputs')
    if not all(torch.isfinite(v).all() for v in [x,bonds,radii]) or (radii<=0).any():
        raise ValueError('Finite geometry, graph and positive radii required')
    if (not torch.equal(bonds,bonds.transpose(1,2)) or bonds.diagonal(dim1=1,dim2=2).any()
            or not ((bonds==0)|(bonds==1)).all()):
        raise ValueError('This guide is qualified only for symmetric single-bond graphs')
    delta=x[:,:,None]-x[:,None,:]
    distance=(delta.square().sum(-1)+1e-12).sqrt()
    reference=radii[:,None]+radii[None,:]
    bonded=bonds>0;off_diagonal=~torch.eye(x.shape[1],device=x.device,dtype=torch.bool)
    stretch=distance-reference
    overlap=(nonbond_factor*reference-distance).clamp_min(0)
    pair_energy=.5*bond_kappa*stretch.square()*bonded+.5*nonbond_kappa*overlap.square()*(~bonded)
    energy=.5*(pair_energy*off_diagonal).sum((1,2))
    derivative=bond_kappa*stretch*bonded-nonbond_kappa*overlap*(~bonded)
    force=-(derivative[...,None]*delta/distance[...,None]*off_diagonal[None,:,:,None]).sum(2)
    return energy,force
