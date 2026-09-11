"""Explicit geometric support for a connected, non-overlapping atomic cluster.

This is not a chemical-valence, molecular-identity or stereochemistry certificate.
It uses the same covalent-radius thresholds as the independent geometry audit.
"""
import torch


@torch.no_grad()
def connected_nonoverlapping(positions,radii,*,contact_factor=1.25,overlap_factor=.6):
    if (positions.ndim!=3 or positions.shape[-1]!=3 or radii.shape!=(positions.shape[1],)
            or positions.shape[1]<2 or not torch.isfinite(positions).all()
            or not torch.isfinite(radii).all() or (radii<=0).any()
            or not 0<overlap_factor<contact_factor):
        raise ValueError('Require finite atomic positions, positive radii and ordered thresholds')
    n=positions.shape[1];squared=(positions[:,:,None,:]-positions[:,None,:,:]).square().sum(-1)
    scale=radii[:,None]+radii[None,:]
    off_diagonal=~torch.eye(n,dtype=torch.bool,device=positions.device)
    overlap=((squared<(overlap_factor*scale).square())&off_diagonal).any((1,2))
    adjacency=squared<=(contact_factor*scale).square()
    reached=torch.zeros(positions.shape[:2],dtype=torch.bool,device=positions.device);reached[:,0]=True
    for _ in range(n-1):
        updated=reached|(adjacency&reached[:,:,None]).any(1)
        if torch.equal(updated,reached):break
        reached=updated
    return reached.all(1)&~overlap
