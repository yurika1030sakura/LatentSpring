"""Four-state contrast representation for non-additive electronic edit work.

Symmetric environment coefficients multiply mixed radial contrasts. Reversing
either edit flips the output exactly; exchanging the edit groups preserves it.
These algebraic properties alone are not an accuracy or novelty certificate.
"""
import math
import torch
from torch import nn
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.chemical_work import mlp


class InteractionWorkModel(nn.Module):
    def __init__(self, elements, hidden=16, radial=16, environment=True, linear=False):
        super().__init__()
        self.configuration=dict(elements=list(elements),hidden=hidden,radial=radial,environment=environment,linear=linear)
        self.environment,self.linear=environment,linear
        self.register_buffer('radii',covalent_radii(range(119)))
        z=torch.arange(119,dtype=torch.float64)
        self.register_buffer('descriptors',torch.stack([z/118,torch.log1p(z)/math.log(119),self.radii/2],-1))
        self.register_buffer('centers',torch.linspace(0,6,radial));self.width=6/(radial-1)
        lookup=torch.full((119,119),-1,dtype=torch.long);n=0
        for i,z in enumerate(elements):
            for w in elements[i:]:lookup[z,w]=lookup[w,z]=n;n+=1
        self.register_buffer('lookup',lookup)
        if linear:
            self.coefficients=nn.Parameter(torch.zeros(n,radial+1))
        else:
            self.elements=mlp(3,hidden,hidden);self.state=mlp(3,hidden,hidden)
            self.context_query=mlp(2*hidden+radial+2,hidden,hidden)
            self.context_update=mlp(2*hidden,hidden,hidden)
            self.coefficient_head=mlp(2*hidden,hidden,radial+1)
            nn.init.zeros_(self.coefficient_head[-1].weight);nn.init.zeros_(self.coefficient_head[-1].bias)

    def basis(self,distances):
        radial=torch.exp(-.5*((distances[...,None]-self.centers)/self.width)**2)
        return torch.cat([radial,distances.clamp_min(.25).reciprocal()[...,None]],-1)

    def forward(self,positions,graphs,numbers,electronic,actions):
        # positions[B,4,N,3], graphs[B,4,N,N], actions[B,2,4].
        batch,_,n,_=positions.shape
        if not 5<=n<=200 or positions.shape[1]!=4 or actions.shape!=(batch,2,4):raise ValueError('A four-state square and two edits on5..200 atoms are required')
        active=actions[:,:,:2].reshape(batch,4)
        if any(len(set(row.tolist()))!=4 for row in active):raise ValueError('Four different moved atoms required')
        b=torch.arange(batch,device=positions.device)[:,None]
        locations=positions.permute(0,2,1,3)[b,active].permute(0,2,1,3)
        za=numbers[active];radius=self.radii[za[:,:2]][:,:,None]+self.radii[za[:,2:]][:,None,:]
        delta=locations[:,:,:2,None]-locations[:,:,None,2:]
        distance=delta.square().sum(-1).clamp_min(1e-20).sqrt()/radius[:,None]
        phi=self.basis(distance)
        contrast=phi[:,3]-phi[:,1]-phi[:,2]+phi[:,0]
        if self.linear:
            indices=self.lookup[za[:,:2,None],za[:,None,2:]]
            coeff=self.coefficients[indices.clamp_min(0)]*(indices>=0)[...,None]
        else:
            state=self.state(electronic.to(positions))[:,None]
            root=self.elements(self.descriptors[za].to(positions))+state
            if self.environment:
                mask=torch.ones((batch,n),dtype=torch.bool,device=positions.device);mask.scatter_(1,active,False)
                passive=torch.arange(n,device=positions.device).expand(batch,n)[mask].reshape(batch,n-4)
                # Four-corner averaging is invariant to reversing either edit.
                # Passive positions are independently centered in each corner.
                pp=positions.permute(0,2,1,3)[b,passive].permute(0,2,1,3)
                center=pp.mean(2,keepdim=True)
                midpoint=(locations-center).mean(1);passive_x=(pp-center).mean(1)
                zp=numbers[passive];nodes=self.elements(self.descriptors[zp].to(positions))+state
                dr=midpoint[:,:,None]-passive_x[:,None]
                radial=self.basis(dr.square().sum(-1).clamp_min(1e-20).sqrt()/(self.radii[za][:,:,None]+self.radii[zp][:,None]))
                mean_graph=graphs.mean(1)
                bonds=mean_graph[b[:,:,None],active[:,:,None],passive[:,None,:]]
                left=root[:,:,None].expand(-1,-1,n-4,-1);right=nodes[:,None].expand(-1,4,-1,-1)
                messages=self.context_query(torch.cat([left,right,radial,bonds[...,None]/3],-1))
                root=root+self.context_update(torch.cat([root,messages.sum(2)/math.sqrt(n-4)],-1))
            left=root[:,:2,None].expand(-1,-1,2,-1);right=root[:,None,2:].expand(-1,2,-1,-1)
            coeff=self.coefficient_head(torch.cat([left+right,(left-right).square()],-1))
        return (contrast*coeff).sum((1,2,3))
