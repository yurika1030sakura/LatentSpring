import math
from types import SimpleNamespace
import dgl
import torch
from cfm_mol.smooth_geometry import patch_smooth_geometry


def test_edge_softening_is_equivariant_and_bounded_at_collision():
    original=lambda *args: 'original'
    field=SimpleNamespace(precompute_distances=original,rbf_dim=4,rbf_dmax=20.)
    model=SimpleNamespace(vector_field=field)
    patch_smooth_geometry(model,.1)
    graph=dgl.graph(([0,1],[1,0]),num_nodes=2)
    x=torch.tensor([[0.,0.,0.],[1.,.3,.5]],dtype=torch.float64)
    graph.ndata['x_t']=x
    direction,rbf=field.precompute_distances(graph)
    angle=.7;rotation=x.new_tensor([[math.cos(angle),-math.sin(angle),0],
                                  [math.sin(angle),math.cos(angle),0],[0,0,1]])
    graph.ndata['x_t']=x@rotation+2
    moved_direction,moved_rbf=field.precompute_distances(graph)
    torch.testing.assert_close(moved_direction,direction@rotation,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(moved_rbf,rbf,rtol=1e-12,atol=1e-12)
    delta=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    def operation(value):
        return field.precompute_distances(graph,torch.stack([value,torch.zeros_like(value)]))[0][0]
    jac=torch.autograd.functional.jacobian(operation,delta,create_graph=True)
    torch.testing.assert_close(jac,torch.eye(3,dtype=jac.dtype)*10)
    assert torch.isfinite(torch.autograd.grad(jac.square().sum(),delta)[0]).all()
    patch_smooth_geometry(model,0.)
    assert field.precompute_distances is original
