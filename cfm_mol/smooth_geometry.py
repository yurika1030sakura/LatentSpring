"""Optional smooth edge directions for a separately identified position field.

FlowMol repeatedly normalizes differences between internal predicted atom
coordinates. Replacing r/||r|| by r/sqrt(||r||^2+rho^2) bounds this operation's
Jacobian by 1/rho and preserves translation/rotation equivariance. This changes
the model and requires its own training and evaluation; it is not a solver fix
that can be silently applied to old likelihoods. No external source is edited.
"""
import math
from types import MethodType
import torch


def patch_smooth_geometry(model,softening):
    softening=float(softening)
    if not math.isfinite(softening) or softening<0:
        raise ValueError('Geometry softening must be finite and nonnegative')
    field=model.vector_field
    if not hasattr(field,'_bgfm_original_precompute_distances'):
        field._bgfm_original_precompute_distances=field.precompute_distances
    if softening==0:
        field.precompute_distances=field._bgfm_original_precompute_distances
    else:
        from flowmol.models.gvp import _rbf
        def distances(self,graph,node_positions=None):
            x=graph.ndata['x_t'] if node_positions is None else node_positions
            source,destination=graph.edges()
            delta=x[source]-x[destination]
            radius=torch.sqrt(delta.square().sum(-1,keepdim=True)+softening**2)
            return delta/radius,_rbf(radius.squeeze(-1),D_max=self.rbf_dmax,D_count=self.rbf_dim)
        field.precompute_distances=MethodType(distances,field)
    field._bgfm_geometry_softening=softening
    return model
