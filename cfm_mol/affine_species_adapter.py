"""Same neural contexts and coupling schedule, affine active-coordinate control.

Each point map is replaced by its exact tangent F(0)+DF(0)u. The conditioned
composition is still nonlinear in all coordinates. This ablation isolates the
additional active-coordinate nonlinearity of the convex point map, rather than
conflating that nonlinearity with the invariant neural conditioner.
"""
import torch

from cfm_mol.centered_convex_flow import convex_point_map
from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter


class AffineSpeciesCouplingAdapter(SpeciesCouplingAdapter):
    def apply_layer(self, x, layer, *, inverse=False, tolerance=1e-11):
        context, origin, active, roles, a, b = self.split_context(x, layer)
        parameters, shift = self.conditioner(context, origin, self.numbers, roles, self.electronic,
            0. if layer[0] == 'internal' else 1.)
        offset, derivative, _ = convex_point_map(torch.zeros_like(active[:, :1]), *parameters)
        matrix = derivative[:, 0]
        volume = 2*torch.linalg.cholesky(matrix).diagonal(dim1=-2, dim2=-1).log().sum(-1)
        if layer[0] == 'internal':
            # Centering cancels F(0); there are n_g-1 independent3-vectors.
            offset = torch.zeros_like(offset)
            volume = (active.shape[1]-1)*volume
        else:
            offset = offset+shift
        if inverse:
            changed = torch.linalg.solve(matrix, (active-offset).transpose(-1, -2)).transpose(-1, -2)
            residual = float((changed@matrix.transpose(-1, -2)+offset-active).abs().max())
            if residual > tolerance:
                raise RuntimeError('Affine active-coordinate inverse failed')
            diagnostic = {'iterations': 0, 'maximum_residual': residual, 'method': 'linear_solve'}
        else:
            changed = active@matrix.transpose(-1, -2)+offset
            diagnostic = None
        out = x.clone()
        if layer[0] == 'internal':
            out[:, a] = origin+changed
        else:
            difference = changed-active
            na, nb = int(a.sum()), int(b.sum())
            out[:, a] = x[:, a]+nb/(na+nb)*difference
            out[:, b] = x[:, b]-na/(na+nb)*difference
        return out, -volume if inverse else volume, diagnostic
