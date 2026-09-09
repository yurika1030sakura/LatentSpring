"""Geometry-only diagnostics; no bond labels or chemical-validity assertion."""
import numpy as np


def distance_profile(positions, atomic_numbers):
    """Sorted distances grouped by unordered element pair, not molecular RMSD.

    The profile is invariant to rigid transforms and permutation of atoms of
    equal element. It is not a complete invariant of molecular structure.
    """
    x=np.asarray(positions,dtype=np.float64);z=np.asarray(atomic_numbers,dtype=np.int64)
    if x.shape!=(len(z),3) or len(z)<1 or not np.isfinite(x).all():
        raise ValueError('Require finite N by 3 positions and N elements')
    i,j=np.triu_indices(len(z),k=1)
    distances=np.linalg.norm(x[i]-x[j],axis=-1)
    pairs=np.sort(np.stack([z[i],z[j]],axis=1),axis=1)
    keys=sorted(set(map(tuple,pairs)))
    return {key:np.sort(distances[np.all(pairs==key,axis=1)]) for key in keys}


def profile_rms(first,second):
    if first.keys()!=second.keys() or any(first[k].shape!=second[k].shape for k in first):
        raise ValueError('Profiles must have the same elemental composition')
    if not first:return 0.
    delta=np.concatenate([first[k]-second[k] for k in first])
    return float(np.sqrt(np.mean(delta**2)))


def contact_summary(positions,covalent_radii,*,contact_factor=1.25,overlap_factor=.6):
    x=np.asarray(positions,dtype=np.float64);r=np.asarray(covalent_radii,dtype=np.float64)
    if x.shape!=(len(r),3) or len(r)<1 or not np.isfinite(x).all() or not np.isfinite(r).all() or np.any(r<=0):
        raise ValueError('Require finite positions and positive covalent radii')
    if not 0<overlap_factor<contact_factor:raise ValueError('Require ordered positive thresholds')
    i,j=np.triu_indices(len(r),k=1)
    ratios=np.linalg.norm(x[i]-x[j],axis=-1)/(r[i]+r[j])
    components=list(range(len(r)))
    def root(k):
        while components[k]!=k:k=components[k]
        return k
    for a,b in zip(i[ratios<=contact_factor],j[ratios<=contact_factor]):components[root(a)]=root(b)
    return {'min_covalent_distance_ratio':float(ratios.min()) if len(ratios) else None,
            'overlap_pairs':int(np.sum(ratios<overlap_factor)),
            'contact_components':len({root(k) for k in range(len(r))}),
            'contact_edges':int(np.sum(ratios<=contact_factor))}
