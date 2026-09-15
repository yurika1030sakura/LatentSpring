"""Data-derived ball-and-stick illustrations with fixed orthographic cameras."""
from functools import lru_cache
import numpy as np
from matplotlib.colors import to_rgb


ATOM_COLORS={1:'#e8edf2',5:'#ddb194',6:'#536277',7:'#397bc4',8:'#e25a63',9:'#75b77b',
    14:'#ae9cbd',15:'#e3a256',16:'#d6be51',17:'#6aa46a',35:'#a86670',53:'#896d9d'}


def camera(positions):
    x=np.asarray(positions);x=x-x.mean(0)
    _,_,vh=np.linalg.svd(x,full_matrices=False);rotation=vh.T
    if np.linalg.det(rotation)<0:rotation[:,-1]*=-1
    a,b=np.deg2rad([18.,-16.])
    yaw=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    pitch=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    return rotation@yaw@pitch


@lru_cache(maxsize=24)
def sphere(color):
    """Render the visible hemisphere with a common light; no geometry editing."""
    grid=np.linspace(-1.03,1.03,100);x,y=np.meshgrid(grid,grid)
    r2=x*x+y*y;z=np.sqrt(np.clip(1-r2,0,1))
    normals=np.stack([x,y,z],-1)
    light=np.array([-.45,.65,1.]);light/=np.linalg.norm(light)
    half=(light+np.array([0,0,1.]));half/=np.linalg.norm(half)
    diffuse=np.maximum(normals@light,0)
    specular=np.maximum(normals@half,0)**30
    rgb=np.array(to_rgb(color))[None,None]*(.34+.63*diffuse[...,None])+.33*specular[...,None]
    alpha=np.clip((1-np.sqrt(r2))*65+.5,0,1)
    return np.dstack([np.clip(rgb,0,1),alpha])


def bonds_from_graph(graph):
    matrix=np.asarray(graph['bond_orders'])
    return [(i,j,float(matrix[i,j])) for i in range(len(matrix)) for j in range(i+1,len(matrix)) if matrix[i,j]>0]


def draw_molecule(ax,positions,numbers,*,rotation=None,bonds=(),scaffold=(),extent=None,forces=None,
                  radius_scale=1.,force_scale=None,alpha=1.):
    x=np.asarray(positions,dtype=float);x=x-x.mean(0)
    rotation=camera(x) if rotation is None else rotation
    view=x@rotation
    if extent is None:
        width=max(np.ptp(view[:,0]),np.ptp(view[:,1]),2.)+1.4
        extent=(-width/2,width/2,-width/2,width/2)
    ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal');ax.axis('off')
    bbox=ax.get_position();points_per_unit=bbox.width*ax.figure.get_figwidth()*72/(extent[1]-extent[0])
    zmin,zmax=view[:,2].min()-1,view[:,2].max()+1
    def order(z):return 2+8*(z-zmin)/(zmax-zmin)
    for edge in scaffold:
        i,j=map(int,edge[:2]);ax.plot(view[[i,j],0],view[[i,j],1],ls=(0,(2,2)),lw=.95,color='#62a79f',alpha=.8*alpha,zorder=1)
    for i,j,bond_order in bonds:
        delta=view[j,:2]-view[i,:2];unit=np.array([-delta[1],delta[0]])/max(np.linalg.norm(delta),1e-8)
        offsets=[0.] if bond_order<1.8 else ([-.09,.09] if bond_order<2.8 else [-.15,0.,.15])
        for offset in offsets:
            for t0,t1 in zip(np.linspace(0,1,17)[:-1],np.linspace(0,1,17)[1:]):
                p=(1-t0)*view[i]+t0*view[j];q=(1-t1)*view[i]+t1*view[j]
                color=ATOM_COLORS.get(int(numbers[i if (t0+t1)/2<.5 else j]),'#938b80')
                ax.plot([p[0]+offset*unit[0],q[0]+offset*unit[0]],
                    [p[1]+offset*unit[1],q[1]+offset*unit[1]],color=color,
                    lw=max(.8,.14*points_per_unit),solid_capstyle='round',alpha=alpha,zorder=order((p[2]+q[2])/2))
    for (xx,yy,zz),number in zip(view,numbers):
        radius=(.205 if number==1 else .29)*radius_scale
        ax.imshow(sphere(ATOM_COLORS.get(int(number),'#938b80')),origin='lower',
            extent=(xx-radius,xx+radius,yy-radius,yy+radius),zorder=order(zz+.1),alpha=alpha,interpolation='bilinear')
    if forces is not None:
        force=np.asarray(forces)@rotation
        if force_scale is None:force_scale=.85/max(np.linalg.norm(force,axis=-1).max(),1e-12)
        indices=np.argsort(np.linalg.norm(force,axis=-1))[-5:]
        for i in indices:
            ax.annotate('',xy=view[i,:2]+force_scale*force[i,:2],xytext=view[i,:2],
                arrowprops=dict(arrowstyle='-|>',lw=1.2,color='#bf7a30',mutation_scale=9),zorder=15)
    return rotation,extent
