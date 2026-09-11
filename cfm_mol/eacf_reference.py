"""Pinned upstream EACF architecture/optimizer recipes for matched-source controls.

Import only in the isolated JAX environment with the pinned EACF checkout on
PYTHONPATH. The learned map acts on x and an auxiliary distributed as N(x,.1^2 I),
matching the published example defaults. This is a joint, not marginal, objective.
"""
from eacf.flow.build_flow import (BaseConfig,ConditionalAuxDistConfig,FlowDistConfig,build_flow)
from eacf.nets.make_egnn import NetsConfig,MLPHeadConfig,EGNNTorsoConfig
from eacf.utils.optimize import OptimizerConfig,get_optimizer

UPSTREAM_COMMIT='beafab1b1ccd2b770572daeef1cf15f3fe199c21'


def recipe_for(nodes,*,engineering=False):
    return {'upstream_commit':UPSTREAM_COMMIT,'nodes':int(nodes),'dim':3,'n_aug':1,
        'n_layers':1 if engineering else 12,'n_blocks':1 if engineering else 3,
        'mlp_units':[4] if engineering else [64,64],
        'n_invariant_feat_hidden':4 if engineering else 128,
        'embedding_dim':32,'num_discrete_feat':119,'spline_num_bins':8,
        'type':'spherical','reflection_invariant':True,'n_inner_transforms':1,
        'dist_spline_max':10.,'identity_init':True,'scaling_layer':False,
        'aux_conditioned_on_x':True,'aux_scale':.1,'aux_regularizer_weight':1.,
        'engineering_only':bool(engineering)}


def build_reference_flow(recipe):
    if recipe['upstream_commit']!=UPSTREAM_COMMIT:raise ValueError('Wrong upstream source')
    nets=NetsConfig(type='egnn',embedding_dim=recipe['embedding_dim'],num_discrete_feat=recipe['num_discrete_feat'],
        egnn_torso_config=EGNNTorsoConfig(name='egnn',n_blocks=recipe['n_blocks'],
            mlp_units=tuple(recipe['mlp_units']),n_invariant_feat_hidden=recipe['n_invariant_feat_hidden'],
            cross_multiplicity_shifts=True),
        mlp_head_config=MLPHeadConfig(tuple(recipe['mlp_units']),stable=True))
    auxiliary=ConditionalAuxDistConfig(conditioned_on_x=recipe['aux_conditioned_on_x'],
        trainable_augmented_scale=False,scale_init=recipe['aux_scale'])
    config=FlowDistConfig(dim=3,n_aug=1,nodes=recipe['nodes'],n_layers=recipe['n_layers'],nets_config=nets,
        type=recipe['type'],identity_init=recipe['identity_init'],scaling_layer=recipe['scaling_layer'],
        scaling_layer_conditioned=False,base=BaseConfig(aug=auxiliary),target_aux_config=auxiliary,
        kwargs={'spherical':{key:recipe[key] for key in
            ['reflection_invariant','n_inner_transforms','dist_spline_max','spline_num_bins']}})
    return build_flow(config)


def build_reference_optimizer(steps):
    config=OptimizerConfig(init_lr=2e-5,optimizer_name='adam',use_schedule=True,
        n_iter_total=steps,n_iter_warmup=min(30,steps-1),peak_lr=2e-4,end_lr=2e-5,
        dynamic_grad_ignore_and_clip=True,dynamic_grad_ignore_factor=10.,
        dynamic_grad_norm_factor=2.,dynamic_grad_norm_window=100)
    return get_optimizer(config)
