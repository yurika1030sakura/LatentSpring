"""Strict continuation of completed finite-work experiments without changing target."""

INVARIANTS=('mode','batch','path_steps','lr','noise','kT','restraint','seed',
    'reference_kernel','mean_parameterization','neutralize_temperature_input',
    'noise_annealing_power','prior_std','max_drift_per_sqrt_dimension')
DEFAULTS={'objective':'mean_work','gradient_diagnostics':0,'noise_annealing_power':0.,
    'mean_parameterization':'reference','neutralize_temperature_input':False}


def validate_resume_recipe(report,checkpoint,requested):
    if not report.get('complete') or report.get('gradient_diagnostics'):
        raise ValueError('Resume requires completed training, not a gradient diagnostic')
    old=report['configuration']
    if checkpoint['proposal_protocol']!=old:raise ValueError('Source checkpoint/recipe mismatch')
    start=checkpoint.get('global_step')
    if not isinstance(start,int) or start<1 or start!=old['steps']:
        raise ValueError('Source global training step mismatch')
    if requested['steps']<=start:raise ValueError('Requested total steps must exceed completed source steps')
    for key in (*INVARIANTS,'objective'):
        if old.get(key,DEFAULTS.get(key))!=requested.get(key,DEFAULTS.get(key)):
            raise ValueError(f'Resume changes fixed protocol: {key}')
    if requested.get('gradient_diagnostics',0):raise ValueError('Resume training and gradient diagnostics must be separate')
    return start


def restore_work_optimizer(optimizer,state,start):
    optimizer.load_state_dict(state)
    steps=[int(value['step']) for value in optimizer.state.values() if 'step' in value]
    if not steps or any(step!=start for step in steps):
        raise ValueError('Optimizer update counts do not match the resumed global step')
