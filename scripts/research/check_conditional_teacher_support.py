#!/usr/bin/env python3
"""Geometry-only check of full-sphere local teachers on all new training contexts."""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.spherical_proposal import vmf_sample
from scripts.research.evaluate_chemical_policy import sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/conditional_teacher_support_protocol_v1.json';protocol=json.loads(pp.read_text())
    ap=args.project/'runs/multicomposition_angular_audit_v1/results.json';audit=json.loads(ap.read_text())
    assert audit['complete'] and sha(ap)==protocol['probe_audit_sha256']
    rows=[];traces=[]
    for ar in audit['rows']:
        index=ar['index'];directory=args.project/f'runs/multicomposition_angular_probe_v1/condition_{index:02d}'
        report=json.loads((directory/'results.json').read_text());assert sha(directory/'results.json')==ar['results_sha256']
        if ar['zero_support']:continue
        assert sha(directory/'trace.pt')==ar['trace_sha256']
        saved=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        target=ChemicalTarget(None,report['condition'],.025851999786435,.1)
        for context,diag in zip(saved['contexts'],report['diagnostics']):
            if not diag['full_rank']:continue
            state=saved['states'][context['center_state_id']];leaf,anchor=context['root'];x=state['positions']
            radius=(x[leaf]-x[anchor]).norm()
            site=torch.tensor(diag['physical_site_parameter'],dtype=torch.float64)
            harmonic=torch.tensor(diag['confinement_parameter'],dtype=torch.float64)
            params={'fitted':torch.tensor(diag['fitted_parameter'],dtype=torch.float64),
                    'site64':site,'site64_confinement':site+harmonic}
            for method_number,method in enumerate(protocol['methods']):
                seed=protocol['seed']+100000000*index+100000*context['context']+100*method_number
                rng=torch.Generator().manual_seed(seed)
                directions,random=vmf_sample(params[method][None].expand(protocol['draws_per_context'],-1),generator=rng)
                record=dict(condition=index,context=context['context'],parent=context['parent_id'],method=method,
                    negative_center_axial_parameter=diag['center_axial_parameter']<0,attempts=len(directions),valid=0,validator_errors=0)
                outcomes=[]
                for direction in directions:
                    y=x.clone();y[leaf]=y[anchor]+radius*direction;y-=y.mean(0)
                    outcome=dict(valid=False)
                    try:
                        candidate=target.coordinate_state(y)
                        if not torch.equal(candidate['graph']['bond_orders'],state['graph']['bond_orders']):
                            raise ValueError('Different conditioning graph')
                        outcome['valid']=True;record['valid']+=1
                    except (ValueError,IndexError,RuntimeError) as exc:
                        outcome.update(error_type=type(exc).__name__,reason=str(exc));record['validator_errors']+=not isinstance(exc,ValueError)
                    outcomes.append(outcome)
                rows.append(record);traces.append(dict(seed=seed,condition=index,context=context['context'],method=method,
                    directions=directions,random=random,outcomes=outcomes,generator_state=rng.get_state()))
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists() or (args.out/'trace.pt').exists():raise FileExistsError(args.out)
    torch.save(traces,args.out/'trace.pt')
    summaries=[]
    for method in protocol['methods']:
        for group in ['all','negative_axial','nonnegative_axial']:
            subset=[r for r in rows if r['method']==method and (group=='all' or
                r['negative_center_axial_parameter']==(group=='negative_axial'))]
            summaries.append(dict(method=method,group=group,contexts=len(subset),attempts=sum(r['attempts'] for r in subset),
                valid=sum(r['valid'] for r in subset),validator_errors=sum(r['validator_errors'] for r in subset)))
    result=dict(complete=True,protocol_sha256=sha(pp),probe_audit_sha256=sha(ap),trace_sha256=sha(args.out/'trace.pt'),
        rows=rows,summaries=summaries,new_physical_queries=0,scientific_submission_ready=False,
        scope='Post hoc TRAINING-context structural-support diagnostic of local full-sphere teachers. Does not measure energy acceptance, neural learning, equilibrium or independent molecular validation.')
    write(args.out/'results.json',result);print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
