#!/usr/bin/env python3
"""Assemble audited conditional work/force labels with frozen parent splits."""
import argparse
import json
from pathlib import Path
import torch

from cfm_mol.conditional_angular_probe import angular_force
from cfm_mol.masked_angular_guide import masked_angular_context
from scripts.research.audit_masked_angular import sha


def build(project,root):
    split_path=root/'research/evidence/multicomposition_angular_probe_protocol_v1.json'
    protocol=json.loads(split_path.read_text())
    physical=json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    audit_path=project/'runs/multicomposition_angular_audit_v1/results.json'
    audit=json.loads(audit_path.read_text());assert audit['complete']
    assert audit['protocol_sha256']==sha(split_path)
    records=[];provenance={str(audit_path.relative_to(project)):sha(audit_path)};counts={}
    for index in [0,1,2,3,5,7]:
        directory=project/f'runs/multicomposition_angular_probe_v1/condition_{index:02d}'
        header=json.loads((directory/'results.json').read_text())
        check=next(r for r in audit['rows'] if r['index']==index)
        assert check['results_sha256']==sha(directory/'results.json')
        assert check['trace_sha256']==sha(directory/'trace.pt')==header['trace_sha256']
        data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        split=protocol['condition_splits'][str(index)]
        endpoints={}
        for run in ['arc_oracle_feasibility','arc_internal_validation']:
            if run=='arc_oracle_feasibility' and index not in [1,2,3,5]:continue
            endpoint_dir=project/f'runs/{run}_v1/condition_{index:02d}'
            endpoint_audit=project/f'runs/{run}_audit_v1/condition_{index:02d}/results.json'
            ac=json.loads(endpoint_audit.read_text());report=json.loads((endpoint_dir/'results.json').read_text())
            assert ac['complete'] and ac['full_replay'] and report['complete']
            assert ac['results_sha256']==sha(endpoint_dir/'results.json')
            assert ac['trace_sha256']==sha(endpoint_dir/'trace.pt')==report['trace_sha256']
            saved=torch.load(endpoint_dir/'trace.pt',map_location='cpu',weights_only=False)
            provenance[str(endpoint_audit.relative_to(project))]=sha(endpoint_audit)
            for row in saved['rows']:
                key=row['context'],row['method'];assert key not in endpoints
                endpoints[key]=(row,saved['states'][row['new_state_id']],run)
        for context in data['contexts']:
            cid=context['context'];parent=context['parent_id'];root_pair=context['root']
            center=data['states'][context['center_state_id']]
            role='fit' if parent in split['fit_parent_ids'] else ('withheld_parent' if parent in split['withheld_parent_ids'] else 'withheld_composition')
            allowed=split['fit_parent_ids']+split['withheld_parent_ids']+split['withheld_composition_parent_ids']
            assert parent in allowed
            states=[('center',center)]
            states += [('local_'+p['role'],data['states'][p['state_id']]) for p in data['probes'] if p['context']==cid and p['valid']]
            for method in ['fitted','site64','site64_confinement','uniform']:
                row,state,run=endpoints[cid,method]
                assert row['parent']==parent and row['center_state_id']==context['center_state_id'] and row['root']==root_pair
                assert (run=='arc_oracle_feasibility')==(role=='fit')
                states.append(('arc_'+method,state))
            roots=torch.tensor([root_pair]);masked,radius,_=masked_angular_context(center['positions'][None],roots)
            directions=[];gradients=[];work=[];sample_roles=[]
            for name,state in states:
                other,r2,_=masked_angular_context(state['positions'][None],roots)
                torch.testing.assert_close(masked,other,atol=1e-8,rtol=1e-9)
                torch.testing.assert_close(radius,r2,atol=1e-8,rtol=1e-9)
                torch.testing.assert_close(state['graph']['bond_orders'],center['graph']['bond_orders'],atol=0,rtol=0)
                direction,score=angular_force(state,root_pair,physical['kT_eV'],physical['restraint_eV_A2'])
                directions.append(direction);gradients.append(-physical['kT_eV']*score)
                work.append(state['potential_eV']-center['potential_eV']);sample_roles.append(name)
            train_mask=torch.tensor([role=='fit' and name!='local_check' for name in sample_roles])
            assert (not train_mask.any())==(role!='fit')
            record=dict(index=index,context=cid,parent=parent,role=role,context_kind=context['kind'],
                positions=center['positions'],bonds=center['graph']['bond_orders'],numbers=torch.tensor(header['condition']['numbers']),
                electronic=torch.tensor([header['condition']['charge'],header['condition']['spin_multiplicity'],physical['kT_eV']],dtype=torch.float64),
                root=torch.tensor(root_pair),directions=torch.stack(directions),angular_energy_gradients_eV=torch.stack(gradients),
                work_eV=torch.stack(work),sample_roles=sample_roles,training_mask=train_mask)
            assert not record['training_mask'][torch.tensor([x=='local_check' for x in sample_roles])].any()
            records.append(record)
            counts[role]=counts.get(role,0)+1
    assert counts=={'withheld_composition':156,'fit':206,'withheld_parent':76}
    assert len(records)==438
    return records,dict(complete=True,records=counts,contexts=len(records),split_protocol_sha256=sha(split_path),
        source_audit_sha256=provenance,new_physical_queries=0,zero_support_conditions_retained=[4,6],
        reserved_outcomes_loaded=False,reference_coordinates_loaded=False,scientific_submission_ready=False,
        scope='Audited conditional potential differences and tangential derivatives at fixed passive geometry/radius. FIT local_check rows and all withheld parent/composition rows are never fitted.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    if args.out.exists():raise FileExistsError(args.out)
    records,report=build(args.project,root)
    args.out.mkdir(parents=True);torch.save(records,args.out/'data.pt')
    report['data_sha256']=sha(args.out/'data.pt')
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
