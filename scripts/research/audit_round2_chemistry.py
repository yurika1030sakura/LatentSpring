"""Assess all archived main FM/GAGA outputs and identify auditable display examples."""
import argparse,json,hashlib,importlib.metadata
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from rdkit import Chem,rdBase
from cfm_mol.chemical_geometry_review import molecule_from_coordinates,geometry_diagnostics,contact_diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project.resolve();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);records=[];examples=[];proof={};metrics=json.loads((root/'research/evidence/seed_replication_audit_v2.json').read_text());arrays=dict(np.load(root/'research/evidence/seed_replication_audit_v2.npz'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    with (a.out/'records.jsonl').open('w') as handle:
        for fit in range(5):
            for family in ['fm','gaga']:
                folder=root/f'runs/seed_replication_v1/evaluation/s{fit}/parents/{family}';report=json.loads((folder/'generation.json').read_text());proof[str((folder/'generation.json').relative_to(root))]=sha(folder/'generation.json')
                for r in report['rows']:
                    arm=int(r['method'][-1]);name=family+('_parent' if arm==0 else '_physical');mi=metrics['methods'].index(name);ci=r['condition_index'];file=folder/r['file'];assert sha(file)==r['sha256']
                    saved=torch.load(file,map_location='cpu',weights_only=False);condition=saved['condition']
                    for j,(x,original) in enumerate(zip(saved['positions'].numpy(),r['records'])):
                        row=dict(method=name,fit=fit,condition=ci,sample=j,graph=bool(original['graph_supported']),force=float(arrays['force'][fit,mi,ci,j]),record=str(file.relative_to(root)),record_sha256=r['sha256'],**contact_diagnostics(x,condition['numbers']))
                        if row['graph']:
                            try:
                                mol=molecule_from_coordinates(x,condition['numbers'],condition['charge']);d=geometry_diagnostics(mol);row.update(d)
                                assert d['smiles']==original['smiles']
                                if name=='fm_physical' and d['closed_shell_geometry_pass'] and row['force']<=5 and len(x)<=32 and set(condition['numbers'])<=set([1,6,7,8,9,16,17]) and d['charged_atoms']==0 and d['ring_sizes'] and min(d['ring_sizes'])>=5 and max(d['ring_sizes'])<=7 and d['small_sp_rings']==0:
                                    candidates=dict(row,n_atoms=len(x),numbers=condition['numbers'],selection_key=hashlib.sha256((row['record']+':'+str(j)).encode()).hexdigest());examples.append(candidates)
                            except Exception as exc:row.update(closed_shell_geometry_pass=False,diagnostic_error=type(exc).__name__+': '+str(exc))
                        else:row['closed_shell_geometry_pass']=False
                        records.append(row);handle.write(json.dumps(row)+'\n')
                handle.flush();print(json.dumps(dict(fit=fit,family=family,processed=len(records),display_candidates=len(examples))),flush=True)
    references=[];data=torch.load(root/'runs/matched_generators_v1/data_v2/data.pt',map_location='cpu',weights_only=False)
    for i,r in enumerate(data['validation']):
        c=r['condition'];x=r['positions'].numpy();row=dict(index=i,**contact_diagnostics(x,c['numbers']))
        try:row.update(graph=True,**geometry_diagnostics(molecule_from_coordinates(x,c['numbers'],c['charge'])))
        except Exception as exc:row.update(graph=False,closed_shell_geometry_pass=False,error=str(exc))
        references.append(row)
    old=json.loads((root/'research/figures/endpoint_method_v1/scenes.json').read_text());old_data=torch.load(root/old['generation_record'],map_location='cpu',weights_only=False);j=old['generation_sample_index']
    old_review=geometry_diagnostics(molecule_from_coordinates(old_data['positions'][j].numpy(),old_data['condition']['numbers'],0))
    summary={}
    for name in ['fm_parent','fm_physical','gaga_parent','gaga_physical']:
        rows=[r for r in records if r['method']==name];den=len(rows);assert den==5120
        summary[name]=dict(attempted=den,graph=sum(r['graph'] for r in rows),closed_shell_geometry=sum(r['closed_shell_geometry_pass'] for r in rows),
            closed_shell_geometry_force5=sum(r['closed_shell_geometry_pass'] and r['force']<=5 for r in rows),
            disconnected=sum(r['contact_components']>1 for r in rows),heavy_disconnected=sum(r['heavy_components']>1 for r in rows),
            hydrogen_only_fragmentation=sum(r['hydrogen_only_fragmentation'] for r in rows),overlap=sum(r['severe_overlap'] for r in rows),
            graph_valid_diagnostic_failures=dict(Counter(k for r in rows if r['graph'] for k in ['bond_lengths_within_bounds','bond_angles_within_bounds','no_internal_clash','planar_groups_pass','nonaromatic_rings_nonflat'] if not r.get(k,False))))
    examples.sort(key=lambda r:r['selection_key'])
    report=dict(complete=True,posebusters_version=importlib.metadata.version('posebusters'),rdkit_version=rdBase.rdkitVersion,summary=summary,
        references=dict(attempted=len(references),graph=sum(r['graph'] for r in references),closed_shell_geometry=sum(r['closed_shell_geometry_pass'] for r in references)),
        old_method_figure_example=old_review,display_candidates=examples,records_sha256=sha(a.out/'records.jsonl'),provenance=proof,
        diagnostic_definition='Graph validation followed by zero assigned radicals and PoseBusters0.4.4 mol-config geometry/flatness checks. No internal-energy ensemble, InChI or reference-identity test: not a full PB-valid certification. Existing primary metrics unchanged. Every original attempt remains in denominators.',
        example_rule='First hash-ranked eligible raw corrected-FM example with17--32 atoms, common elements, uncharged atoms, a5--7-membered ring and no small sp ring, passing geometry checks and force<=5. This selects an illustration, not benchmark outputs.',new_energy_queries=0,geometry_optimization=False)
    (a.out/'references.json').write_text(json.dumps(references,indent=2)+'\n');(a.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
