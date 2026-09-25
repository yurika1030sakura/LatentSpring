"""Export exact inference tensors for the final models, baselines and controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();root = a.project.resolve();a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = dict(format='latentspring_release_v2', upstream='../vendor/edm', files={},
        models={}, source_records={}, default_model='latentspring_s0',
        settings=dict(backbone_calls=128, batch_size=8, geometry_strength=1., physical_strength=4.,
                      charge=0, spin_multiplicity=1, max_atoms=200, main_hydrogen_readout=False))

    def save(name, payload, source, source_hash=None):
        if source_hash is not None:assert sha(source)==source_hash, source
        payload = dict(payload)
        payload['state_dict'] = {k:v.detach().cpu().contiguous() for k,v in payload['state_dict'].items()}
        calculated = state_hash(payload['state_dict'])
        if 'state_sha256' in payload:assert payload['state_sha256']==calculated, name
        payload['state_sha256'] = calculated
        if 'spec' in payload:
            payload['spec'] = dict(payload['spec'], upstream='../vendor/edm')
        path = a.out / name
        torch.save(payload, path)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        assert state_hash(loaded['state_dict'])==calculated
        manifest['files'][name] = sha(path)
        manifest['source_records'][name] = dict(source_checkpoint=source.name,
            source_checkpoint_sha256=sha(source), tensor_state_sha256=calculated,
            bytes=path.stat().st_size)
        return name

    legacy = json.loads((a.legacy/'manifest.json').read_text())
    for name, checksum in legacy['files'].items():
        source = a.legacy/name;assert sha(source)==checksum
        payload = torch.load(source, map_location='cpu', weights_only=True)
        save(name, payload, source)
    for key, record in legacy['models'].items():
        manifest['models'][key] = dict(parent=record['parent'])
        manifest['models'][key.replace('_s','_physical_s')] = dict(
            parent=record['parent'], physical=record['head'], hydrogen=record['hydrogen'], physical_strength=4.)

    primary = json.loads((root/'research/evidence/geometry_primary_confirmation_v1.json').read_text())
    extra = json.loads((root/'research/evidence/geometry_candidate_replication_v1.json').read_text())
    fixtures = []
    def backbone(name, arm, expected=None):
        source = root/arm['checkpoint'];assert sha(source)==arm['checkpoint_sha256']
        saved = torch.load(source, map_location='cpu', weights_only=False)
        record = dict(spec=arm['spec'], state_dict=saved['ema_state_dict'])
        if expected is not None:record['state_sha256']=expected
        return save(name, record, source, arm['checkpoint_sha256'])
    for fit in range(5):
        if fit<2:
            arm=primary['parents_by_method']['full_geometry_physics'][fit]
            geometry=primary['geometry_heads'][fit];physical=primary['physical_heads'][fit]
            folder=root/f'runs/geometry_primary_confirmation_v1/s{fit}/full_geometry_physics/evaluation'
            seed=primary['evaluation_seeds'][fit]
        else:
            resolved=json.loads((root/f'runs/geometry_candidate_replication_v1/s{fit}/resolved_inference.json').read_text())
            arm=resolved['parent'];geometry=resolved['geometry'];physical=resolved['physical_head']
            folder=root/f'runs/geometry_candidate_replication_v1/s{fit}/evaluation'
            seed=extra['evaluation_seeds'][fit]
        done=json.loads((folder/'complete.json').read_text());assert done['complete']
        parent_file=backbone(f'latentspring_s{fit}.pt',arm,done['model_state_sha256'])
        source=root/geometry['path'];assert sha(source)==geometry['sha256']
        saved=torch.load(source,map_location='cpu',weights_only=False)
        geometry_file=save(f'geometry_s{fit}.pt',dict(configuration=saved['configuration'],
            state_dict=saved['ema_state_dict'],state_sha256=geometry['ema_state_sha256']),source)
        head_file=f'fm_head_s{fit}.pt'
        assert manifest['source_records'][head_file]['tensor_state_sha256']==done['head_state_sha256']
        assert sha(root/physical['path'])==physical['sha256']
        key=f'latentspring_s{fit}'
        manifest['models'][key]=dict(parent=parent_file,geometry=geometry_file,physical=head_file,
                                    geometry_strength=1.,physical_strength=4.)
        fixture_file=folder/'generation/full_geometry_physics_c0.pt'
        fixture=torch.load(fixture_file,map_location='cpu',weights_only=False)
        fixtures.append(dict(model=key,seed=seed,stream=0,source_sha256=sha(fixture_file),
                             numbers=fixture['condition']['numbers'],positions=fixture['positions'],
                             initial_positions=fixture['initial_positions']))
    for fit, arm in enumerate(primary['parents_by_method']['continued_physical']):
        name=backbone(f'continuation_s{fit}.pt',arm)
        manifest['models'][f'continuation_s{fit}']=dict(parent=name,physical=f'fm_head_s{fit}.pt',physical_strength=4.)
    transfer=json.loads((root/'research/evidence/geometric_correction_transfer_v1.json').read_text())
    for family, arms in transfer['target_parents'].items():
        for fit, arm in enumerate(arms):
            if family=='gaga':name=f'gaga_s{fit}.pt'
            else:name=backbone(f'{family}_s{fit}.pt',arm)
            manifest['models'][f'{family}_s{fit}']=dict(parent=name)
            key=f'{family}_transfer_s{fit}'
            manifest['models'][key]=dict(parent=name,geometry=f'geometry_s{fit}.pt',
                physical=f'fm_head_s{fit}.pt',geometry_strength=1.,physical_strength=4.)
            folder=root/f'runs/geometric_correction_transfer_v1/{family}/s{fit}/evaluation/generation'
            generation=json.loads((folder/'generation.json').read_text())
            record=next(row for row in generation['rows'] if row['condition_index']==0)
            file=folder/record['file'];assert sha(file)==record['sha256']
            fixture=torch.load(file,map_location='cpu',weights_only=False)
            fixtures.append(dict(model=key,seed=transfer['target_seeds'][family][fit],stream=0,
                source_sha256=sha(file),numbers=fixture['condition']['numbers'],positions=fixture['positions'],
                initial_positions=fixture['initial_positions']))
    for family in ['fm','gaga']:
        folder=root/'runs/hydrogen_physical_confirmation_v1/s0'
        parent_file=folder/f'parents/{family}/{family}_a0_c14.pt'
        output_file=folder/f'readouts/raw/{family}_molecule_start0_c14.pt'
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        output=torch.load(output_file,map_location='cpu',weights_only=False)
        fixtures.append(dict(model=f'{family}_physical_s0',seed=63051,stream=14,hydrogen=True,
            source_sha256=sha(output_file),numbers=parent['condition']['numbers'],
            positions=output['positions'],initial_positions=parent['initial_positions']))
    examples=a.out.parent/'examples';examples.mkdir(exist_ok=True)
    rows=[];arrays={}
    for index,item in enumerate(fixtures):
        for key in ['positions','initial_positions']:arrays[f'{index}_{key}']=item.pop(key).numpy()
        rows.append(item)
    np.savez_compressed(examples/'sampling_fixtures.npz',**arrays)
    (examples/'sampling_fixtures.json').write_text(json.dumps(rows,indent=2)+'\n')
    panel=json.loads((root/'research/evidence/organic_geometry_panel_v1.json').read_text())
    condition=panel['conditions'][0]['condition']
    (examples/'condition.json').write_text(json.dumps({k:condition[k] for k in ['numbers','charge','spin_multiplicity']},indent=2)+'\n')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(models=len(manifest['models']),weight_files=len(manifest['files']),
        bytes=sum((a.out/n).stat().st_size for n in manifest['files']),fixtures=len(fixtures)),indent=2),flush=True)


if __name__=='__main__':main()
