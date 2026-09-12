import copy
import json
from pathlib import Path

from scripts.research.freeze_proposal_training_panel import select


def test_training_selection_is_energy_blind_disjoint_and_reproducible():
    root=Path(__file__).resolve().parents[1]/'research/evidence'
    candidates=json.loads((root/'official_development_candidates.json').read_text())['rows']
    panels=[json.loads((root/p).read_text()) for p in ['development_panel_v1.json','transfer_development_panel_v1.json']]
    excluded={r['composition_hex'] for panel in panels for r in panel['rows']}
    expected=json.loads((root/'proposal_training_panel_v1.json').read_text())
    rows,counts=select(candidates,excluded)
    assert rows==expected['rows'] and counts==expected['eligible_compositions_by_size']
    changed=copy.deepcopy(candidates)
    for i,row in enumerate(changed):row['energy_eV']=1e20*(-1 if i%2 else 1)
    assert select(changed,excluded)==(rows,counts)
    compositions={r['composition_hex'] for r in rows}
    assert len(compositions)==8 and not compositions&excluded
    assert all('energy_eV' not in r and r['partition']=='new_development' for r in rows)
    assert sum(8<=r['n_atoms']<=12 for r in rows)==4
    assert sum(13<=r['n_atoms']<=24 for r in rows)==4
