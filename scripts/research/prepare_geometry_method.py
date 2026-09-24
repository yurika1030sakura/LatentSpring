"""Prepare a chemically checked raw example, a true source, and a 2D connectivity inset."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors,rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from ase.data import chemical_symbols
from cfm_mol.chemical_geometry_review import molecule_from_coordinates,geometry_diagnostics
from cfm_mol import matched_egnn as base
from cfm_mol.tree_prior_controls import cayley_tree,edge_embedding
from scripts.research.molecular_figure_tools import camera


def main():
    root=Path(__file__).resolve().parents[2];folder=root/'runs/organic_geometry_panel_v1'
    review=json.loads((folder/'uncharged_illustration_candidates.json').read_text())
    eligible=review['rows'];assert eligible,'No checked uncharged raw example'
    row=sorted(eligible,key=lambda v:(v['fit'],v['condition'],v['sample']))[0];path=root/row['file'];saved=torch.load(path,map_location='cpu',weights_only=False);c=saved['condition'];j=row['sample'];x=saved['positions'][j].numpy();mol=molecule_from_coordinates(x,c['numbers'],0);check=geometry_diagnostics(mol)
    assert check['closed_shell_geometry_pass'] and check['charged_atoms']==0
    source=base.HarmonicSource();source.sample(c['numbers'],np.random.default_rng(0));u,scale=source.cache[tuple(c['numbers'])]
    protocol=json.loads((root/'research/evidence/organic_geometry_panel_v1.json').read_text())
    rng=np.random.default_rng(protocol['evaluation_seeds'][row['fit']]*1000003+row['condition']*100003+(j//8)*8)
    for _ in range(j%8+1):
        edges=cayley_tree(u,rng);std=np.array([scale[a,b] for a,b in edges]);x0=torch.from_numpy(edge_embedding(len(x),edges)@(std[:,None]*rng.normal(size=(len(x)-1,3)))).float().double()
    torch.testing.assert_close(x0,saved['initial_positions'][j],atol=0,rtol=0)
    kek=Chem.Mol(mol);Chem.Kekulize(kek,clearAromaticFlags=True);bonds=[[b.GetBeginAtomIdx(),b.GetEndAtomIdx(),b.GetBondTypeAsDouble()] for b in kek.GetBonds()]
    rot=camera(x);origin=x.mean(0);symbols=[chemical_symbols[z] for z in c['numbers']]
    out=root/'research/figures/geometric_method_v1';out.mkdir(parents=True,exist_ok=False)
    scenes=[dict(name='source',camera_group='generation',symbols=symbols,positions=((x0.numpy()-origin)@rot).tolist(),bonds=[],scaffold=edges),dict(name='output',camera_group='generation',symbols=symbols,positions=((x-origin)@rot).tolist(),bonds=bonds)]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(scenes=scenes,generation_record=str(path.relative_to(root)),generation_record_sha256=sha(path),fit=row['fit'],condition_index=row['condition'],sample_index=j,formula=rdMolDescriptors.CalcMolFormula(mol),smiles=check['smiles'],force_rms_eV_A=row['force'],geometry_checks=check,source_rng_replayed_exactly=True,geometry_optimized=False,
        source_edges='Replayed auxiliary tree; dashed virtual springs',output_edges='Kekule representation of the coordinate-inferred RDKit graph; same connectivity and charges',
        example_scope='Raw output of the final full generator on the12-composition common-element validation application panel. Separate from the64-composition stress benchmark.',
        selection='Illustration only: first output in fit/condition/sample order passing the declared geometry and force screens with zero assigned atom charges. No aromatic or ring requirement, no coordinate optimization or removal of outputs.')
    (out/'scenes.json').write_text(json.dumps(receipt,indent=2)+'\n')
    Chem.MolToMolFile(mol,str(out/'raw_output.sdf'));np.savetxt(out/'raw_output_coordinates.txt',x,fmt='%.16g')
    picture=Chem.RemoveHs(Chem.Mol(mol));Chem.RemoveStereochemistry(picture);picture.RemoveAllConformers();rdDepictor.Compute2DCoords(picture)
    for ext in ['svg','png']:
        draw=rdMolDraw2D.MolDraw2DSVG(520,350) if ext=='svg' else rdMolDraw2D.MolDraw2DCairo(520,350)
        opts=draw.drawOptions();opts.bondLineWidth=3.;opts.fixedFontSize=52;opts.clearBackground=False;opts.padding=.06
        draw.DrawMolecule(picture);draw.FinishDrawing();value=draw.GetDrawingText()
        (out/('connectivity.'+ext)).write_text(value) if isinstance(value,str) else (out/('connectivity.'+ext)).write_bytes(value)
    print(json.dumps({k:receipt[k] for k in ['formula','smiles','condition_index','sample_index','force_rms_eV_A','example_scope']},indent=2))


if __name__=='__main__':main()
