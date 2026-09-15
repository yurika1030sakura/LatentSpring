#!/usr/bin/env python3
"""Show deterministic raw examples of distinct inferred chemical connectivities."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cfm_mol.chemical_moves import infer_chemical_graph
from scripts.research.molecular_figure_tools import draw_molecule,bonds_from_graph
from scripts.research.train_electronic_fm import sha


def main():
    root=Path(__file__).resolve().parents[2];torch.set_num_threads(1)
    trajectory=root/'runs/editorial_review_20260915/trajectory/manifest.json';manifest=json.loads(trajectory.read_text())
    sample=root/manifest['original_sample'];assert sha(sample)==manifest['original_sample_sha256']
    data=torch.load(sample,weights_only=False,map_location='cpu');c=data['condition']
    reportfile=sample.parent/'escort_delta_results.json';report=json.loads(reportfile.read_text());row=report['rows'][manifest['condition_index']]
    assert row['sample_sha256']==sha(sample);chosen=[];seen=set()
    for index,item in enumerate(row['records']):
        if item['graph_supported'] and item['smiles'] not in seen:
            seen.add(item['smiles']);chosen.append(index)
        if len(chosen)==4:break
    assert len(chosen)==4
    out=root/'research/figures/raw_molecular_gallery_v1';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig=plt.figure(figsize=(7.4,2.1),facecolor='white');records=[]
    for order,index in enumerate(chosen):
        graph=infer_chemical_graph(data['positions'][index],c['numbers'],c['charge']);bonds=bonds_from_graph(graph)
        rank=len(bonds)-c['n_atoms']+1
        ax=fig.add_axes([.02+order*.245,.17,.225,.70]);draw_molecule(ax,data['positions'][index].numpy(),c['numbers'],bonds=bonds,radius_scale=1.1)
        ax.set_title(f'Example {order+1}',fontsize=8.5,pad=1,color='#536277')
        fig.text(.132+order*.245,.08,f'{rank} independent cycles',ha='center',fontsize=8,color='#536277')
        records.append(dict(sample_index=index,connectivity_smiles=graph['connectivity_smiles'],inferred_edges=len(bonds),independent_cycles=rank))
    fig.text(.5,.95,r'One composition $\mathrm{C_{10}H_9N_3O_5}$, multiple generated connectivities',ha='center',fontsize=10,weight='bold',color='#263344')
    for suffix in ['pdf','svg','png']:fig.savefig(out/f'gallery.{suffix}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    provenance=dict(sample_sha256=sha(sample),report_sha256=sha(reportfile),trajectory_manifest_sha256=sha(trajectory),
        condition=c,examples=records,selection='First four graph-valid, distinct-connectivity draws in the first CHON composition with valid outputs; no energy ranking.',
        cycle_definition='For a connected inferred graph, independent cycle count is edge count minus atom count plus one; multiple bond order does not multiply edge count.',
        coordinates='Original raw coordinates, centered and rigidly rotated for display only.',new_neural_outputs=0,new_physical_queries=0)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
