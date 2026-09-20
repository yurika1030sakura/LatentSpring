# Conditional hydrogen flow: model and evidence

This optional readout is established on the shared EGNN FM and GAGA models.
The main pretrained FlowMol model remains a separate recipe; transfer of this
readout to it has not been evaluated.

The frozen recipe is `configs/research/latentspring_hydrogen_flow_v1.json`.
It points to the parent, physical-head, and hydrogen-flow checkpoints with hashes.
Given atomic numbers, charge, and spin, generation returns every coordinate
sample. No energy calculation, bond graph, or output ranking is used. A molecule
with a detached hydrogen receives eight small EGNN calls; all of its hydrogens
may move, while relative heavy-atom positions remain fixed. The output file
contains both parent and completed coordinates and the activation mask.

In the FlowMol environment, from the project root:

```bash
python -m scripts.generate_hydrogen_flow --condition condition.json \
  --family fm --fit 0 --samples 16 --seed 42001 --out runs/my_hydrogen_samples
```

`condition.json` contains only, for example,
`{"numbers":[6,6,8,1,1,1,1,1,1],"charge":0,"spin_multiplicity":1}`.
The interface supports the checkpoint's element vocabulary and at most 200
atoms. The confirmed readout results cover neutral organic singlets with 17--40
atoms; support by the interface does not establish performance at other sizes.

Saved parent coordinates can also be processed without a GPU:

```bash
python -m scripts.generate_hydrogen_flow \
  --replay-parent runs/hydrogen_physical_confirmation_v1/s0/parents/fm/fm_a0_c14.pt \
  --device cpu --fit 0 --out runs/my_hydrogen_replay
```

`samples.pt` retains full provenance, `samples.xyz` contains every output, and
`manifest.json` reports actual calls, times, coordinate hashes, and zero oracle
queries. The batch layout is part of the random-seed convention.

## Evidence and interpretation

`hydrogen_physical_confirmation_audit_v1.json` confirms the independently frozen
physical hypothesis on 30 fresh compositions. FM's learned readout lowers mean
all-output GFN2 energy by 15.21 meV/atom relative to the radial rule, with a
95% composition interval [10.10,21.71]. Their graph-validity difference is only
0.10 points [-0.21,0.42]. Against the unchanged parent, graph validity rises
25.31% to27.40% and joint graph/force yield24.17% to26.04%.

Moving all hydrogens instead of just detached ones lowers energy by a further
10.25 meV/atom with identical validity. This is the direct ablation supporting
the concerted readout. The same decoder also improves GAGA's energy. Comparing
equally augmented generators gives FM+3.33 graph-validity points [0.52,6.15],
while joint-yield superiority remains uncertain (+2.29 [-0.63,5.21]).

The original yield-only development screen stays failed. A separate physical
hypothesis was frozen before the fresh test. Hydrogen decoration is established
prior work (Quetzal, arXiv:2505.13791); this experiment supports a task-specific
conditional readout and its physical benefit, not a first hydrogen-generation
task or global Boltzmann sampling.

The atom-normalized/mobility-head factorial follow-up also remains negative:
its independent confirmation did not establish the selected head's benefit.
It is not adopted in this recipe. Registries v32--v35 count all these studies,
including failed candidates, readouts, new oracle calculations, and training.
