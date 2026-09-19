# Native weighted-endpoint pilot fixtures

These are four graph-global endpoint measures for three fixed compositions at
300 K under a restricted two-torsion MMFF94s target. The butanol and ether graphs
share one composition and have explicit equal outer mass. These coordinates are
restricted angular minima, not full Cartesian or DFT minima. Graph and basin
labels organize the target store; the neural model reads only composition.

`proposal_min/` preserves the uncorrected proposal frequencies.
`full_work_min/` preserves complete density/Jacobian-corrected thermal weights
after mapping to the same restricted minima. NPZ checksums are in each manifest.
Both are supplied teacher data, not newly generated physical results.

The actual native pilot is frozen in
`research/evidence/weighted_minima_native_v1.json`; outcomes are in
`weighted_minima_native_summary_v1.json`. Its original paths point into the
hashed supplied archive; these copies contain the same manifest and array bytes.

Run the native adapter in the existing FlowMol environment:

```bash
python -u -m scripts.research.train_weighted_minima \
  --checkpoint runs/source_sc_v1/s0/study/harmonic_tree/last.ckpt \
  --config configs/sweep/a1_fm_only_s2.yaml \
  --endpoints examples/weighted_endpoints/restricted_two_torsion/full_work_min \
  --out runs/new_minima_fit --device cuda --steps 500 --seed 260938 \
  --allow-legacy-pickle
```

The legacy-pickle switch is for this locally verified original checkpoint only.
The sampler saves every raw output; it uses no oracle or external optimizer.
