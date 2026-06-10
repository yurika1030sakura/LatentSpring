"""Training entry point.

TODO[Week 1-2]:
  - Week 1: run baselines/flowmol3/ unchanged on QM9 to reproduce their 99.9%
    validity number as our baseline.
  - Week 2: subclass FlowMol3's LightningModule to use
    `cfm_mol.flow.cfm_loss` for the coordinate channel (replacing FlowMol3's
    default coord loss), and add `cfm_mol.projection.project_valence` after
    each discrete step.

Config: configs/qm9_cfm.yaml (TODO, mirror FlowMol3 config + constraint settings).
"""
raise NotImplementedError("Week 1-2 work.")
