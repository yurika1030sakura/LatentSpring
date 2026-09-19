"""Generate raw coordinates with a saved LatentSpring physical-correction model.

Use the flowmol environment and the checkpoint/config paths recorded in
configs/research/latentspring_force_correction_v1.json. This entry point performs
no energy evaluation, geometry optimization, or output filtering.
"""
from scripts.research.generate_weighted_minima import main

if __name__=='__main__':
    main()
