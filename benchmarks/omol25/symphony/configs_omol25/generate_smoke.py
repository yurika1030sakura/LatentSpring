"""OMol25 Symphony smoke WITH generation enabled (tiny, fast) to exercise the
native generation -> validity path. Heavy speedups so it runs on CPU."""
import ml_collections
from configs.omol25 import smoke


def get_config() -> ml_collections.ConfigDict:
    c = smoke.get_config()
    c.num_train_steps = 1
    c.generate = True
    c.generate_during_training = True
    c.generate_every_steps = 1
    c.eval = False
    c.eval_during_training = False
    # tiny generation
    c.generation.num_seeds = 1
    c.generation.num_seeds_per_chunk = 1
    c.generation.max_num_atoms = 5
    c.generation.init_molecules = "C"
    c.generation.species = (1, 6, 7, 8)   # H,C,N,O subset for a fast smoke
    c.generation.posebusters = False
    c.generation.padding_mode = "fixed"
    # cheap angular sampling
    c.target_position_predictor.angular_predictor.sampling_num_steps = 4
    return c
