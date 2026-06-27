"""Real OMol25 training config for Symphony (E3SchNet, GPU). 50k-mol shard."""
import ml_collections
from configs.omol25 import smoke


def get_config() -> ml_collections.ConfigDict:
    c = smoke.get_config()
    c.num_train_molecules = 46816
    c.num_val_molecules = 2464
    c.num_test_molecules = 2464
    c.num_train_steps = 200_000
    c.log_every_steps = 500
    c.eval = True
    c.eval_during_training = True
    c.eval_every_steps = 20_000
    c.num_eval_steps = 100
    c.generate = True
    c.generate_during_training = True
    c.generate_every_steps = 50_000
    c.generation.num_seeds = 8
    c.generation.num_seeds_per_chunk = 4
    c.generation.max_num_atoms = 60
    c.generation.species = (1, 6, 7, 8, 9, 15, 16, 17)  # common organics for fast gen-eval
    c.generation.posebusters = False
    # Fixed padding (inherited compute_padding_dynamically=False). Budget holds
    # the densest single 200-atom molecule. Capped for the 2080ti via
    # XLA_PYTHON_CLIENT_MEM_FRACTION.
    c.max_n_graphs = 4
    c.max_n_nodes = 1000
    c.max_n_edges = 50000
    return c
