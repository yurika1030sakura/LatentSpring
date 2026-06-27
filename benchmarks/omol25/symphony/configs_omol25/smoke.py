"""Tiny OMol25 smoke config for Symphony (E3SchNet backbone, bond-free).
A few training steps to validate the OMol25 I/O + fragment pipeline + model."""
import ml_collections
from configs.qm9 import e3schnet


def get_config() -> ml_collections.ConfigDict:
    config = e3schnet.get_config()

    # Dataset -> OMol25 (83 elements, bond-free). root_dir holds omol25_symphony.npz.
    config.dataset = "omol25"
    config.root_dir = "/home/renhaozhang_umass_edu/scratch_workspace/bgfm/processed_data/omol25_smoke"
    config.use_edm_splits = False
    config.num_train_molecules = 30
    config.num_val_molecules = 8
    config.num_test_molecules = 8
    config.heavy_first = False
    config.transition_first = False

    # Tiny training; eval/generate off for the smoke.
    config.num_train_steps = 3
    config.log_every_steps = 1
    config.eval = False
    config.eval_during_training = False
    config.generate = False
    config.generate_during_training = False
    config.num_eval_steps = 2
    config.eval_every_steps = 1000
    config.generate_every_steps = 1000

    # FIXED padding (not dynamic): variable shapes retrace the jitted train/eval
    # steps and trip chex's assert_max_traces. Budgets must hold the densest
    # single OMol25 molecule (~200 atoms can be near-fully-connected at 5 A).
    config.compute_padding_dynamically = False
    config.max_n_graphs = 2
    config.max_n_nodes = 512
    config.max_n_edges = 50000

    # OMol25 has focus->target distances > 3 A (metal-ligand, longer bonds), which
    # overflow the radial spline's default max_radius=3.0 -> inf radial/position
    # loss. Widen it to cover the 5 A neighbour cutoff.
    config.target_position_predictor.radial_predictor.max_radius = 6.0
    config.radial_cutoff = 5.0
    return config
