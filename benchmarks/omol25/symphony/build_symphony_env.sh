#!/bin/bash
# WORKING Symphony (JAX) env recipe on UMass Unity — CPU jax for smokes.
# The hard part is a coherent 2023-era stack: tensorflow==2.13 caps numpy<=1.24.3
# while modern jax needs numpy>=2, so pin numpy 1.24.3 + jax 0.4.23 and pin the
# transitive deps that drifted (scipy, ml-dtypes, orbax). Plus: e3nn-jax 0.20.1's
# wheel ships no module — use 0.20.7 (has SphericalSignal.replace_values that
# Symphony needs). biotite must be old (1.x needs numpy2). mace-jax needs
# flax.nnx (flax>=0.8) so its import is made optional in create_model.py.
set -eo pipefail
SCRATCH=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PREFIX=$SCRATCH/envs/symphony
export CONDA_PKGS_DIRS=$SCRATCH/.conda_pkgs
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -x "$PREFIX/bin/python" ] || conda create --prefix "$PREFIX" python=3.11 -y
PIP="$PREFIX/bin/pip"
"$PIP" install --upgrade pip
"$PIP" install "numpy==1.24.3" "jax[cpu]==0.4.23" ml_collections absl-py \
    "flax==0.7.5" "dm-haiku==0.0.11" jraph "clu==0.0.9" "optax==0.1.7" \
    "e3nn-jax==0.20.7" "chex==0.1.85" "distrax==0.1.5" "tensorflow==2.13.0" \
    "tensorflow-probability==0.21.0" ase rdkit pandas pyarrow \
    "scipy==1.11.4" "ml-dtypes==0.2.0" "orbax-checkpoint==0.4.4" \
    posebusters py3dmol sh wandb tqdm matscipy
"$PIP" install --no-deps "biotite==0.39.0"     # 1.x needs numpy2
"$PIP" install --no-deps \
    "git+https://github.com/mariogeiger/nequip-jax" \
    "git+https://github.com/mariogeiger/allegro-jax" \
    "git+https://github.com/ACEsuit/mace-jax"
cd /home/renhaozhang_umass_edu/bgfm/baselines/symphony
"$PIP" install -e . --no-deps
# GPU jax (for GPU training): jaxlib cuda12 + the MATCHING cuDNN 8.9 (jaxlib
# 0.4.23 is +cuda12.cudnn89; the cuda12_pip extra wrongly pulls cuDNN 9). At
# runtime set LD_LIBRARY_PATH to the pip nvidia libs + XLA_PYTHON_CLIENT_MEM_FRACTION
# (see scripts/unity/train_symphony.slurm). Skip this block for a CPU-only env.
"$PIP" install --no-cache-dir "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
"$PIP" install --no-cache-dir --no-deps "nvidia-cudnn-cu12==8.9.7.29" "numpy==1.24.3" "scipy==1.11.4" "ml-dtypes==0.2.0"
echo "verify:"
"$PREFIX/bin/python" -c "import symphony.train; from symphony.data.datasets import omol25; print('symphony OK | species', omol25.OMol25Dataset.get_atomic_numbers().shape[0])" 2>&1 | tail -1
