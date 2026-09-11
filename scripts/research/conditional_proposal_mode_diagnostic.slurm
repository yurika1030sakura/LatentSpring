#!/bin/bash
#SBATCH -J bgfm-mode-diagnostic
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
exec bash "$SOURCE_ROOT/scripts/research/conditional_proposal_toy.slurm" "$SOURCE_ROOT" "$RUN_ROOT" mode_jump
