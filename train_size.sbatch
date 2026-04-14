#!/bin/bash
#SBATCH --job-name=ltd-train-size
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --time=24:00:00

set -euo pipefail

# Run in the directory where sbatch was submitted from.
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p logs checkpoints

# If your cluster needs modules, uncomment and adjust:
# module load cuda/12.1

# Activate your venv (change path if needed).
source /scratch/yf3005/ea_env/bin/activate

# Disable online W&B logging in batch jobs without API key.
export WANDB_MODE=offline

# Optionally pick one GPU index on node; usually SLURM sets this automatically.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Model Paths
base_model_path="meta-llama/Llama-3.1-8B-Instruct"
ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
rl_token_model_path="" # Leave empty to use default 60 tokens
rl_checkpoint_path=""  # Leave empty to start from scratch

# Data and Save Directories
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints"

# RL Hyperparameters
total_timesteps=100000
batch_size=256
n_steps=2048
lr=1e-3
gamma=0.9
n_epochs=20
ent_coef=0.01

python3 -m rl.rl_total \
    --base_model_path ${base_model_path} \
    --ea_model_path ${ea_model_path} \
    --rl_token_model_path "${rl_token_model_path}" \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir ${data_dir} \
    --dataset_train ${dataset_train} \
    --save_path ${save_path} \
    --total_timesteps ${total_timesteps} \
    --batch_size ${batch_size} \
    --n_steps ${n_steps} \
    --lr ${lr} \
    --gamma ${gamma} \
    --n_epochs ${n_epochs} \
    --ent_coef ${ent_coef} \
    --pi_arch 1024 256 \
    --vf_arch 1024 256
