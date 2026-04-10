#!/bin/bash

#==============================================================
# Audio-JEPA demo training (1x full H100, 1 hour, resumable)
#==============================================================
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=ajepa-demo
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
set -x

cd /scratch/lovenya/Audio-JEPA
mkdir -p logs/slurm

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TMPDIR="${SCRATCH:-/tmp}"
export MPLCONFIGDIR="${SCRATCH:-/tmp}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

module load StdEnv/2023 cuda
source ajepa_env/bin/activate

RUN_NAME="audiojepa_demo_1h"
RUN_DIR="logs/train/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

CKPT_ARG=()
if [[ -f "${RUN_DIR}/checkpoints/last.ckpt" ]]; then
  CKPT_ARG=("ckpt_path=${RUN_DIR}/checkpoints/last.ckpt")
fi

echo "Node: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-N/A}"
echo "Resuming: ${CKPT_ARG[*]:-fresh run}"

srun python -u src/train.py \
  paths=default_linux \
  paths.data_dir="/scratch/lovenya/datasets" \
  data=audioset \
  trainer=gpu \
  trainer.devices=1 \
  trainer.max_steps=2000 \
  logger=wandb \
  logger.wandb.offline=True \
  data.batch_size=64 \
  +model.encoder.use_flash_attn=False \
  +model.predictor.use_flash_attn=False \
  extras.enforce_tags=False \
  tags='[demo,1h,h100]' \
  hydra.run.dir="${RUN_DIR}" \
  callbacks.model_checkpoint.save_weights_only=true \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_train_steps=200 \
  "${CKPT_ARG[@]}"
