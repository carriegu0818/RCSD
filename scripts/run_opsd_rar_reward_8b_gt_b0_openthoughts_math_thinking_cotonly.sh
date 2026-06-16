#!/bin/bash
#SBATCH --job-name=opsd-reward-gt-b0-ot-math-thinking-cotonly
#SBATCH --output=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j.out
#SBATCH --error=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2         # request 2 GPUs
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --exclude=r4513u10n01,r4513u20n01,r4513u30n01,r4516u01n01,r4518u01n01,r4518u09n01,r4518u05n01,r4507u05n01


source /gpfs/radev/pi/ying_rex/sg2768/miniconda3/etc/profile.d/conda.sh

conda activate opsd

export CONDA_PKGS_DIRS=/gpfs/radev/pi/ying_rex/sg2768/.conda/pkgs
export CONDA_ENVS_PATH=/gpfs/radev/pi/ying_rex/sg2768/.conda/envs
export PIP_CACHE_DIR=/gpfs/radev/pi/ying_rex/sg2768/.cache/pip
export TMPDIR=/gpfs/radev/pi/ying_rex/sg2768/.tmp
export TEMP=/gpfs/radev/pi/ying_rex/sg2768/.tmp
export TMP=/gpfs/radev/pi/ying_rex/sg2768/.tmp
export HF_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache/huggingface
export HF_HUB_CACHE=/gpfs/radev/pi/ying_rex/sg2768/.cache/huggingface/hub
export HF_DATASETS_CACHE=/gpfs/radev/pi/ying_rex/sg2768/.cache/huggingface/datasets
export CUDA_HOME=/gpfs/radev/apps/avx512/software/CUDA/12.1.1
export WANDB_API_KEY=1b0ca0f0dc2e90d460addf9ccaeda83188ad001d


export XDG_CACHE_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache
export TORCH_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch/inductor

module load GCC/12.2.0

echo "==== GPU DEBUG ===="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nvidia-smi -L:"
nvidia-smi -L || true
python - <<'PY'
import torch
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}, cc {p.major}.{p.minor}, {p.total_memory/1e9:.1f} GB")
PY
echo "==================="


cd /gpfs/radev/pi/ying_rex/sg2768/OPSD

DATA_SOURCE="${DATA_SOURCE:-openthoughts_math}"
RUBRIC_SOURCE="${RUBRIC_SOURCE:-cache}"
# Controls what the teacher model receives: cot=False, rubric=True by default
TEACHER_COT="${TEACHER_COT:-true}"
TEACHER_RUBRIC="${TEACHER_RUBRIC:-false}"

# Build a tag reflecting what the teacher sees
TEACHER_TAG=""
if [ "${TEACHER_RUBRIC}" = "true" ] || [ "${TEACHER_RUBRIC}" = "True" ]; then
    TEACHER_TAG="${TEACHER_TAG}ru"
fi
if [ "${TEACHER_COT}" = "true" ] || [ "${TEACHER_COT}" = "True" ]; then
    TEACHER_TAG="${TEACHER_TAG}cot"
fi
if [ -z "${TEACHER_TAG}" ]; then
    TEACHER_TAG="none"
fi

RUN_TAG="${DATA_SOURCE//-/_}_${RUBRIC_SOURCE//-/_}"
RUN_CONFIG="qwen3_8b_reward_rfirst_lr5e6_4096_b0_${RUN_TAG}_25k_thinking_t${TEACHER_TAG}"
# Use the save_to_disk cache (not *_work, which only holds intermediate rubrics.jsonl).
# Cache is pre-filtered to drop empty/invalid rubrics (~22.8k valid examples).
RUBRIC_CACHE_DIR="${RUBRIC_CACHE_DIR:-/gpfs/radev/pi/ying_rex/sg2768/OPSD/rubric_cache/openthoughts_math_30k_qwen3_14b_rubrics}"
export RUBRIC_CACHE_DIR

echo "DATA_SOURCE=${DATA_SOURCE}"
echo "RUBRIC_SOURCE=${RUBRIC_SOURCE}"
echo "TEACHER_COT=${TEACHER_COT}"
echo "TEACHER_RUBRIC=${TEACHER_RUBRIC}"
echo "TEACHER_TAG=${TEACHER_TAG}"
echo "RUN_CONFIG=${RUN_CONFIG}"
echo "RUBRIC_CACHE_DIR=${RUBRIC_CACHE_DIR}"

python - <<'PY'
from datasets import load_from_disk
import os

cache_dir = os.environ["RUBRIC_CACHE_DIR"]
ds = load_from_disk(cache_dir)
print(f"RUBRIC_CACHE rows={len(ds)} columns={ds.column_names}")
PY

resume_args=()

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 16 \
    --main_process_port 11816 \
    opsd_train_reward.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16 \
    --output_dir /gpfs/radev/pi/ying_rex/sg2768/OPSD/outputs \
    --run_config "${RUN_CONFIG}" \
    --num_train_epochs 3 \
    --max_completion_length 4096 \
    --save_steps 20 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 25000 \
    --beta 0 \
    --use_peft \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --temperature 1.2 \
    --top_p 0.95 \
    --top_k 20 \
    --data_source "${DATA_SOURCE}" \
    --rubric_source "${RUBRIC_SOURCE}" \
    --rubric_cache_dir "${RUBRIC_CACHE_DIR}" \
    --rubric_sample_size 25000 \
    --student_thinking True \
    --teacher_thinking True \
    --teacher_solution_cot "${TEACHER_COT}" \
    --teacher_solution_rubric "${TEACHER_RUBRIC}" \
    --jsd_token_clip 0.05 \
    --reason_first \
    --fixed_teacher \
    --wandb_entity sgu33-stanford-university \
    --wandb_project OPSD \
    "${resume_args[@]}"
