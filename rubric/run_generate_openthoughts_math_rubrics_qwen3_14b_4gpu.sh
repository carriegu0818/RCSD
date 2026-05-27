#!/bin/bash
#SBATCH --job-name=ot-math-rubrics
#SBATCH --output=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j.out
#SBATCH --error=/gpfs/radev/pi/ying_rex/sg2768/OPSD/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --exclude=r4513u10n01,r4513u20n01,r4513u30n01,r4516u01n01,r4518u01n01,r4518u09n01,r4518u05n01,r4507u05n01


set -euo pipefail

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
export XDG_CACHE_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache
export TORCH_HOME=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/gpfs/radev/pi/ying_rex/sg2768/.cache/torch/inductor

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

module load GCC/12.2.0

mkdir -p "${TMPDIR}"

echo "==== GPU DEBUG ===="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
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

DATASET_NAME="${DATASET_NAME:-siyanzhao/Openthoughts_math_30k_opsd}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"

RUBRIC_MODEL_PATH="${RUBRIC_MODEL_PATH:-Qwen/Qwen3-14B}"
RUBRIC_CACHE_DIR="${RUBRIC_CACHE_DIR:-/gpfs/radev/pi/ying_rex/sg2768/OPSD_runtime/outputs/rubric_cache/openthoughts_math_30k_qwen3_14b_rubrics}"
RUBRIC_WORK_DIR="${RUBRIC_WORK_DIR:-${RUBRIC_CACHE_DIR}_work}"
RUBRIC_SAMPLE_SIZE="${RUBRIC_SAMPLE_SIZE:-25000}"
RUBRIC_PROMPT_MAX_LENGTH="${RUBRIC_PROMPT_MAX_LENGTH:-8192}"
RUBRIC_MAX_NEW_TOKENS="${RUBRIC_MAX_NEW_TOKENS:-2048}"
RUBRIC_BATCH_SIZE="${RUBRIC_BATCH_SIZE:-8}"
RUBRIC_TEMPERATURE="${RUBRIC_TEMPERATURE:-0.6}"
RUBRIC_TOP_P="${RUBRIC_TOP_P:-0.95}"
RUBRIC_TOP_K="${RUBRIC_TOP_K:-20}"

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-10240}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"

MODEL_IS_LORA="${MODEL_IS_LORA:-0}"
REGENERATE_RUBRICS="${REGENERATE_RUBRICS:-0}"
RESUME_RUBRICS="${RESUME_RUBRICS:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
FILTER_INVALID="${FILTER_INVALID:-0}"
GUIDED_DECODING_REGEX="${GUIDED_DECODING_REGEX:-}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"

echo "DATASET_NAME=${DATASET_NAME}"
echo "DATASET_SPLIT=${DATASET_SPLIT}"
echo "RUBRIC_MODEL_PATH=${RUBRIC_MODEL_PATH}"
echo "RUBRIC_CACHE_DIR=${RUBRIC_CACHE_DIR}"
echo "RUBRIC_WORK_DIR=${RUBRIC_WORK_DIR}"
echo "RUBRIC_SAMPLE_SIZE=${RUBRIC_SAMPLE_SIZE}"
echo "RUBRIC_PROMPT_MAX_LENGTH=${RUBRIC_PROMPT_MAX_LENGTH}"
echo "RUBRIC_MAX_NEW_TOKENS=${RUBRIC_MAX_NEW_TOKENS}"
echo "RUBRIC_BATCH_SIZE=${RUBRIC_BATCH_SIZE}"
echo "RUBRIC_TEMPERATURE=${RUBRIC_TEMPERATURE}"
echo "RUBRIC_TOP_P=${RUBRIC_TOP_P}"
echo "RUBRIC_TOP_K=${RUBRIC_TOP_K}"
echo "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
echo "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "MODEL_IS_LORA=${MODEL_IS_LORA}"
echo "REGENERATE_RUBRICS=${REGENERATE_RUBRICS}"
echo "RESUME_RUBRICS=${RESUME_RUBRICS}"
echo "ENABLE_THINKING=${ENABLE_THINKING}"
echo "FILTER_INVALID=${FILTER_INVALID}"

CMD=(
    python generate_openthoughts_math_rubrics_vllm.py
    --dataset_name "${DATASET_NAME}"
    --split "${DATASET_SPLIT}"
    --rubric_model_path "${RUBRIC_MODEL_PATH}"
    --rubric_cache_dir "${RUBRIC_CACHE_DIR}"
    --work_dir "${RUBRIC_WORK_DIR}"
    --sample_size "${RUBRIC_SAMPLE_SIZE}"
    --prompt_max_length "${RUBRIC_PROMPT_MAX_LENGTH}"
    --max_new_tokens "${RUBRIC_MAX_NEW_TOKENS}"
    --batch_size "${RUBRIC_BATCH_SIZE}"
    --temperature "${RUBRIC_TEMPERATURE}"
    --top_p "${RUBRIC_TOP_P}"
    --top_k "${RUBRIC_TOP_K}"
    --tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max_model_len "${VLLM_MAX_MODEL_LEN}"
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
    --max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
)

if [[ -n "${RUBRIC_BASE_MODEL_NAME_OR_PATH:-}" ]]; then
    CMD+=(--rubric_base_model_name_or_path "${RUBRIC_BASE_MODEL_NAME_OR_PATH}")
fi
if [[ -n "${TOKENIZER_NAME_OR_PATH:-}" ]]; then
    CMD+=(--tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}")
fi
if [[ "${MODEL_IS_LORA}" == "1" ]]; then
    CMD+=(--model_is_lora)
fi
if [[ "${REGENERATE_RUBRICS}" == "1" ]]; then
    CMD+=(--regenerate)
fi
if [[ "${RESUME_RUBRICS}" == "1" ]]; then
    CMD+=(--resume)
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
    CMD+=(--enable_thinking)
else
    CMD+=(--disable_thinking)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    CMD+=(--trust_remote_code)
fi
if [[ "${FILTER_INVALID}" == "1" ]]; then
    CMD+=(--filter_invalid)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
    CMD+=(--disable_custom_all_reduce)
fi
if [[ -n "${GUIDED_DECODING_REGEX}" ]]; then
    CMD+=(--guided_decoding_regex "${GUIDED_DECODING_REGEX}")
fi

"${CMD[@]}"

RUBRIC_CACHE_DIR="${RUBRIC_CACHE_DIR}" python - <<'PY'
import os
from datasets import load_from_disk

cache_dir = os.environ["RUBRIC_CACHE_DIR"]
dataset = load_from_disk(cache_dir)
print(f"Validated rubric cache load from: {cache_dir}")
print(f"Rows: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
expected = ["problem", "reference_answer", "solution", "rubrics"]
missing = [name for name in expected if name not in dataset.column_names]
if missing:
    raise SystemExit(f"Generated cache is missing columns: {missing}")
if len(dataset) > 0 and not isinstance(dataset[0]["rubrics"], str):
    raise SystemExit("Generated cache 'rubrics' column is not string-valued.")
PY

echo "Finished OpenThoughts math rubric generation."
